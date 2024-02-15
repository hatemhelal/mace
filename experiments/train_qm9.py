import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
from e3nn import o3
from jsonargparse import CLI
from torch_geometric import seed_everything
from torch_geometric.datasets.qm9 import QM9, atomrefs
from torch_geometric.transforms import BaseTransform
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from tqdm import tqdm

import wandb
from mace import modules, tools
from mace.data import AtomicData, Configuration
from mace.tools.torch_geometric import DataLoader
from mace.tools import compile

QM9_atomic_energies = np.array(atomrefs[7])
QM9_atomic_numbers = [1, 6, 7, 8, 9]


class ReduceLROnPlateauWithWarmup:
    def __init__(self, optimizer) -> None:
        self.warmup_scheduler = LinearLR(optimizer, start_factor=0.1)
        self.reduce_scheduler = ReduceLROnPlateau(optimizer, "min", patience=5)

    def step(self, metric):
        self.warmup_scheduler.step()
        self.reduce_scheduler.step(metric)


def create_scheduler(use_warmup, optimizer):
    if use_warmup:
        scheduler = ReduceLROnPlateauWithWarmup(optimizer)
    else:
        scheduler = ReduceLROnPlateau(optimizer, "min", patience=5)
    return scheduler


class QM9AtomicDataAdapter(BaseTransform):
    def __init__(self, cutoff: float):
        """
        Adapter to convert between the PyG data QM9 instance into mace.data.AtomicData

        Args:
            cutoff (float): the cutoff radius for the nearest neighbour graph edges
        """
        # target 7 corresponds to the internal energy at 0K (eV)
        self.target = 7
        self.z_table = tools.AtomicNumberTable([1, 6, 7, 8, 9])
        self.cutoff = cutoff

    def __call__(self, data):
        energy = float(data.y[:, self.target])
        config = Configuration(
            atomic_numbers=data.z.numpy(), positions=data.pos.numpy(), energy=energy
        )
        return AtomicData.from_config(config, z_table=self.z_table, cutoff=self.cutoff)


@dataclass
class ModelConfig:
    """
    MACE Model Configuration dataclass

    Args:
        num_interactions (int, optional): Number of interaction blocks in the MACE model
        max_ell (int, optional): Maximum l used in spherical harmonic expansion
        correlation (int, optional): correlation order
        num_channels (int, optional): number of embedding channels
        r_max (float, optional): the cutoff radius for nearest neighbours
        num_bessel (int, optional): Number of radial Bessel functions
        max_L (int, optional): max L equivariance of the message
    """

    num_interactions: int = 2
    max_ell: int = 3
    correlation: int = 3
    num_channels: int = 128
    r_max: float = 6.0
    num_bessel: int = 10
    max_L: int = 1


def train(
    seed: int = 1702,
    lr: float = 1e-3,
    use_warmup: bool = True,
    compile_mode: Optional[str] = "default",
    device: str = "cuda",
    batch_size: int = 64,
    data_root: str = "/data/qm9",
    num_train_examples: int = int(1e5),
    num_data_workers: int = 8,
    check_loader_perf: bool = True,
    max_num_epochs: int = 50,
    model_config: ModelConfig = ModelConfig(),
    use_wandb: bool = True,
):
    """Training MACE on QM9 dataset from PyTorch Geometric

    Args:
        seed (int, optional): Random number generator seed
        lr (float, optional): Optimizer learning rate
        use_warmup (bool, optional): enable initial linear learning rate warmup period
        compile_mode (Optional[str], optional): torch.compile mode
        device (str, optional): PyTorch device used
        batch_size (int, optional): Batch size used
        data_root (str, optional): Folder to download then load the QM9 dataset from.
        num_train_examples (int, optional): Subset of QM9 to use for training.
        num_data_workers (int, optional): Number of workers to use when loading data.
        check_loader_perf (int, optional): Benchmark data loader.
        max_num_epochs (int, optional): The maximum number of training epochs to run
        model_config (ModelConfig, optional): MACE architecture configuration options
        use_wandb (bool, optional): Enable wandb logging
    """
    args = locals()
    run = wandb.init(config=args) if use_wandb else None

    if device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")

    seed_everything(seed)
    dataset = QM9(data_root, transform=QM9AtomicDataAdapter(model_config.r_max))
    dataset = dataset.shuffle()
    train_data = dataset[:num_train_examples]
    valid_data = dataset[num_train_examples:]
    train_loader = dataloader(train_data, batch_size, num_data_workers, shuffle=True)
    valid_loader = dataloader(valid_data, batch_size, num_data_workers)

    if check_loader_perf:
        loader_perf(device, train_loader, "train loader")
        loader_perf(device, valid_loader, "valid loader")

    if compile_mode is None:
        model = create_model(model_config, train_loader)
    else:
        model = compile.prepare(create_model)(model_config, train_loader)
        model = torch.compile(model, mode=compile_mode)

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = create_scheduler(use_warmup, optimizer)

    for _ in (epoch_bar := tqdm(range(max_num_epochs))):
        loss, train_time = training_step(device, train_loader, model, optimizer)
        valid, eval_time = eval(device, valid_loader, model)
        scheduler.step(valid)

        results = {
            "lr": optimizer.param_groups[0]["lr"],
            "time/train": train_time,
            "time/eval": eval_time,
            "train": float(loss),
            "valid": float(valid),
        }

        if use_wandb:
            run.log(results)

        epoch_bar.set_postfix(**results)


def create_model(config, train_loader):
    hidden_irreps = o3.Irreps(
        (config.num_channels * o3.Irreps.spherical_harmonics(config.max_L))
        .sort()
        .irreps.simplify()
    )
    # fmt:off
    model_config = dict(
        num_elements=QM9_atomic_energies.shape[0],  # number of chemical elements
        atomic_energies=QM9_atomic_energies,  # atomic energies used for normalisation
        avg_num_neighbors=modules.compute_avg_num_neighbors(tqdm(train_loader)),  # avg number of neighbours of the atoms, used for internal normalisation of messages
        atomic_numbers=QM9_atomic_numbers,  # atomic numbers, used to specify chemical element embeddings of the model
        num_polynomial_cutoff=6,  # smoothness of the radial cutoff
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],  # interation block of first layer
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],  # interaction block of subsequent layers
        hidden_irreps=hidden_irreps,  # 32: number of embedding channels, 0e, 1o is specifying which equivariant messages to use. Here up to L_max=1
        MLP_irreps=o3.Irreps("16x0e"),  # number of hidden dimensions of last layer readout MLP
        gate=torch.nn.functional.silu,  # nonlinearity used in last layer readout MLP
        **{k: v for k, v in asdict(config).items() if k not in ["num_channels", "max_L"]}
    )
    # fmt:on

    model = modules.ScaleShiftMACE(
        **model_config, atomic_inter_scale=1.0, atomic_inter_shift=0.0
    )
    return model


def loader_perf(device, loader, desc):
    """
    Measures time perform one complete pass of the dataloader
    """
    for batch in tqdm(loader, desc=desc):
        batch = batch.to(device)
        batch = batch.to_dict()


def dataloader(dataset, batch_size, num_workers, **kwargs):
    """
    Create a PyTorch dataloader with some performance motivated overrides
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=8 if num_workers > 0 else None,
        **kwargs,
    )


def eval(device, valid_loader, model):
    tic = time.perf_counter()
    model = model.eval()
    valid_fn = MeanAbsoluteError().to(device)

    for batch in valid_loader:
        batch = batch.to(device)
        batch = batch.to_dict()
        output = model(batch, training=True)
        valid_fn(output["energy"], batch["energy"])
    return valid_fn.compute(), time.perf_counter() - tic


def training_step(device, train_loader, model, optimizer):
    tic = time.perf_counter()
    model = model.train()
    loss_fn = MeanSquaredError().to(device)

    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        batch = batch.to_dict()
        output = model(batch, training=True)
        loss = loss_fn(output["energy"], batch["energy"])
        loss.backward()
        optimizer.step()

    return loss_fn.compute(), time.perf_counter() - tic


if __name__ == "__main__":
    CLI(train)
