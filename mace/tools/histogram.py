import logging
import os
import os.path as osp
from typing import Any, Optional

import numpy as np
import pandas as pd
from matplotlib.pyplot import Axes
from numpy.typing import NDArray
from torch.nn import Module

from mace.tools.torch_tools import to_numpy


def fpbins(low: np.dtype = np.float32, high: np.dtype = np.float64) -> NDArray:
    """Create bins for floating point exponent histogram

    Args:
        low (np.dtype, optional): Defaults to np.float32.
        high (np.dtype, optional): Defaults to np.float64.

    Returns:
        NDArray: bins
    """
    low_info = np.finfo(low)
    high_info = np.finfo(high)
    _, low_eps_exp = np.frexp(low_info.eps)
    _, high_eps_exp = np.frexp(high_info.eps)
    lbin = np.array([high_info.minexp, high_eps_exp])
    rbin = np.array([-high_eps_exp, high_info.maxexp])
    bins = np.arange(low_eps_exp, -low_eps_exp)
    bins = np.concatenate([lbin, bins, rbin])
    return bins


def expcounts(array: NDArray, bins: NDArray) -> NDArray:
    """Histogram of exponent values

    Args:
        array (NDArray): the floating point data to summarize.
        bins (NDArray): the bins used by the histogram

    Returns:
        NDArray: histogram of exponent values
    """
    _, ex = np.frexp(array)
    counts, bins = np.histogram(ex, bins=bins)
    return counts


def module_histogram(module: Module) -> pd.DataFrame:
    """Collect exponent histograms for all module parameters and their gradients

    Args:
        module (Module): the root module

    Returns:
        pd.DataFrame: Exponent histograms indexed by the module parameter names
    """
    bins = fpbins()
    names = []
    counts = []

    for name, param in module.named_parameters():
        p = to_numpy(param)
        counts.append(expcounts(p, bins))
        names.append(name)

        if param.grad is not None:
            g = to_numpy(param.grad)
            counts.append(expcounts(g, bins))
            names.append(f"{name}.grad")

    if len(names) == 0:
        return None

    columns = [f"[{bins[n]}, {bins[n+1]})" for n in range(len(bins) - 1)]
    columns[-1] = columns[-1].replace(")", "]")
    return pd.DataFrame(np.stack(counts), columns=columns, index=names)


def plot_histogram(
    data: pd.DataFrame, epoch: int = None, heatmap_kw: dict[str, Any] = {}
) -> Axes:
    """Plot exponent histogram using seaborn.heatmap

    Args:
        data (pd.DataFrame): exponent histograms
        epoch (int, optional): Optional epoch title. Defaults to None.
        heatmap_kw (dict[str, Any], optional): keyword arguments to seaborn.heatmap.
            Defaults to {}.

    Returns:
        Axes: heatmap axes
    """
    from seaborn import heatmap

    defaults = {"cmap": "Blues", "vmin": 0.0, "vmax": 0.2, "square": True}

    for k, v in defaults.items():
        heatmap_kw.setdefault(k, v)

    ax = heatmap(data, **heatmap_kw)
    ax.axes.set_xlabel("Scale ($log_2$)")

    if epoch is not None:
        ax.set_title(f"Epoch {epoch}")

    return ax


class HistogramLogger:
    def __init__(self, root_dir: Optional[str] = None):
        """Collect exponent histograms across multiple training steps.

        This object should be used as a context manager to aggregate the collected
        histograms within a single training epoch.

        Args:
            root_dir (Optional[str], optional): Location to save aggregated histograms.
                Defaults to None which disables the logger.
        """
        self.disable = root_dir is None
        if self.disable:
            return

        self.epoch = -1
        self.data = []

        root_dir = osp.expanduser(osp.normcase(root_dir))
        os.makedirs(root_dir, exist_ok=True)
        self.root_dir = root_dir
        logging.info(f"{self.__class__.__name__} logging to {self.root_dir}")

    def __enter__(self):
        """Increment the epoch used to aggregate histograms over multiple training steps"""
        if self.disable:
            return

        self.epoch += 1

    def step(self, module: Module):
        """Log a training step

        Args:
            module (Module): module to log
        """
        if self.disable:
            return

        self.data.append(module_histogram(module))

    def __exit__(self, *_):
        """Aggregates the collected exponent histograms"""
        if self.disable:
            return

        values = np.concatenate([d.values[:, :, None] for d in self.data], axis=2)
        values = np.sum(values, axis=2)
        W = np.sum(values, axis=1)
        density = values / W[:, None]
        index = self.data[0].index
        columns = self.data[0].columns
        data = pd.DataFrame(density, index=index, columns=columns)
        file = f"{self.root_dir}/ep{self.epoch:04d}.parquet"
        data.to_parquet(file)
        self.data = []
