import numpy as np
import functools
import warnings
import torch
from typing import Callable, TypeVar
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang

from typing_extensions import ParamSpec

from exceptions import ExperimentalWarning

INTERNAL_ERROR_MESSAGE = (
    'Internal error. Please, open an issue here:'
    ' https://github.com/Yura52/rtdl/issues/new'
)


def all_or_none(values):
    return all(x is None for x in values) or all(x is not None for x in values)


P = ParamSpec('P')
T = TypeVar('T')


def experimental(x: Callable[P, T]) -> Callable[P, T]:
    if not callable(x):
        raise ValueError('Only callable objects can be experimental')

    @functools.wraps(x)
    def experimental_x(*args: P.args, **kwargs: P.kwargs):
        warnings.warn(
            f'{x.__name__} (full name: {x.__qualname__}) is an experimental feature of rtdl',
            ExperimentalWarning,
        )
        return x(*args, **kwargs)

    return experimental_x

def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window

def calibrate_mean_var(matrix, m1, v1, m2, v2, clip_min=0.1, clip_max=10):
    if torch.sum(v1) < 1e-10:
        return matrix
    if (v1 == 0.).any():
        valid = (v1 != 0.)
        factor = torch.clamp(v2[valid] / v1[valid], clip_min, clip_max)
        matrix[:, valid] = (matrix[:, valid].clone() - m1[valid]) * torch.sqrt(factor) + m2[valid]
        return matrix

    factor = torch.clamp(v2 / v1, clip_min, clip_max)
    return (matrix - m1) * torch.sqrt(factor) + m2