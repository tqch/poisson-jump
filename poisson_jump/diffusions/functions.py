import math
import torch
from typing import Union, Tuple


__all__ = [
    "extract",
    "flat_mean",
    "rand_zero",
    "poisson_kl",
    "stable_log1mexp",
    "parse_range"
]


def extract(arr_func, x, t: Union[torch.Tensor, int], ndim=None, dtype=None, device=None):
    ndim, dtype, device = ndim or x.ndim, dtype or x.dtype, device or x.device
    if isinstance(t, int):
        t = torch.as_tensor([t, ], dtype=torch.int64, device=device)
    dims = [t.shape[0], ] + [1 for _ in range(ndim - 1)]
    if callable(arr_func):
        outs = arr_func(t)
        if isinstance(outs, tuple):
            return [out.to(dtype=dtype, device=device).reshape(dims) for out in outs]
        else:
            return outs.to(dtype=dtype, device=device).reshape(dims)
    else:
        assert t.ndim == 1 and arr_func.ndim == 1 and t.dtype == torch.int64
        return arr_func.to(dtype=dtype, device=device).gather(0, t).reshape(dims)


def flat_mean(x):
    reduce_dims = list(range(1, x.ndim))
    return x.mean(dim=reduce_dims)


def rand_zero(x, prob):
    zero_mask = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    return torch.where(torch.rand(zero_mask, device=x.device) < prob, x, torch.zeros_like(x))


@torch.jit.script
def poisson_kl(rate_1, rate_2, eps: float = 1e-12):
    """
    Bregman divergence induced by (generalized) negative entropy on non-negative orthant
    """
    return (rate_1 - rate_2).neg().add(rate_1.mul(
        rate_1.clamp(min=eps).log() - rate_2.clamp(min=eps).log()))


@torch.jit.script
def poisson_loglik(x, rate, start: int = 0, eps: float = 1e-12):
    return torch.lgamma(x - start + 1).neg() + (x - start) * rate.clamp(min=eps).log() - rate


@torch.jit.script
def approx_std_normal_cdf(x):
    """
    Reference:
    Page, E. “Approximations to the Cumulative Normal Function and Its Inverse for Use on a Pocket Calculator.”
     Applied Statistics 26.1 (1977): 75–76. Web.
    """
    return 0.5 * (1. + torch.tanh(math.sqrt(2. / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


@torch.jit.script
def discretized_gaussian_loglik(
        x, means, log_scale, precision: float = 1./255,
        cutoff: Union[float, Tuple[float, float]] = (-0.999, 0.999), tol: float = 1e-12):
    if isinstance(cutoff, float):
        cutoff = (-cutoff, cutoff)
    # Assumes data is integers [0, 255] rescaled to [-1, 1]
    x_centered = x - means
    inv_stdv = torch.exp(-log_scale)
    upper = inv_stdv * (x_centered + precision)
    cdf_upper = torch.where(
        x > cutoff[1], torch.as_tensor(1, dtype=torch.float32, device=x.device), approx_std_normal_cdf(upper))
    lower = inv_stdv * (x_centered - precision)
    cdf_lower = torch.where(
        x < cutoff[0], torch.as_tensor(0, dtype=torch.float32, device=x.device), approx_std_normal_cdf(lower))
    log_probs = torch.log(torch.clamp(cdf_upper - cdf_lower - tol, min=0).add(tol))
    return log_probs


@torch.jit.script
def categorical_kl(logits_1, logits_2, eps: float = 1e-12):
    return torch.softmax(logits_1, dim=-1) * (
            torch.log_softmax(logits_1 + eps, dim=-1) - torch.log_softmax(logits_2 + eps, dim=-1))


def stable_log1mexp(x):
    """
    numerically stable version of log(1-exp(x)), x<0
    """
    assert torch.all(x < 0.)
    return torch.where(
        x < -9,
        torch.log1p(torch.exp(x).neg()),
        torch.log(torch.expm1(x).neg()))


def parse_range(seq):
    if seq is None:
        return None
    else:
        assert hasattr(seq, "__len__") and len(seq) == 2 and hasattr(seq, "__iter__")
        return tuple(seq)
