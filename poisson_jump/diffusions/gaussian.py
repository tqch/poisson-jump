import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseDiffusion
from .functions import *
from typing import Callable


__all__ = []


def register_diffusion(cls):
    __all__.append(cls.__name__)
    return cls


def logsnr_to_posterior(logsnr_s, logsnr_t, var_type: str):
    assert logsnr_s.dtype == logsnr_t.dtype == torch.float64

    log_alpha_st = 0.5 * (F.logsigmoid(logsnr_s) - F.logsigmoid(logsnr_t))
    logr = logsnr_t - logsnr_s
    log_one_minus_r = stable_log1mexp(logr)
    mean_coef1 = (logr + log_alpha_st).exp()
    mean_coef2 = (log_one_minus_r + 0.5 * F.logsigmoid(logsnr_s)).exp()

    # strictly speaking, only when var_type == "small",
    # does `logvar` calculated here represent the logarithm
    # of the true posterior variance
    if var_type == "fixed_large":
        logvar = log_one_minus_r + F.logsigmoid(-logsnr_t)
    elif var_type == "fixed_small":
        logvar = log_one_minus_r + F.logsigmoid(-logsnr_s)
    else:
        raise NotImplementedError(var_type)

    return mean_coef1, mean_coef2, logvar


@register_diffusion
class GaussianDiffusion(BaseDiffusion):
    name = "gaussian"
    alias = "ddpm"

    def __init__(
            self,
            betas: torch.Tensor = None,
            logsnr_fn: Callable = None,
            pred_type: str = "x_0",
            var_type: str = "fixed_small",
            loss_type: str = "mse",
            timesteps: int = 1000,
            clip_range: tuple = None,
            normalize: tuple = None,
            p_self_cond: float = 0.,
            time_diff: float = 0.,
            momentum: float = 0.,
            **kwargs
    ):
        assert (betas is None) ^ (logsnr_fn is None)
        self.betas = betas
        self.logsnr_fn = logsnr_fn

        if betas is None:
            logsnrs = logsnr_fn(torch.linspace(1. / timesteps, 1., timesteps, dtype=torch.float64))  # noqa
            logsnrs_prev = torch.cat([logsnr_fn(torch.as_tensor([0., ], dtype=torch.float64)), logsnrs[:-1]])  # noqa
            self.timesteps = timesteps
            alphas_bar = torch.sigmoid(logsnrs)
            self.sqrt_alphas_bar = alphas_bar.sqrt()
            self.sqrt_one_minus_alphas_bar = torch.sigmoid(-logsnrs).sqrt()
            self.posterior_mean_coef1, self.posterior_mean_coef2, self.posterior_logvar = logsnr_to_posterior(
                logsnrs_prev, logsnrs, var_type="fixed_small")
            *_, self.log_betas = logsnr_to_posterior(logsnrs_prev, logsnrs, var_type=var_type)
            if time_diff > 0:
                logsnrs_prev_asym = (logsnrs_prev - time_diff / timesteps).clamp(min=0.)
                self.posterior_mean_coef1_asym, self.posterior_mean_coef2_asym, self.posterior_logvar =\
                    logsnr_to_posterior(logsnrs_prev, logsnrs, var_type="fixed_small")
                *_, self.log_betas_asym = logsnr_to_posterior(logsnrs_prev_asym, logsnrs, var_type=var_type)
        else:
            self.timesteps = len(betas)
            self.log_betas = torch.log(betas)
            alphas = 1 - betas
            alphas_bar = torch.cumprod(alphas, dim=0)
            alphas_bar_prev = torch.cat([torch.as_tensor([1., ], dtype=torch.float64), alphas_bar[:-1]])
            self.sqrt_alphas_bar = torch.sqrt(alphas_bar)
            sqrt_alphas_bar_prev = torch.sqrt(alphas_bar_prev)
            self.sqrt_one_minus_alphas_bar = torch.sqrt(1. - alphas_bar)
            self.posterior_mean_coef1 = torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar)
            self.posterior_mean_coef2 = betas * sqrt_alphas_bar_prev / (1. - alphas_bar)
            self.posterior_logvar = torch.log(betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

        # for fixed model_var_type's
        self.p_logvar = {
            "fixed_large": torch.cat([self.posterior_logvar[[1]], self.log_betas]),
            "fixed_small": self.posterior_logvar
        }[var_type]

        self.sqrt_recip_alphas_bar = torch.sqrt(1. / alphas_bar)
        self.sqrt_recip_m1_alphas_bar = torch.sqrt(1. / alphas_bar - 1.)  # m1: minus 1

        self.pred_type = pred_type
        self.var_type = var_type
        self.loss_type = loss_type
        self.normalize = parse_range(normalize)
        self.clip_range = parse_range(clip_range)
        self.p_self_cond = p_self_cond
        self.time_diff = time_diff
        self.momentum = momentum

    def _q_sample_coefs_fn(self, t):
        logsnrs = self.logsnr_fn(t)
        sqrt_alphas_bar = torch.sigmoid(logsnrs).sqrt()
        sqrt_one_minus_alphas_bar = torch.sigmoid(-logsnrs).sqrt()
        return sqrt_alphas_bar, sqrt_one_minus_alphas_bar

    def _get_q_sample_coefs(self, x, t):
        if self.betas is None and not isinstance(t, int):
            sqrt_alphas_bar, sqrt_one_minus_alphas_bar = extract(self._q_sample_coefs_fn, x, t)
        else:
            sqrt_alphas_bar = extract(self.sqrt_alphas_bar, x, t)
            sqrt_one_minus_alphas_bar = extract(self.sqrt_one_minus_alphas_bar, x, t)
        return sqrt_alphas_bar, sqrt_one_minus_alphas_bar

    def q_sample(self, x_0, t, return_eps=False):
        eps = torch.randn_like(x_0)
        coef1, coef2 = self._get_q_sample_coefs(x_0, t)
        z_t = coef1 * x_0 + coef2 * eps
        return (z_t, eps) if return_eps else z_t

    def q_posterior(self, x_0, z_t, t, p_var=False, rng=None):
        posterior_mean_coef1 = extract(self.posterior_mean_coef1, x_0, t)
        posterior_mean_coef2 = extract(self.posterior_mean_coef2, x_0, t)
        posterior_mean = posterior_mean_coef1 * z_t + posterior_mean_coef2 * x_0
        posterior_logvar = extract(self.p_logvar if p_var else self.posterior_logvar, x_0, t)
        x_prev = torch.where(
            t.reshape((-1,) + tuple(1 for _ in range(x_0.ndim - 1))) > 0,
            posterior_logvar.mul(0.5).exp() * torch.empty_like(x_0).normal_(generator=rng),
            torch.as_tensor(0., device=x_0.device)) + posterior_mean
        return x_prev

    def _sqrt_recip_fn(self, t):
        logsnrs = self.logsnr_fn(t)
        alphas_bar = torch.sigmoid(logsnrs)
        sqrt_recip_alphas_bar = torch.sqrt(1. / alphas_bar)
        sqrt_recip_m1_alphas_bar = torch.sqrt(1. / alphas_bar - 1.)  # m1: minus 1
        return sqrt_recip_alphas_bar, sqrt_recip_m1_alphas_bar

    def _pred_x_0_from_eps(self, z_t, eps, t):
        if self.betas is None and not isinstance(t, int):
            coef1, coef2 = extract(self._sqrt_recip_fn, z_t, t)
        else:
            coef1 = extract(self.sqrt_recip_alphas_bar, z_t, t)
            coef2 = extract(self.sqrt_recip_m1_alphas_bar, z_t, t)
        return coef1 * z_t - coef2 * eps

    # === sample ===
    def p_sample_step(self, model, z_t, t, accum_x_0=None, return_pred=False, rng=None):
        _t = torch.full((z_t.shape[0],), fill_value=t, device=z_t.device)
        if self.logsnr_fn is not None:
            _t = _t.to(torch.float64).div(self.timesteps)
        _z_t = z_t
        if self.p_self_cond > 0 and accum_x_0 is not None:
            _z_t = torch.cat([z_t, accum_x_0], dim=1)
        model_out = model(_z_t, t=_t)
        # calculate the mean estimate
        if self.pred_type == "x_0":
            pred_x_0 = model_out
        elif self.pred_type == "eps":
            pred_x_0 = self._pred_x_0_from_eps(z_t=z_t, eps=model_out, t=t)
        else:
            raise NotImplementedError
        if self.clip_range is not None:
            pred_x_0.clamp_(*self.clip_range)
        _t = torch.atleast_1d(torch.as_tensor(t, device=z_t.device))
        z_prev = self.q_posterior(x_0=pred_x_0, z_t=z_t, t=_t, p_var=True, rng=rng)
        return (z_prev, pred_x_0) if return_pred else z_prev

    @torch.inference_mode()
    def p_sample(self, model: nn.Module, z_T, seed=None, return_pred=False):
        x_0, pred_x_0 = super().p_sample(model, z_T, seed, return_pred=True)
        if self.normalize is not None:
            x_0.mul_(self.normalize[1]).add_(self.normalize[0])
        return x_0, (pred_x_0 if return_pred else None)

    @torch.inference_mode()
    def p_sample_progressive(self, model, z_T, freq=10, seed=None):
        sample_path = super().p_sample_progressive(model, z_T, freq, seed)
        if self.normalize is not None:
            sample_path.mul_(self.normalize[1]).add_(self.normalize[0])
        return sample_path

    def train_loss(self, model, x_0, t):
        if self.normalize is not None:
            x_0 = x_0.sub(self.normalize[0]).div(self.normalize[1])
        z_t, eps = self.q_sample(x_0, t, return_eps=True)

        if self.p_self_cond > 0:
            with torch.no_grad():
                model_out = model(torch.cat([z_t, torch.zeros_like(z_t)], dim=1), t=t)
                if self.pred_type == "eps":
                    model_out = self._pred_x_0_from_eps(z_t=z_t, eps=model_out, t=t)
                z_t = torch.cat([z_t, rand_zero(model_out, prob=self.p_self_cond)], dim=1)

        model_out = model(z_t, t=t)
        target = {"x_0": x_0, "eps": eps}[self.pred_type]
        loss = {
            "mae": nn.L1Loss(reduction="mean"),
            "mse": nn.MSELoss(reduction="mean"),
            "huber": nn.SmoothL1Loss(reduction="mean")
        }[self.loss_type](model_out, target)

        return loss


@register_diffusion
class BitGaussianDiffusion(GaussianDiffusion):
    name = "bit_gaussian"
    alias = None

    def __init__(
            self,
            betas: torch.Tensor = None,
            logsnr_fn: Callable = None,
            pred_type: str = "x_0",
            var_type: str = "fixed_small",
            loss_type: str = "l2",
            timesteps: int = 1000,
            clip_range: tuple = None,
            normalize: tuple = None,
            p_self_cond: float = 0.,
            time_diff: float = 0.,
            momentum: float = 0.,
            num_bits: int = 8,
            **kwargs
    ):
        super().__init__(
            betas=betas, logsnr_fn=logsnr_fn, pred_type=pred_type, var_type=var_type, loss_type=loss_type,
            timesteps=timesteps, clip_range=clip_range, normalize=normalize, p_self_cond=p_self_cond,
            time_diff=time_diff, momentum=momentum)
        self.num_bits = num_bits

    def int_to_bits(self, x):
        # bases = 2 ** torch.arange(self.num_bits - 1, -1, -1, device=x.device).reshape((1, 1, -1) + (1,) * (x.ndim - 2))
        # bits = ((bases & x.int().unsqueeze(2)) != 0).float() * 2 - 1
        # bits = bits.reshape((bits.shape[0], -1) + bits.shape[3:])
        shift = torch.arange(self.num_bits - 1, -1, -1, device=x.device).reshape((1, 1, -1) + (1, ) * (x.ndim - 2))
        bits = torch.fmod(torch.bitwise_right_shift(x.int().unsqueeze(2), shift), 2)
        bits = (bits * 2. - 1.).reshape((x.shape[0], -1) + x.shape[2:])  # 0 -> -1, 1 -> 1
        return bits

    def bits_to_int(self, bits):
        bases = 2 ** torch.arange(
            self.num_bits - 1, -1, -1, device=bits.device
        ).reshape((1, 1, -1) + (1,) * (bits.ndim - 2))
        x = ((bits > 0).int().reshape((bits.shape[0], -1, self.num_bits) + bits.shape[2:]) * bases).sum(dim=2)
        return x.float()

    def train_loss(self, model, x_0, t, **kwargs):
        bits = self.int_to_bits(x_0)
        return super().train_loss(model=model, x_0=bits, t=t)

    @torch.inference_mode()
    def p_sample(self, model, z_T, seed=None, return_pred=False):
        x_0, pred_x_0 = super().p_sample(model=model, z_T=z_T, seed=seed, return_pred=True)
        return self.bits_to_int(x_0), (self.bits_to_int(pred_x_0) if return_pred else None)

    @torch.inference_mode()
    def p_sample_progressive(self, model, z_T, freq=10, seed=None):
        sample_path = super().p_sample_progressive(model=model, z_T=z_T, freq=freq, seed=seed)
        return self.bits_to_int(sample_path.reshape((-1,) + sample_path.shape[2:])).reshape(sample_path.shape)
