import torch
from .base import BaseDiffusion
from .functions import *
from typing import Callable


__all__ = []


def register_diffusion(cls):
    __all__.append(cls.__name__)
    return cls


@register_diffusion
class OrdinalJumpDiffusion(BaseDiffusion):
    name = "ordinal_jump"
    alias = "poisson_jump"

    def __init__(
            self,
            alphas: torch.Tensor = None,
            alpha_fn: Callable = None,
            lbd: float = 1.,
            pred_type: str = "x_0",
            loss_type: str = "const",
            timesteps: int = 1000,
            clip_range: tuple = None,
            input_clip: tuple = None,
            normalize: tuple = None,  # torchvision-like data normalization
            z_rescale: bool = False,
            p_self_cond: float = 0.,
            time_diff: float = 0.,
            momentum: float = 0.,
            **kwargs
    ):
        assert (alphas is None) ^ (alpha_fn is None)
        if alpha_fn is None:
            self.timesteps = len(alphas)  # noqa
        else:
            assert timesteps is not None
            alphas = alpha_fn(torch.linspace(1 / timesteps, 1, timesteps, dtype=torch.float64))
            self.timesteps = timesteps
            # To use real-valued time difference, a continuous alpha schedule is required
            if time_diff > 0:
                alphas_prev_asym = alpha_fn(torch.linspace(
                    0, 1 - 1 / timesteps, timesteps, dtype=torch.float64).sub(time_diff / timesteps).clamp(min=0))
                self.deltas_asym = (alphas_prev_asym - alphas) * lbd

        self.alphas = alphas
        self.alpha_fn = alpha_fn
        self.lbd = lbd
        self.alphas_prev = torch.cat([torch.as_tensor([1., ]), alphas[:-1]], dim=0)
        self.deltas = self.deltas_asym = (self.alphas_prev - alphas) * lbd

        self.pred_type = pred_type
        self.loss_type = loss_type
        self.clip_range = parse_range(clip_range)
        self.input_clip = parse_range(input_clip)
        self.normalize = parse_range(normalize)
        self.z_rescale = z_rescale
        self.p_self_cond = p_self_cond
        self.time_diff = time_diff
        self.momentum = momentum

    def _get_alpha(self, x, t):
        if self.alpha_fn is not None and not isinstance(t, int):
            alpha = extract(self.alpha_fn, x, t)
        else:
            alpha = extract(self.alphas, x, t)
        return alpha

    def _delta_fn(self, t, asym=False):
        t_prev = t - 1. / self.timesteps
        if asym:
            t_prev.sub_(self.time_diff / self.timesteps)
        t_prev.clamp_(min=0.)
        alphas_prev = self.alpha_fn(t_prev)
        alphas = self.alpha_fn(t)
        deltas = alphas_prev - alphas
        return deltas.mul(self.lbd)

    def _get_delta(self, x, t, asym=False):
        if self.alpha_fn is not None and not isinstance(t, int):
            delta_fn = lambda t: self._delta_fn(t, asym=asym)
            delta = extract(delta_fn, x, t)
        else:
            delta = extract(self.deltas_asym if asym else self.deltas, x, t)
        return delta

    def q_sample(self, x_0, t, return_rate=False):
        rate = self._get_alpha(x_0, t) * self.lbd
        z_t = torch.poisson(rate * x_0)
        return (z_t, rate) if return_rate else z_t

    def q_posterior(self, x_0, z_t, t, asym=False, rng=None):
        rate = self._get_delta(x_0, t, asym=asym) * x_0
        z_prev = torch.poisson(rate, generator=rng) + z_t
        return z_prev

    def _pred_x_0_from_eps(self, z_t, eps, t):
        rate = self._get_alpha(z_t, t) * self.lbd
        x_0 = z_t.div(rate) - eps.div(rate.sqrt())
        return x_0

    def _pred_x_0_from_eps_anscombe(self, z_t, eps, t):
        rate = self._get_alpha(z_t, t) * self.lbd
        x_0 = (((z_t + 3. / 8).sqrt() - .5 * eps) ** 2 - 3. / 8) / rate
        return x_0

    def _pred_x_0_from_eps_freeman_tukey(self, z_t, eps, t):
        rate = self._get_alpha(z_t, t) * self.lbd
        x_0 = (torch.sqrt(z_t + 1) + torch.sqrt(z_t) - eps).pow(2).div(4 * rate)
        return x_0

    def p_sample_step(self, model, z_t, t, accum_x_0=None, return_pred=False, rng=None):  # noqa
        _t = torch.full((z_t.shape[0],), fill_value=t, device=z_t.device)
        if self.alpha_fn is not None:
            _t = _t.to(torch.float64).div(self.timesteps)
        _z_t = z_t
        if self.z_rescale:
            _z_t = z_t.div(self.lbd * self.alphas[t])
            if self.input_clip is not None:
                _z_t.clamp_(*self.input_clip)  # input clipping
        if self.p_self_cond > 0 and accum_x_0 is not None:
            _z_t = torch.cat([_z_t, accum_x_0], dim=1)
        model_out = model(_z_t, t=_t)
        if self.pred_type == "x_0":
            pred_x_0 = model_out
        elif self.pred_type == "eps":
            pred_x_0 = self._pred_x_0_from_eps(z_t, eps=model_out, t=_t)
        elif self.pred_type == "eps_anscombe":
            pred_x_0 = self._pred_x_0_from_eps_anscombe(z_t, eps=model_out, t=_t)
        elif self.pred_type == "eps_freeman_tukey":
            pred_x_0 = self._pred_x_0_from_eps_freeman_tukey(z_t, eps=model_out, t=_t)
        else:
            raise NotImplementedError(self.pred_type)
        pred_x_0 = pred_x_0.clamp(min=0)
        if self.clip_range is not None:
            pred_x_0.clamp_(*self.clip_range)  # prediction clipping
        z_prev = self.q_posterior(pred_x_0, z_t, t, asym=True, rng=rng)
        return (z_prev, pred_x_0) if return_pred else z_prev

    @torch.inference_mode()
    def p_sample(self, model, z_T, seed=None, return_pred=False):  # noqa
        z_t = z_T
        pred_x_0 = torch.zeros_like(z_t)
        accum_x_0 = torch.zeros_like(z_t)
        rng = None
        if seed is not None:
            device = next(model.parameters()).device
            rng = torch.Generator(device).manual_seed(seed)
        for t in range(self.timesteps - 1, -1, -1):
            accum_x_0 += (1 - self.momentum) * (pred_x_0 - accum_x_0)
            z_t, pred_x_0 = self.p_sample_step(
                model, z_t, t, accum_x_0=accum_x_0, return_pred=True, rng=rng)
        x_0 = z_t.div_(self.lbd)
        if self.normalize is not None:
            x_0.mul_(self.normalize[1]).add_(self.normalize[0])
            pred_x_0.mul_(self.normalize[1]).add_(self.normalize[0])
        return x_0, (pred_x_0 if return_pred else None)

    @torch.inference_mode()
    def p_sample_progressive(self, model, z_T, freq=10, seed=None):
        z_t = z_T
        pred_x_0 = torch.zeros_like(z_t)
        accum_x_0 = torch.zeros_like(z_t)
        B, *D = z_t.shape
        T = self.timesteps // freq + 1
        sample_path = torch.empty((T, B, *D))
        n = 0
        rng = None
        if seed is not None:
            device = next(model.parameters()).device
            rng = torch.Generator(device).manual_seed(seed)
        for t in range(self.timesteps - 1, -1, -1):
            accum_x_0 += (1 - self.momentum) * (pred_x_0 - accum_x_0)
            z_t, pred_x_0 = self.p_sample_step(
                model, z_t, t, accum_x_0=accum_x_0, return_pred=True, rng=rng)
            if (t + 1) % freq == 0:
                sample_path[n] = pred_x_0.cpu()
                n += 1
        sample_path[n] = z_t.div(self.lbd).cpu()
        if self.normalize is not None:
            sample_path.mul_(self.normalize[1]).add_(self.normalize[0])
        return sample_path

    def train_loss(self, model, x_0, t):
        if self.normalize is not None:
            x_0 = x_0.sub(self.normalize[0]).div(self.normalize[1])

        z_t, rate = self.q_sample(x_0, t, return_rate=True)
        _z_t = z_t
        if self.z_rescale:
            _z_t = z_t.div(rate)  # unbiased mean estimator of x_0
            if self.input_clip is not None:
                _z_t.clamp_(*self.input_clip)  # input clipping

        if self.p_self_cond > 0:
            with torch.no_grad():
                _z_t = torch.cat([
                    _z_t, rand_zero(model(torch.cat([
                        _z_t, torch.zeros_like(_z_t)
                    ], dim=1), t=t), prob=self.p_self_cond)
                ], dim=1)

        model_out = model(_z_t, t=t)

        if self.pred_type == "x_0":
            pred_x_0 = model_out
        elif self.pred_type == "eps":
            pred_x_0 = self._pred_x_0_from_eps(z_t, eps=model_out, t=t)
        elif self.pred_type == "eps_anscombe":
            pred_x_0 = self._pred_x_0_from_eps_anscombe(z_t, eps=model_out, t=t)
        elif self.pred_type == "eps_freeman_tukey":
            pred_x_0 = self._pred_x_0_from_eps_freeman_tukey(z_t, eps=model_out, t=t)
        else:
            raise NotImplementedError(self.pred_type)

        pred_x_0 = pred_x_0.clamp(min=0)  # not necessary?

        # KL(q(z_{t-1}|z_t, x_0) || p_\theta(z_{t-1}|z_t))
        # z_{t-1}|z_t, x_0 = z_t + Pois(\lambda(\alpha_{t-1}-\alpha_t})x_0)
        if self.loss_type == "kl":  # no re-weighting
            delta = self._get_delta(z_t, t)
            loss = flat_mean(poisson_kl(delta * x_0, delta * pred_x_0))
        elif self.loss_type == "kl_rev":  # no re-weighting
            delta = self._get_delta(z_t, t)
            loss = flat_mean(poisson_kl(delta * pred_x_0, delta * x_0))
        elif self.loss_type == "kl_simple":  # weight: 1 / delta
            loss = flat_mean(poisson_kl(x_0, pred_x_0))
        elif self.loss_type == "kl_alpha":  # weight: sqrt(alpha) / delta
            alpha = self._get_alpha(x_0, t)
            loss = flat_mean(poisson_kl(x_0, pred_x_0)) * alpha.sqrt()
        else:
            raise NotImplementedError(self.loss_type)

        return loss


@register_diffusion
class BitJumpDiffusion(OrdinalJumpDiffusion):
    name = "bit_jump"
    alias = "bit_poisson_jump"

    def __init__(
            self,
            alphas: torch.Tensor = None,
            alpha_fn: Callable = None,
            lbd: float = 1.,
            pred_type: str = "x_0",
            loss_type: str = "const",
            timesteps: int = 1000,
            clip_range: tuple = None,
            input_clip: tuple = None,
            normalize: tuple = None,
            z_rescale: bool = False,
            p_self_cond: float = 0.,
            time_diff: float = 0.,
            momentum: float = 0.,
            num_bits: int = 8,
            **kwargs
    ):
        super().__init__(
            alphas=alphas, alpha_fn=alpha_fn, lbd=lbd, pred_type=pred_type, loss_type=loss_type, timesteps=timesteps,
            clip_range=clip_range, input_clip=input_clip, normalize=normalize, z_rescale=z_rescale,
            p_self_cond=p_self_cond, time_diff=time_diff, momentum=momentum)
        self.num_bits = num_bits

    def int_to_bits(self, x):
        shift = torch.arange(self.num_bits - 1, -1, -1, device=x.device).reshape((1, 1, -1) + (1,) * (x.ndim - 2))
        bits = torch.fmod(torch.bitwise_right_shift(
            x.int().unsqueeze(2), shift), 2).reshape((x.shape[0], -1) + x.shape[2:])
        return bits.float()

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
        return self.bits_to_int(x_0.round()), (self.bits_to_int(pred_x_0.round()) if return_pred else None)

    @torch.inference_mode()
    def p_sample_progressive(self, model, z_T, freq=10, seed=None):
        sample_path = super().p_sample_progressive(model=model, z_T=z_T, freq=freq, seed=seed).round()
        return self.bits_to_int(sample_path.reshape((-1,) + sample_path.shape[2:])).reshape(sample_path.shape)


if __name__ == "__main__":
    print(__all__)
