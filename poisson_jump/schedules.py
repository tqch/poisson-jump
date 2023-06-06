import math
import numpy as np
import scipy.optimize
import torch

__all__ = ["get_decay_schedule"]


def log_sigmoid(x):
    if x < -9 or 9:
        out = x
    elif x > 9:
        out = -np.exp(-x)
    else:
        out = -np.log(1 + np.exp(-x))
    return out


def input_check(dtype=torch.float64, cont=False):
    def check(func):
        def func_w_check(t):
            assert t.dtype == dtype
            if cont:
                assert torch.all(torch.logical_and(0 <= t, t <= 1))
            return func(t)
        return func_w_check
    return check


def _warmup_schedule(start, end, timesteps, warmup_frac):
    coefs = end * torch.ones(timesteps, dtype=torch.float64)
    warmup_time = int(timesteps * warmup_frac)
    coefs[:warmup_time] = torch.linspace(start, end, warmup_time, dtype=torch.float64)
    return coefs


def _signal_decay_sequence(schedule, start, end, timesteps):
    """
    schedules of signal decay
    alpha: cumulative decay coefficient
    beta: 1 - decay ratio
    """
    if schedule == "quad":
        coefs = torch.linspace(start ** 0.5, end ** 0.5, timesteps, dtype=torch.float64) ** 2
    elif schedule == "linear":
        coefs = torch.linspace(start, end, timesteps, dtype=torch.float64)
    elif schedule == "linear2":
        _start, _end = 1 - math.sqrt(1 - start), 1 - math.sqrt(1 - end)
        coefs = torch.linspace(_start, _end, timesteps, dtype=torch.float64)
        coefs = coefs * (2 - coefs)
    elif schedule == "warmup10":
        coefs = _warmup_schedule(start, end, timesteps, 0.1)
    elif schedule == "warmup50":
        coefs = _warmup_schedule(start, end, timesteps, 0.5)
    elif schedule == "const":
        coefs = torch.full((timesteps,), fill_value=start, dtype=torch.float64)
    elif schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        coefs = 1. / torch.linspace(timesteps, 1, timesteps, dtype=torch.float64)
    elif schedule == "cosine":
        coefs = end + (start - end) * torch.cos(0.5 * math.pi * torch.linspace(0, 1, timesteps, dtype=torch.float64))
    elif schedule == "cosine2":
        coefs = end + 0.5 * (start - end) * (
                torch.cos(math.pi * torch.linspace(0, 1, timesteps, dtype=torch.float64)) + 1)
    elif schedule == "cosine_improved":
        # from Nichol, Alexander Quinn, and Prafulla Dhariwal.
        # "Improved denoising diffusion probabilistic models." International Conference on Machine Learning. PMLR, 2021.
        coefs = torch.cos(0.5 * math.pi * torch.linspace(
            1. / timesteps, 1, timesteps, dtype=torch.float64
        ).add(0.008).div(1.008))
        coefs /= math.cos(0.004 * math.pi / 1.008)
    elif schedule == "invcdf_uniform":
        coefs = (2 * torch.arange(timesteps - 1, -1, -1, dtype=torch.float64) + 1) /\
                (torch.arange(timesteps, 0, -1, dtype=torch.float64) ** 2)
    elif schedule == "sigmoid":
        _start, _end = math.log(start) - math.log1p(-start), math.log(end) - math.log1p(-end)
        coefs = torch.sigmoid(torch.linspace(_start, _end, timesteps, dtype=torch.float64))
    else:
        raise NotImplementedError(schedule)
    assert coefs.shape == (timesteps, ) and coefs.dtype == torch.float64
    return coefs


def infer_lbd_from_logsnr(logsnr_start, diffusion_type, signal_stat=1.):
    assert diffusion_type.endswith("jump")
    if diffusion_type.endswith("beta_jump"):
        assert 0. < signal_stat < 1.
        return math.exp(logsnr_start) / signal_stat * (1. - signal_stat) - 1.
    else:
        return math.exp(logsnr_start) / signal_stat


def infer_alpha_from_beta(beta):
    return torch.cumprod(1. - beta, dim=0).sqrt()


def infer_alpha_from_logsnr(logsnr, diffusion_type, lbd=1., signal_stat=1.):
    if not isinstance(logsnr, torch.Tensor):
        logsnr = torch.as_tensor(logsnr)
    if diffusion_type.endswith("jump"):
        if diffusion_type.endswith("beta_jump"):
            return torch.sigmoid(logsnr - math.log(lbd + 1.)) / signal_stat
        else:
            return torch.exp(logsnr) / lbd / signal_stat
    elif diffusion_type.endswith("gaussian"):
        return torch.sqrt(torch.sigmoid(logsnr)).item()
    else:
        raise NotImplementedError(diffusion_type)


def infer_beta_end_from_logsnr(logsnr_end, beta_start, lbd, signal_stat, timesteps=1000, diffusion_type="gaussian"):
    def logsnr_fn(beta_end):
        a, b = 1 - beta_start, 1 - beta_end
        fb = 0.5 * timesteps * ((b * math.log(b) - a * math.log(a)) / (b - a) - 1)
        if diffusion_type.endswith("jump"):
            if diffusion_type.endswith("beta_jump"):
                target = - np.log1p(np.exp(-logsnr_end + np.log(lbd + 1.))) - np.log(signal_stat)
            else:
                target = logsnr_end - np.log(lbd) - np.log(signal_stat)
        elif diffusion_type.endswith("gaussian"):
            target = log_sigmoid(logsnr_end)
        else:
            raise NotImplementedError(diffusion_type)
        return fb - target
    return scipy.optimize.fsolve(logsnr_fn, np.array(0.00015), xtol=1e-6, maxfev=1000)[0].item()  # noqa


def _get_decay_sequence(schedule, start, end, timesteps, diffusion_type="gaussian", lbd=None, signal_stat=None):
    """
    returns decay coefficient alphas s.t. (z_t - \alpha_t x_0) \perp x_0, for any t
    and (legacy) betas, i.e. variance of additive noise in Gaussian Diffusion
    schedule must be in the format of {y_type}_{decay_schedule}
    y_type:
        alpha: cumulative signal decay factor
        beta: beta_t = 1 - \alpha_t / \alpha_{t-1}
        alpha2: alpha square
        logsnr:
            (Poisson Thinning) log(lbd * alpha * signal_mean/signal_peak)
            (Gaussian Diffusion) 2 * (log(alpha) - log(sqrt(1 - alpha)))
    """

    y_type, schedule = schedule.split("_", maxsplit=1)
    coefs = _signal_decay_sequence(schedule, start, end, timesteps)

    if y_type == "beta":
        betas = coefs
        alphas = infer_alpha_from_beta(betas)
    else:
        if y_type == "alpha":
            alphas = coefs
        elif y_type == "alpha2":
            alphas = torch.sqrt(coefs)
        elif y_type == "logsnr":
            if diffusion_type.endswith("jump"):
                alphas = torch.exp(coefs) / lbd / signal_stat
            elif diffusion_type.endswith("gaussian"):
                alphas = torch.sqrt(torch.sigmoid(coefs))
            else:
                raise NotImplementedError(diffusion_type)
        else:
            raise NotImplementedError(y_type)
        betas = 1. - torch.cat([torch.atleast_1d(alphas[0]), alphas[1:].div(alphas[:-1])]) ** 2
    if schedule in ("cosine_improved", "invcdf_uniform"):
        betas.clamp_(max=0.999)
        alphas = infer_alpha_from_beta(betas)
    return {"betas": betas, "alphas": alphas}


def _get_decay_function(schedule, start, end, timesteps, diffusion_type="gaussian", lbd=None, signal_stat=None):

    y_type, schedule = schedule.split("_", maxsplit=1)
    if y_type == "beta":
        assert schedule in ("linear", )

    schedule_fn = None
    if schedule == "cosine":
        if y_type == "alpha":
            if not diffusion_type.endswith("gaussian"):
                assert 0. <= start <= 1. and 0 <= end <= 1.
                _start, _end = math.acos(start), math.acos(end)

                @input_check(cont=True)
                def schedule_fn(t):
                    return torch.cos(_start + (_end - _start) * t)

        else:
            @input_check(cont=True)
            def schedule_fn(t):
                return end + (start - end) * torch.cos(0.5 * math.pi * t)

    elif schedule == "cosine2":
        if y_type == "alpha":
            assert 0. <= start <= 1. and 0 <= end <= 1.
            _start, _end = math.acos(2 * start - 1), math.acos(2 * end - 1)

            @input_check(cont=True)
            def schedule_fn(t):
                return torch.cos(_start + (_end - _start) * t).add(1.).div(2.)
        else:
            @input_check(cont=True)
            def schedule_fn(t):
                return end + 0.5 * (start - end) * (torch.cos(math.pi * t) + 1)

    elif schedule == "cosine_improved":
        assert y_type == "alpha"

        @input_check(cont=True)
        def schedule_fn(t):
            return torch.cos(0.5 * math.pi * (t + 0.008).div(1.008))

    elif schedule == "linear":
        start, end = 1 - start, 1 - end

        @input_check(cont=True)
        def schedule_fn(t):
            k, b = end - start, start
            x = k * t + b
            return torch.exp(0.5 * timesteps / k * (x * x.log() - b * math.log(b) - k * t))

    else:
        raise NotImplementedError(schedule)

    out_dict = {f"{y_type}_fn": schedule_fn}
    if diffusion_type.endswith("jump"):
        alpha_fn = None
        if y_type == "beta":
            alpha_fn = schedule_fn

        elif y_type == "logsnr":

            @input_check(cont=True)
            def alpha_fn(t):
                return infer_alpha_from_logsnr(
                    schedule_fn(t), diffusion_type=diffusion_type, lbd=lbd, signal_stat=signal_stat)

        if alpha_fn is not None:
            out_dict = {"alpha_fn": alpha_fn}
    
    elif diffusion_type.endswith("gaussian"):
        if y_type == "alpha":
            _start = math.atan(math.exp(0.5 * start)) / (0.5 * math.pi)
            _end = math.atan(math.exp(0.5 * end)) / (0.5 * math.pi)

            @input_check(cont=True)
            def logsnr_fn(t):
                return 2 * torch.log(torch.tan(
                    0.5 * math.pi * (_start + (_end - _start) * t)))

            out_dict = {"logsnr_fn": logsnr_fn}

    return out_dict


def get_decay_schedule(schedule, timesteps, return_function=False, diffusion_type="gaussian", **kwargs):
    assert schedule.split("_")[0] in ("beta", "alpha", "logsnr")
    var_type = "" if {"start", "end"}.issubset(kwargs) else schedule.split("_", maxsplit=1)[0] + "_"
    if diffusion_type.endswith("gaussian") and var_type.startswith("alpha"):
        var_type = "logsnr_"
    start, end = kwargs[var_type + "start"], kwargs[var_type + "end"]

    if diffusion_type.endswith("jump"):
        assert {"lbd", "signal_stat"}.issubset(kwargs)
        if kwargs["lbd"] == "auto":
            assert "logsnr_start" in kwargs
            kwargs["lbd"] = infer_lbd_from_logsnr(
                logsnr_start=kwargs["logsnr_start"], diffusion_type=diffusion_type, signal_stat=kwargs["signal_stat"])
    diffusion_kwargs = {"diffusion_type": diffusion_type, "lbd": kwargs["lbd"], "signal_stat": kwargs["signal_stat"]}

    if end == "auto":
        assert "logsnr_end" in kwargs
        if schedule.startswith("alpha"):
            kwargs["alpha_end"] = end = infer_alpha_from_logsnr(kwargs["logsnr_end"], **diffusion_kwargs)
        elif schedule.startswith("beta"):
            kwargs["beta_end"] = end = infer_beta_end_from_logsnr(
                kwargs["logsnr_end"], start, timesteps=timesteps, **diffusion_kwargs)

    return (_get_decay_function if return_function else _get_decay_sequence)(
        schedule, start, end, timesteps, **diffusion_kwargs), kwargs


if __name__ == "__main__":
    import os
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from collections import namedtuple

    mpl.rcParams["figure.dpi"] = 144
    fig_dir = "./figs"
    decay_kwargs = namedtuple(
        "decay_kwargs",
        ["schedule", "start", "end", "timesteps", "return_function"],
        defaults=[None, False])

    beta_start = 0.001
    logsnr_start, logsnr_end = 10, -12
    signal_mean = 0.4734  # CIFAR10
    lbd = infer_lbd_from_logsnr(logsnr_start, "ordinal_jump", signal_mean)
    timesteps = 1000

    other_kwargs = dict(
        diffusion_type="ordinal_jump",
        lbd=lbd,
        signal_stat=signal_mean
    )

    alpha_end = infer_alpha_from_logsnr(logsnr_end, **other_kwargs)
    beta_end = infer_beta_end_from_logsnr(logsnr_end, beta_start, timesteps=timesteps, **other_kwargs)
    beta_end_ = infer_beta_end_from_logsnr(logsnr_end, 0.05, timesteps=timesteps, **other_kwargs)
    test_cases = (
        decay_kwargs("alpha_cosine", 1, alpha_end, timesteps, False),
        decay_kwargs("alpha_cosine", 1, alpha_end, timesteps, True),
        decay_kwargs("alpha_cosine_improved", 1, alpha_end, timesteps, False),
        decay_kwargs("alpha_cosine2", 1, alpha_end, timesteps, False),
        decay_kwargs("alpha_cosine2", 1, alpha_end, timesteps, True),
        decay_kwargs("alpha_sigmoid", 0.9999, alpha_end, timesteps, False),
        decay_kwargs("beta_linear", beta_start, beta_end, timesteps, False),
        decay_kwargs("beta_linear", beta_start, beta_end, timesteps, True),
        decay_kwargs("beta_linear2", beta_start, beta_end, timesteps, False),
        decay_kwargs("logsnr_cosine", logsnr_start, logsnr_end, timesteps, False),
        decay_kwargs("logsnr_cosine2", logsnr_start, logsnr_end, timesteps, False),
        decay_kwargs("logsnr_cosine2", logsnr_start, logsnr_end, timesteps, True),
    )

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    def alpha2logsnr(alpha, diffusion_type, lbd, signal_stat):
        assert diffusion_type in {"ordinal_jump", "gaussian"}
        if diffusion_type == "ordinal_jump":
            return torch.log(alpha * lbd * signal_stat)
        else:
            return torch.logit(alpha ** 2)

    fig_alpha, ax_alpha = plt.subplots(figsize=(6, 6), dpi=144)
    fig_logsnr, ax_logsnr = plt.subplots(figsize=(6, 6), dpi=144)

    for case in test_cases:
        if case.return_function:
            t = torch.linspace(1 / 1000, 1, 1000, dtype=torch.float64)
            alphas = get_decay_schedule(
                **case._asdict(), **other_kwargs
            )[0]["alpha_fn"](t)
            ax_alpha.plot(alphas, label=case.schedule + "(cont.)", linestyle="--")
            ax_logsnr.plot(alpha2logsnr(alphas, **other_kwargs), label=case.schedule + "(cont.)", linestyle="--")
        else:
            alphas = get_decay_schedule(**case._asdict(), **other_kwargs)[0]["alphas"]
            ax_alpha.plot(alphas, label=case.schedule)
            ax_logsnr.plot(alpha2logsnr(alphas, **other_kwargs), label=case.schedule)
    ax_alpha.legend()
    ax_logsnr.legend()
    fig_alpha.savefig(os.path.join(fig_dir, "alpha_schedules.png"), bbox_inches="tight")
    fig_logsnr.savefig(os.path.join(fig_dir, "logsnr_schedules.png"), bbox_inches="tight")
