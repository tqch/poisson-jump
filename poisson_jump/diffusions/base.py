import torch
from abc import ABC, abstractmethod


class BaseDiffusion(ABC):
    timesteps: int = None
    momentum: float = 0.

    @abstractmethod
    def q_sample(self, x_0, t, return_eps=None):  # q(z_t | x_0)
        pass

    @abstractmethod
    def q_posterior(self, z_t, x_0, t, eps=None):  # q(z_{t-1} | z_t, x_0)
        pass

    @abstractmethod
    def p_sample_step(self, model, z_t, t, accum_x_0=None, return_pred=False, rng=None):  # p(z_{t-1} | z_t)
        pass

    @torch.inference_mode()
    def p_sample(self, model, z_T, seed=None, return_pred=False):
        z_t = z_T
        pred_x_0 = torch.zeros_like(z_t)
        accum_x_0 = torch.zeros_like(z_t)
        rng = None
        if seed is not None:
            device = next(model.parameters()).device
            rng = torch.Generator(device).manual_seed(seed)
        for t in range(self.timesteps - 1, -1, -1):
            if accum_x_0 is not None:
                accum_x_0 += (1 - self.momentum) * (pred_x_0 - accum_x_0)
            z_t, pred_x_0 = self.p_sample_step(
                model, z_t, t, accum_x_0=accum_x_0, return_pred=True, rng=rng)
        x_0 = z_t
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
        sample_path[n] = z_t.cpu()
        return sample_path

    @abstractmethod
    def train_loss(self, model, x_0, t):
        pass
