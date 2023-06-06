import torch.nn as nn
from functools import partial
try:
    from .functions import get_timestep_embedding
    from .modules import Linear, Sequential
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from poisson_jump.nets.functions import get_timestep_embedding
    from poisson_jump.nets.modules import Linear, Sequential
from collections.abc import Iterable
from itertools import repeat


NONLINEARITY = partial(nn.LeakyReLU, negative_slope=0.02)
NORMALIZER = nn.LayerNorm


class MLP(nn.Module):
    def __init__(
            self,
            in_dim=1,
            base_dim=64,
            out_dim=None,
            multiplier=1,
            num_layers=3,
            drop_rate=0.
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.drop_rate = drop_rate

        self.in_fc = Linear(in_dim, base_dim)
        if not isinstance(multiplier, Iterable):
            multiplier = tuple(repeat(multiplier, num_layers))
        for i in range(num_layers):
            self.layers.append(self.make_layer(
                base_dim * (multiplier[i - 1] if i else 1),
                base_dim * multiplier[i]))
        out_dim = out_dim or in_dim
        self.out_fc = Sequential(
            NONLINEARITY(), Linear(base_dim * multiplier[-1], out_dim))

    def make_layer(self, in_dim, out_dim):
        return nn.Sequential(
            NORMALIZER(in_dim),
            NONLINEARITY(),
            nn.Dropout(p=self.drop_rate),
            Linear(in_dim, out_dim))

    def forward(self, x):
        out = self.in_fc(x)
        for layer in self.layers:
            out = layer(out)
        return self.out_fc(out)


class ConditionalLayer(nn.Module):
    def __init__(self, in_dim, out_dim, temb_dim, drop_rate=0.):
        super().__init__()
        self.act = NONLINEARITY()
        self.in_norm = NORMALIZER(in_dim)
        self.in_fc = Linear(in_dim, out_dim)
        self.out_norm = NORMALIZER(out_dim)
        self.out_fc = Linear(out_dim, out_dim)
        self.proj = Linear(temb_dim, out_dim)
        self.dropout = nn.Dropout1d(drop_rate) if drop_rate else nn.Identity()

    def forward(self, x, t_emb=None):
        out = self.in_fc(self.act(self.in_norm(x)))
        if t_emb is not None:
            out = out + self.proj(self.act(t_emb))
        return self.out_fc(self.dropout(self.act(self.out_norm(out))))


class ConditionalMLP(nn.Module):
    def __init__(
            self,
            in_dim=1,
            base_dim=64,
            out_dim=None,
            multiplier=1,
            temb_dim=None,
            num_layers=3,
            drop_rate=0.,
            continuous_t=False
    ):
        super().__init__()
        self.base_dim = base_dim
        temb_dim = temb_dim or (4 * base_dim)
        self.embed = nn.Sequential(
            Linear(base_dim, temb_dim),
            NONLINEARITY(),
            Linear(temb_dim, temb_dim)
        )
        self.temb_dim = temb_dim
        self.in_fc = Linear(in_dim, base_dim)
        self.layers = nn.ModuleList()

        if not isinstance(multiplier, Iterable):
            multiplier = tuple(repeat(multiplier, num_layers))
        for i in range(num_layers):
            self.layers.append(ConditionalLayer(
                base_dim * (multiplier[i - 1] if i else 1),
                base_dim * multiplier[i], temb_dim, drop_rate=drop_rate))

        out_dim = out_dim or in_dim
        self.out_fc = nn.Sequential(NONLINEARITY(), Linear(base_dim * multiplier[-1], out_dim))

        self.continuous_t = continuous_t

    def forward(self, x, t=None):
        if t is not None:
            if self.continuous_t:
                t = 1000 * t
            t_emb = get_timestep_embedding(t, self.base_dim)
            t_emb = self.embed(t_emb)
        else:
            t_emb = None
        out = self.in_fc(x)
        for layer in self.layers:
            out = layer(out, t_emb)
        return self.out_fc(out)
