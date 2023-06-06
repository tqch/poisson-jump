import math
import numpy as np
import torch
import torch.nn as nn
import weakref
from collections.abc import Iterable
from itertools import zip_longest

__all__ = [
    "get_transform",
    "get_activation",
    "FuncChainer",
    "ModelWrapper",
    "DummyWriter",
    "sym_sqrt_scale_functions",
    "resume_from_chkpt",
    "EMA",
    "dotteddict",
    "EncodeDict",
    "TrainDict",
    "SaveDict",
    "DataDict"
]


def get_transform(name):
    if name == "anscombe":
        def forward(x):
            return 2 * (torch.sqrt(x + 3. / 8) - math.sqrt(3. / 8))

        def backward(x):
            return (x / 2 + math.sqrt(3. / 8)) ** 2 - 3. / 8

    elif name == "freeman-tukey":
        def forward(x):
            return torch.sqrt(1. + x) + torch.sqrt(x)

        def backward(x):
            return ((x ** 2 - 1) / (2 * x)) ** 2

    elif name == "log":
        def forward(x):
            return torch.log(x + 1.)

        def backward(x):
            return torch.exp(x) - 1.

    elif name == "normalize":

        def forward(x):
            return (x - 0.5).mul(2.)

        def backward(x):
            return (x + 1.).div(2.)

    else:
        return None, None

    forward.__name__ = f"{name}_forward"
    backward.__name__ = f"{name}_backward"

    return forward, backward


def get_activation(act):

    def none():
        return None

    return {"none": none, "relu": nn.ReLU, "softplus": nn.Softplus, "sigmoid": nn.Sigmoid}[act]()


class FuncChainer:
    def __init__(self, func):
        if not isinstance(func, Iterable):
            func_list = (func, )
        else:
            func_list = func
        self.func_list = func_list

    def __call__(self, x):
        for func in self.func_list:
            if func is None:
                continue
            x = func(x)
        return x


class ModelWrapper(nn.Module):
    def __init__(
            self,
            model,
            pre_transform=None,
            post_transform=None
    ):
        super().__init__()
        self._model = model
        self.pre_transform = pre_transform
        self.post_transform = post_transform

    def forward(self, x, **kwargs):
        if self.pre_transform is not None:
            x = self.pre_transform(x)
        out = self._model(x, **kwargs)
        if self.post_transform is not None:
            out = self.post_transform(out)
        return out


class DummyWriter:
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def add_text(self, *args, **kwargs):
        pass

    def add_image(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass


def sym_sqrt_scale_functions():
    def forward(arr):
        return np.sqrt(np.abs(arr)) * np.sign(arr)

    def backward(arr):
        return np.power(arr, 2) * np.sign(arr)

    return forward, backward


def resume_from_chkpt(chkpt, **kwargs):
    for k, v in kwargs.items():
        try:
            if v is not None and hasattr(v, "load_state_dict"):
                try:
                    v.load_state_dict(chkpt[k])
                except RuntimeError:
                    _chkpt = chkpt[k]["shadow"] if k == "ema" else chkpt[k]
                    for kk in tuple(_chkpt.keys()):
                        if kk.split(".")[0] == "module":
                            _chkpt[kk.split(".", maxsplit=1)[1]] = _chkpt.pop(kk)
                    v.load_state_dict(chkpt[k])
                    del _chkpt
                finally:
                    del chkpt[k]
        except KeyError:
            continue
    start_epoch = chkpt.pop("epoch", 0) or chkpt.pop("_epoch", 0)
    return start_epoch


class EMA:
    """
    exponential moving average
    inspired by:
    [1] https://github.com/fadel/pytorch_ema
    [2] https://github.com/tensorflow/tensorflow/blob/v2.9.1/tensorflow/python/training/moving_averages.py#L281-L685
    """

    def __init__(self, model, decay=0.9999):
        shadow = []
        refs = []
        for k, v in model.named_parameters():
            if v.requires_grad:
                shadow.append((k, v.detach().clone()))
                refs.append((k, weakref.ref(v)))
        self.shadow = dict(shadow)
        self._refs = dict(refs)
        self.decay = decay
        self.num_updates = 0
        self.backup = None

    def update(self):
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for k, _ref in self._refs.items():
            assert _ref() is not None, "referenced object no longer exists!"
            self.shadow[k] += (1 - decay) * (_ref().data - self.shadow[k])

    def apply(self):
        self.backup = dict([
            (k, _ref().detach().clone()) for k, _ref in self._refs.items()])
        for k, _ref in self._refs.items():
            _ref().data.copy_(self.shadow[k])

    def restore(self):
        for k, _ref in self._refs.items():
            _ref().data.copy_(self.backup[k])
        self.backup = None

    def __enter__(self):
        self.apply()

    def __exit__(self, *exc):
        self.restore()

    def state_dict(self):
        return {
            "decay": self.decay,
            "shadow": self.shadow,
            "num_updates": self.num_updates
        }

    @property
    def extra_states(self):
        return {"decay", "num_updates"}

    def load_state_dict(self, state_dict, strict=True):
        _dict_keys = set(self.__dict__["shadow"]).union(self.extra_states)
        dict_keys = set(state_dict["shadow"]).union(self.extra_states)
        incompatible_keys = set.symmetric_difference(_dict_keys, dict_keys) \
            if strict else set.difference(_dict_keys, dict_keys)
        if incompatible_keys:
            raise RuntimeError(
                "Key mismatch!\n"
                f"Missing key(s): {', '.join(set.difference(_dict_keys, dict_keys))}."
                f"Unexpected key(s): {', '.join(set.difference(dict_keys, _dict_keys))}"
            )
        self.__dict__.update(state_dict)


class DottedDict(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getattr__(self, name):
        if name not in self:
            raise AttributeError(name)
        return self[name]

    def __setattr__(self, name, val):
        self[name] = val


def dotteddict(names, values=tuple(), default=None, class_name=""):
    """
    Factory function of DottedDict class with pre-defined keys
    """
    class NewDottedDict(DottedDict):
        def __init__(self):
            super().__init__(names=set(names))
            for name, val in zip_longest(names, values, fillvalue=default):
                self[name] = val

        def __setattr__(self, name, val):
            if name in self.names:
                self[name] = val
            else:
                raise KeyError(f"Invalid key: {name}!")

        def __setitem__(self, name, val):
            if name in self.names:
                return super().__setitem__(name, val)
            else:
                raise KeyError(f"Invalid key: {name}!")

        def init(self, **kwargs):
            for k, v in kwargs.items():
                self[k] = v
            return self

    if class_name:
        NewDottedDict.__name__ = class_name

    return NewDottedDict


EncodeDict = dotteddict(names=("type", "lbd", "timesteps", "continuous"), class_name="EncodeDict")

TrainDict = dotteddict(names=(
    "trainloader", "diffusion", "model", "optimizer", "scheduler", "ema",
    "num_accum", "grad_norm", "writer", "start_epoch", "epochs", "mode", "use_proj",
    "is_leader", "device", "verbose", "seed"), class_name="TrainDict")

SaveDict = dotteddict(names=(
    "num_samples", "topk", "ndocs", "z_T", "xsqrt", "eval_intv", "eval_batch_size", "use_pred",
    "chkpt_intv", "image_dir", "text_dir", "chkpt_dir", "log_dir", "exp_name"), class_name="SaveDict")

DataDict = dotteddict(names=("dataset", "input_shape", "is_real", "is_bow", "is_ml", "is_image"), class_name="DataDict")


if __name__ == "__main__":
    LowerDict = dotteddict([chr(i+ord("a")) for i in range(26)], tuple(range(10)), class_name="LowerDict")
    lower_dict = LowerDict()
    print(lower_dict)
    print(lower_dict.__class__.__name__)
