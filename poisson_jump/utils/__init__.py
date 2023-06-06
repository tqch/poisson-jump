import numpy as np
import random
import torch
from .train import *

__all__ = ["seed_all", "dict2str", "safe_get"]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dict2str(d, num_indent=0):
    out_str = []
    indents = "  " * num_indent
    for i, (k, v) in enumerate(d.items()):
        line = indents + str(k) + ": "
        if isinstance(v, str):
            line += v
        elif isinstance(v, float):
            line += f"{v:.3e}"
        elif isinstance(v, dict):
            line += "{\n" + dict2str(v, num_indent + 1)
            line += indents + "}"
        else:
            line += str(v)
        if i != len(d) - 1:
            line += ","
        line += "\n"
        out_str.append(line)
    return "".join(out_str)


def safe_get(obj, name, default=None):
    try:
        return getattr(obj, name)
    except AttributeError:
        return default


if __name__ == "__main__":
    print(dict2str({
      "unet": {
        "in_channels": 3,
        "hid_channels": 128,
        "out_channels": 3,
        "ch_multipliers": [1, 1, 1],
        "num_res_blocks": 3,
        "apply_attn": [False, True, True],
        "drop_rate": 0.2
      }}))
