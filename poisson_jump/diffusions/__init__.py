from inspect import isclass
from .base import BaseDiffusion
from .jump import *
from .gaussian import *

__all__ = []
DIFFUSION_DICT = {}
local_copy = locals().copy()


def register_diffusion():
    for cls_name, cls in local_copy.items():
        if isclass(cls) and issubclass(cls, BaseDiffusion):
            __all__.append(cls_name)
            if hasattr(cls, "name"):
                DIFFUSION_DICT[cls.name] = cls
            if hasattr(cls, "alias"):
                if cls.alias is not None:
                    DIFFUSION_DICT[cls.alias] = cls


register_diffusion()
del local_copy
__all__.append("DIFFUSION_DICT")

