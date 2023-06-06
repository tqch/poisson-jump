import csv
import hashlib
import logging
import math
import numpy as np
import os
import pandas as pd
import pickle
import PIL.Image
import scipy.sparse as sp
import scipy.stats as stats
import scipy.special as special
import sklearn.datasets as skds
import torch
import torchvision.datasets as tvds
import torch.utils.data as data_utils
import wget
from abc import ABC, abstractmethod
from collections import namedtuple
from functools import partial
from poisson_jump.metrics import data_emd
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from zipfile import ZipFile
from .utils import safe_get

__all__ = ["DATASET_CONFIGS", "DATASET_DICT", "get_dataloader", "isimage", "isreal", "BOWDataset", "MovieLensBase"]

DataLoader2D = data_utils.DataLoader
ROOT = os.path.expanduser("~/datasets")
CSV = namedtuple("CSV", ["header", "index", "data"])
DATASET_DICT = dict()
IMAGE_DATASETS = set()
REAL_DATASETS = set()
INT_DATASETS = set()
OTHER_DATASETS = set()


def register_dataset(dataset):
    try:
        name = dataset.name
    except AttributeError:
        name = dataset.__name__
    DATASET_DICT[name] = dataset
    if hasattr(dataset, "data_type"):
        data_type = dataset.data_type
    else:
        data_type = "other"
    {
        "image": IMAGE_DATASETS,
        "real": REAL_DATASETS,
        "int": INT_DATASETS
    }.get(data_type, OTHER_DATASETS).add(name)
    return dataset


def isimage(dataset):
    return dataset in IMAGE_DATASETS


def isreal(dataset):
    return dataset in REAL_DATASETS


def crop_celeba(img):
    # the cropping parameters match the ones used in DDIM
    return transforms.functional.crop(img, top=57, left=25, height=128, width=128)  # noqa


DEFAULT_SIZE = 100000
DATASET_CONFIGS = {
    "cat": {
        "probs": [1, 3, 1, 5, 0, 0, 10, 3, 0, 1, 0],
        "size": DEFAULT_SIZE  # default size is 100k for toy data
    },
    "nbinom": {
        "ns": [1, 10],
        "ps": [0.9, 0.1],
        "mix_coefs": [0.75, 0.25],
        "size": DEFAULT_SIZE
    },
    "poisson": {
        "mus": [1, 100],
        "mix_coefs": [0.9, 0.1],
        "size": DEFAULT_SIZE
    },
    "beta": {
        "a": 0.5,
        "b": 0.5,
        "size": DEFAULT_SIZE
    },
    "bnb": {
        "a": 1.5,
        "b": 1.5,
        "n": 1,
        "size": DEFAULT_SIZE
    },
    "fcauchy": {
        "c": 0,
        "size": DEFAULT_SIZE
    },
    "fstudent": {
        "df": 2,
        "size": DEFAULT_SIZE
    },
    "gamma": {
        "a": 1,
        "b": 0.1,
        "size": DEFAULT_SIZE
    },
    "norm": {
        "loc": 0.,
        "scale": 1.,
        "mean": {
            "norm": (0.7979, 0.7979)  # sqrt(2 / pi)
        }
    },
    "20news": {
        "subset": "all"
    },
    "nips": {
        "root": "./data"
    },
    "mnist": {
        "train": True,
        # normalize to [0, 1]
        # despite being re-scaled, the discrete nature of pixel values remains unchanged
        # the normalization is for numerical stability reason
        "out_type": "0-1"
    },
    "cifar10": {
        "train": True,
        "hflip": True,
        "out_type": "0-1"
    },
    "celeba": {
        "split": "all",
        "hflip": True,
        "out_type": "0-1"
    },
    "ml_1m": {
        "root": "./data",
        "binary": True
    },
    "ml_10m": {
        "root": "./data",
        "binary": True
    }
}


class BaseDataset(ABC):
    def __init__(self, size, random_state=None):
        self.size = size
        if random_state is None:
            random_state = 1234
        self.random_state = random_state
        self._data = self.sample()

    @property
    def data(self):
        return self._data

    @property
    def max(self):
        return self._data.max()

    @property
    def mean(self):
        return self._data.mean()

    def resample(self):
        self._data = self.sample()

    @abstractmethod
    def sample(self):
        pass

    @property
    @abstractmethod
    def shape(self):
        pass

    def __len__(self):
        return self.size

    def shuffle(self, data):
        np.random.seed(self.random_state)
        np.random.shuffle(data)
        self.random_state = (self.random_state + 71) % (2 ** 32)


def get_modes(arr, a_min=0.):
    return np.where(np.logical_and(
        np.diff(arr, n=1, append=a_min) <= 0, np.diff(arr, n=1, prepend=a_min) >= 0))


def get_scale_functions(scale):
    def forward(arr):
        return np.log(np.abs(arr) + scale) - np.log(scale)

    def inverse(arr):
        return np.exp(arr + np.log(scale)) - scale

    return forward, inverse


def one_hot(array, num_classes=-1):
    if num_classes == -1:
        num_classes = np.max(array) + 1
    encoding = np.zeros((len(array), num_classes))
    encoding[np.arange(len(array)), array] = 1
    return encoding


@register_dataset
class Categorical(BaseDataset):
    name = "cat"
    shape = (1, )
    data_type = "cat"

    def __init__(self, size, probs=None, logits=None, use_gumbel=False, random_state=None):
        assert not (probs is None and logits is None)
        if probs is not None:
            assert all(p >= 0. for p in probs)
            probs = np.array(probs, dtype=np.float64)
            probs /= np.sum(probs)  # sum to 1
            logits = np.log(probs / np.max(probs))
        else:
            probs = np.exp(logits)
            probs /= np.sum(probs)
        self.probs = probs
        self.logits = logits
        self.use_gumbel = use_gumbel  # gumbel softmax
        self.num_classes = len(self.probs)
        super().__init__(size, random_state)

    def sample(self):
        if self.use_gumbel:
            data = one_hot(stats.gumbel_r.rvs(
                size=(self.size, self.num_classes),
                loc=[self.logits], random_state=self.random_state).argmax(axis=1))
        else:
            data = stats.multinomial.rvs(
                size=self.size, n=1, p=self.probs, random_state=self.random_state)
        self.shuffle(data)
        return data

    def plot(self, ax, bar=False, **plot_kwargs):
        x_min, x_max = 0, self.num_classes - 1
        xlim = x_min - 0.5, x_max + 0.5
        ax.set_xlim(xlim)
        xs = np.arange(self.num_classes)
        ys = self.probs
        ax.set_ylim((0, np.max(ys) * 1.2))
        mode_min = np.min(ys[get_modes(ys)])
        if bar:
            ax.bar(xs, ys, width=1, label="True PMF", **plot_kwargs)
        else:
            ax.plot(xs, ys, label="True PMF", **plot_kwargs)
        return get_scale_functions(mode_min)

    @property
    def mean(self):
        return 1. / self.num_classes


@register_dataset
class FoldedCauchy(BaseDataset):
    name = "fcauchy"
    shape = (1, )
    data_type = "real"

    def __init__(self, size, c, random_state=None):
        self.c = c
        super().__init__(size, random_state)

    def sample(self):
        data = stats.foldcauchy.rvs(size=self.size, c=self.c, random_state=self.random_state)
        self.shuffle(data)
        return data

    def density(self, xs: np.array) -> np.array:
        return stats.foldcauchy.pdf(xs, c=self.c)

    def plot(self, ax, **plot_kwargs):
        upper = stats.foldcauchy.ppf(0.999, c=self.c)
        xs = np.linspace(0, upper, 1000)
        mar = 0.05
        xlim = -mar, upper + mar
        ax.set_xlim(xlim)
        ys = self.density(xs)
        ax.plot(xs, ys, label="True PDF", **plot_kwargs)
        return None


@register_dataset
class FoldedStudent(BaseDataset):
    name = "fstudent"
    shape = (1, )
    data_type = "real"

    def __init__(self, size, df, random_state=None):
        self.df = df
        super().__init__(size, random_state)

    def sample(self):
        data = np.abs(stats.t.rvs(
            size=self.size, df=self.df, loc=0, scale=1, random_state=self.random_state))
        self.shuffle(data)
        return data

    def density(self, xs: np.array) -> np.array:
        return stats.t.pdf(xs, df=self.df, loc=0, scale=1) * 2

    def plot(self, ax, **plot_kwargs):
        upper = stats.t.ppf(0.9995, df=self.df, loc=0, scale=1)
        xs = np.linspace(0, upper, 1000)
        mar = 0.05
        xlim = -mar, upper + mar
        ax.set_xlim(xlim)
        ys = self.density(xs)
        ax.plot(xs, ys, label="True PDF", **plot_kwargs)
        return None


@register_dataset
class Gamma(BaseDataset):
    name = "gamma"
    shape = (1, )
    data_type = "real"

    def __init__(self, size, a, b, random_state=None):
        self.a = a
        self.b = b
        super().__init__(size, random_state)

    def sample(self):
        data = stats.gamma.rvs(
            size=self.size, a=self.a, loc=0, scale=1/self.b, random_state=self.random_state)
        self.shuffle(data)
        return data

    def density(self, xs: np.array) -> np.array:
        return stats.gamma.pdf(xs, a=self.a, loc=0, scale=1/self.b)

    def plot(self, ax, **plot_kwargs):
        upper = stats.gamma.ppf(0.999, a=self.a, loc=0, scale=1 / self.b)
        xs = np.linspace(0, upper, 1000)
        mar = 0.05
        xlim = -mar, upper + mar
        ax.set_xlim(xlim)
        ys = self.density(xs)
        ax.plot(xs, ys, label="True PDF", **plot_kwargs)
        return None


@register_dataset
class NegativeBinomialMixture(BaseDataset):
    name = "nbinom"
    shape = (1, )
    data_type = "int"

    def __init__(self, size, ns, ps, mix_coefs=None, random_state=None):
        self.ns = ns
        self.ps = ps
        self.mix_coefs = [1. / len(ns) for _ in range(len(ns))] \
            if mix_coefs is None else mix_coefs
        super().__init__(size, random_state)

    def sample(self):
        sizes = stats.multinomial.rvs(
            self.size, self.mix_coefs, random_state=self.random_state)
        data = [stats.nbinom.rvs(
            n=n, p=p, size=sz, random_state=self.random_state + 1)
            for n, p, sz in zip(self.ns, self.ps, sizes)]
        data = np.concatenate(data, axis=0)
        self.shuffle(data)
        return data

    @property
    def n_components(self):
        return len(self.mix_coefs)

    def density(self, xs: np.array) -> np.array:
        dsty = 0
        for coef, n, p in zip(self.mix_coefs, self.ns, self.ps):
            x_max = stats.nbinom.ppf(1-1e-12, n=n, p=p)
            _dsty = stats.nbinom.pmf(np.clip(
                xs, a_min=None, a_max=x_max), n=n, p=p)
            dsty += coef * np.where(xs <= x_max, _dsty, 0)
        return dsty

    def plot(self, ax, bar=False, **plot_kwargs):
        x_min, x_max = np.inf, -np.inf
        for n, p in zip(self.ns, self.ps):
            x_min = min(x_min, stats.nbinom.ppf(1e-6, n=n, p=p))
            x_max = max(x_max, stats.nbinom.ppf(1-1e-6, n=n, p=p))
        xlim = x_min - 0.5, x_max + 0.5
        ax.set_xlim(xlim)
        xs = np.arange(x_min, x_max + 1)
        ys = self.density(xs)
        ax.set_ylim((0, np.max(ys) * 1.2))
        mode_min = np.min(ys[get_modes(ys)])  # noqa
        if bar:
            ax.bar(xs, ys, width=1, label="True PMF", **plot_kwargs)
        else:
            ax.plot(xs, ys, label="True PMF", **plot_kwargs)
        return get_scale_functions(mode_min)


@register_dataset
class PoissonMixture(BaseDataset):
    name = "poisson"
    shape = (1, )
    data_type = "int"

    def __init__(self, size, mus, mix_coefs=None, random_state=None):
        self.mus = mus
        self.mix_coefs = [1. / len(mus) for _ in range(len(mus))] \
            if mix_coefs is None else mix_coefs
        super().__init__(size, random_state)

    def sample(self):
        sizes = stats.multinomial.rvs(
                n=self.size, p=self.mix_coefs, random_state=self.random_state)
        data = [stats.poisson.rvs(
            mu, size=sz, random_state=self.random_state + 1)
            for mu, sz in zip(self.mus, sizes)]
        data = np.concatenate(data, axis=0)
        self.shuffle(data)
        return data

    @property
    def n_components(self):
        return len(self.mix_coefs)

    def density(self, xs: np.array) -> np.array:
        dsty = 0
        for coef, mu in zip(self.mix_coefs, self.mus):
            dsty += coef * stats.poisson.pmf(xs, mu)
        return dsty

    def plot(self, ax, bar=False, **plot_kwargs):
        x_min, x_max = math.floor(stats.poisson.ppf(1e-4, min(self.mus))),\
               math.ceil(stats.poisson.ppf(1-1e-4, max(self.mus)))
        xlim = x_min - 0.5, x_max + 0.5
        ax.set_xlim(xlim)
        xs = np.arange(x_min, x_max + 1)
        ys = self.density(xs)
        ax.set_ylim((0, np.max(ys) * 1.2))
        mode_min = np.min(ys[get_modes(ys)])  # noqa
        if bar:
            ax.bar(xs, ys, width=1, label="True PMF", **plot_kwargs)
        else:
            ax.plot(xs, ys, label="True PMF", **plot_kwargs)
        return get_scale_functions(mode_min)


@register_dataset
class Beta(BaseDataset):
    name = "beta"
    shape = (1, )
    data_type = "real"

    def __init__(self, size, a, b, random_state=None):
        self.a = a
        self.b = b
        super().__init__(size, random_state)

    def sample(self):
        data = stats.beta.rvs(
            size=self.size, a=self.a, b=self.b, random_state=self.random_state)
        self.shuffle(data)
        return data

    def density(self, xs: np.array) -> np.array:
        return stats.beta.pdf(xs, a=self.a, b=self.b)

    def plot(self, ax, **plot_kwargs):
        xs = np.linspace(0, 1, 100)
        xlim = - 0.05, 1.05
        ax.set_xlim(xlim)
        ys = self.density(xs)
        ax.plot(xs, ys, label="True PDF", **plot_kwargs)
        upper_thres = max(stats.iqr(ys, rng=(5, 95)), 0.05) + np.median(ys)  # noqa
        ax.set_ylim((0, min(np.max(ys) * 1.2, upper_thres)))
        return None


def clip(x, clip_range, boundary="trunc"):
    if clip_range is None:
        return x
    else:
        if boundary == "trunc":
            return np.clip(x, *clip_range)
        elif boundary == "reflect":
            return 2 * np.clip(x, *clip_range) - x
        else:
            raise NotImplementedError(boundary)


def rand_like(x):
    if isinstance(x, np.ndarray):
        return 2 * np.random.rand(*x.shape).astype(x.dtype) - 1
    else:
        return 2 * torch.rand_like(x) - 1


def randn_like(x):
    if isinstance(x, np.ndarray):
        return np.random.randn(*x.shape).astype(x.dtype)
    else:
        return torch.randn_like(x)


def get_smoothing_transform(type, bandwidth, **clip_kwargs):
    def transform(x):
        return clip(x + bandwidth * {
            "uniform": rand_like,
            "gaussian": randn_like
        }[type](x), **clip_kwargs)

    return transform


@register_dataset
class DiscretizedBeta(Beta):
    name = "dbeta"
    shape = (1, )
    data_type = "int"
    transform = None

    def __init__(self, size, a, b, k: int = 10, random_state=None, smooth_dict=None):
        self.k = k
        if smooth_dict is not None:
            self.transform = get_smoothing_transform(**smooth_dict)
        super().__init__(size, a, b, random_state)

    def sample(self):
        data = super().sample()
        data = np.round(data * self.k, 0)
        return data

    def density(self, xs: np.array) -> np.array:
        splits = np.linspace(.5 / self.k, 1 - .5 / self.k, self.k)
        return np.diff(stats.beta.cdf(splits, a=self.a, b=self.b), prepend=0, append=1)

    def plot(self, ax, bar=False, **plot_kwargs):
        xlim = - 0.5, self.k + 0.5
        ax.set_xlim(xlim)
        xs = np.arange(0, self.k + 1)
        ys = self.density(xs)
        ax.set_ylim((0, np.max(ys) * 1.2))
        mode_min = np.min(ys[get_modes(ys)])  # noqa
        if bar:
            ax.bar(xs, ys, width=1, label="True PMF", **plot_kwargs)
        else:
            ax.plot(xs, ys, label="True PMF", **plot_kwargs)
        return get_scale_functions(mode_min)


@register_dataset
class BetaNbinom(BaseDataset):
    name = "bnb"
    shape = (1, )
    data_type = "int"

    def __init__(self, size, a, b, n, random_state=None):
        self.a = a
        self.b = b
        self.n = n
        super().__init__(size, random_state)

    def sample(self):
        p = stats.beta.rvs(size=self.size, a=self.a, b=self.b, random_state=self.random_state)
        data = stats.nbinom.rvs(n=self.n, p=p, random_state=self.random_state + 1)
        self.shuffle(data)
        return data

    def density(self, xs: np.array) -> np.array:
        dsty = special.betaln(self.n + xs, self.a + self.b) \
            - special.betaln(self.n, self.a) + special.gammaln(self.b + xs) \
            - special.gammaln(1 + xs) - special.gammaln(self.b)
        dsty = np.exp(dsty)
        return dsty

    def plot(self, ax, bar=False, **plot_kwargs):
        upper = np.quantile(self.data, 0.999)
        xlim = - 0.5, upper + 0.5
        ax.set_xlim(xlim)
        xs = np.arange(0, upper + 1)
        ys = self.density(xs)
        ax.set_ylim((0, np.max(ys) * 1.2))
        mode_min = np.min(ys[get_modes(ys)])  # noqa
        if bar:
            ax.bar(xs, ys, width=1, label="True PMF", **plot_kwargs)
        else:
            ax.plot(xs, ys, label="True PMF", **plot_kwargs)
        return get_scale_functions(mode_min)


@register_dataset
class Normal(BaseDataset):
    name = "norm"
    shape = (1, )
    data_type = "real"

    def __init__(self, size, loc, scale, random_state=None):
        self.loc = loc
        self.scale = scale
        super().__init__(size, random_state)

    def sample(self):
        data = stats.norm.rvs(
            size=self.size, loc=self.loc, scale=self.scale, random_state=self.random_state)
        self.shuffle(data)
        return data

    def density(self, xs: np.array) -> np.array:
        return stats.norm.pdf(xs, loc=self.loc, scale=self.scale)

    def plot(self, ax, **plot_kwargs):
        xs = np.linspace(self.loc - 3 * self.scale, self.loc + 3 * self.scale, 201)
        ys = self.density(xs)
        ax.plot(xs, ys, label="True PDF", **plot_kwargs)
        upper_thres = stats.iqr(ys, rng=(5, 95)) + np.median(ys)  # noqa
        ax.set_ylim((0, min(ys[100] * 1.2, upper_thres)))
        return None


bilinear = transforms.InterpolationMode.BILINEAR


def to_float32(x):
    return x.to(torch.float32)


def to_numpy(x):
    return np.array(x).transpose((2, 0, 1))


def transform_patch(transform, out_type="numpy", smooth_dict=None):
    if out_type == "0-1":
        transform.append(transforms.ToTensor())
    elif out_type == "norm":
        transform.extend([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])
    elif out_type == "raw":
        transform.extend([transforms.PILToTensor(), to_float32])
    elif out_type == "numpy":
        transform.append(to_numpy)
    elif out_type == "smooth_0-1":
        assert smooth_dict is not None, "smooth_dict must be provided!"
        transform.extend([
            transforms.PILToTensor(), to_float32,
            get_smoothing_transform(**smooth_dict),
            partial(torch.div, other=255.)
        ])
    else:
        raise NotImplementedError(out_type)


class ImageFolder(tvds.ImageFolder):
    def find_classes(self, directory):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            return [None, ], {None: 0}
        else:
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
            return classes, class_to_idx

    def make_dataset(self, directory, class_to_idx, extensions=None, is_valid_file=None):
        if class_to_idx is None:
            raise ValueError("The class_to_idx parameter cannot be None.")
        directory = os.path.expanduser(directory)

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:
            def is_valid_file(x: str) -> bool:
                return x.endswith(tuple(extensions))  # type: ignore[arg-type]

        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class) if target_class else directory
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)

                        if target_class not in available_classes:
                            available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances


@register_dataset
class MNIST(tvds.MNIST):
    name = "mnist"
    shape = (1, 32, 32)
    mean_dict = {
        "0-1": 0.1307,
        "norm": 0.9136,
        "smooth_0-1": 0.1307,
    }
    peak = 1.
    data_type = "image"

    def __init__(self, root=ROOT, train=True, out_type="numpy", smooth_dict=None, **kwargs):
        transform = [transforms.Resize((32, 32), interpolation=bilinear)]
        transform_patch(transform, out_type=out_type, smooth_dict=smooth_dict)
        transform = transforms.Compose(transform)
        super().__init__(root=root, train=train, transform=transform)

        self.out_type = out_type
        self.size = len(self.data)

    @property
    def mean(self):
        return self.mean_dict.get(self.out_type, None)


@register_dataset
class CIFAR10(tvds.CIFAR10):
    name = "cifar10"
    shape = (3, 32, 32)
    mean_dict = {
        "0-1": 0.4734,
        "norm": 0.4235,
        "smooth_0-1": 0.4734,
    }
    peak = 1.
    data_type = "image"

    def __init__(self, root=ROOT, train=True, hflip=False, out_type="numpy", smooth_dict=None, **kwargs):
        transform = []
        if hflip:
            transform.append(transforms.RandomHorizontalFlip())
        transform_patch(transform, out_type=out_type, smooth_dict=smooth_dict)
        transform = transforms.Compose(transform)
        super().__init__(root=root, train=train, transform=transform)

        self.out_type = out_type
        self.size = len(self.data)

    @property
    def mean(self):
        return self.mean_dict.get(self.out_type, None)


@register_dataset
class CelebA(tvds.VisionDataset):
    """
    Large-scale CelebFaces Attributes (CelebA) Dataset [1]
    source: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    [^1]: Liu, Ziwei, et al. ‘Deep Learning Face Attributes in the Wild’.
     Proceedings of International Conference on Computer Vision (ICCV), 2015.
    """
    name = "celeba"
    base_folder = "celeba"
    shape = (3, 64, 64)
    mean_dict = {
        "0-1": 0.4424,
        "norm": 0.4682,
        "smooth_0-1": 0.4424,
    }
    peak = 1.
    data_type = "image"

    def __init__(
            self,
            root=ROOT,
            split="all",
            download=False,
            hflip=False,
            out_type="numpy",
            smooth_dict=None,
            **kwargs
    ):
        transform = [crop_celeba, transforms.Resize((64, 64))]
        if hflip:
            transform.append(transforms.RandomHorizontalFlip())
        transform_patch(transform, out_type=out_type, smooth_dict=smooth_dict)
        transform = transforms.Compose(transform)
        super().__init__(root, transform=transform)
        self.split = split
        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[split.lower()]
        splits = self._load_csv("list_eval_partition.txt")
        mask = slice(None) if split_ is None else (splits.data == split_).squeeze()
        if mask == slice(None):  # if split == "all"
            self.filename = splits.index
        else:
            self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]
        self.download = download

        self.out_type = out_type

    def _load_csv(
            self,
            filename,
            header=None,
    ):
        with open(os.path.join(self.root, self.base_folder, filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1:]
        else:
            headers = []

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return CSV(headers, indices, torch.as_tensor(data_int))

    @property
    def mean(self):
        return self.mean_dict.get(self.out_type, None)

    def __getitem__(self, index):
        X = PIL.Image.open(os.path.join(
            self.root, self.base_folder, "img_align_celeba", self.filename[index]))

        if self.transform is not None:
            X = self.transform(X)

        return X, 0

    def __len__(self):
        return len(self.filename)

    def extra_repr(self):
        lines = ["Split: {split}", ]
        return "\n".join(lines).format(**self.__dict__)


@register_dataset
class DSprites(tvds.VisionDataset):
    """
    dSprites - Disentanglement testing Sprites dataset [2]
    source: https://github.com/deepmind/dsprites-dataset
    [^2]: Matthey, Loic, et al. DSprites: Disentanglement Testing Sprites Dataset. 2017,
     https://github.com/deepmind/dsprites-dataset/.
    """
    name = "dsprites"
    shape = (1, 64, 64)
    mean_dict = {
        "0-1": 0.042494423521889584,
        "norm": 1
    }
    peak = 1.
    data_type = "image"
    binary = True
    url = "https://github.com/deepmind/dsprites-dataset/raw/fa310c66517cfc1939d77fe17c725154efc97127/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    npz_md5 = "7da33b31b13a06f4b04a70402ce90c2e"

    def __init__(self, root=ROOT, download=False, **kwargs):
        self.data_dir = os.path.join(root, "dsprites-dataset")
        npz_file = os.path.basename(self.url)
        self.data_dir = os.path.join(root, self.name)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.fpath = os.path.join(self.data_dir, npz_file)
        assert os.path.exists(self.fpath) or download, "Dataset NPZ file does not exists! Please set download=True!"
        if download:
            self.download()
        super().__init__(root=root, transform=None)
        self.data = self.load_data()

    def download(self):
        if not os.path.exists(self.fpath):
            wget.download(self.url, self.data_dir)
        with open(self.fpath, "rb") as f:
            assert hashlib.md5(f.read()).hexdigest() == self.npz_md5,\
                "MD5 Validation failed! Data file might be corrupted!"

    def load_data(self):
        return np.load(self.fpath)["imgs"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx, np.newaxis].astype("float32")), 0


class BOWDataset:
    data: sp.csr_matrix = None
    vocab: dict = None

    @staticmethod
    def _sort_vocab(vocab):
        words, ids = [], []
        for w, i in vocab.items():
            words.append(w)
            ids.append(i)
        return np.array(words)[np.argsort(ids)]

    def id2word(self, x):
        return self.vocab[x]

    def topk(self, x, k=10, to_string=False):
        out = self.id2word(np.argsort(x, axis=1)[:, -k:])[-1::-1]
        if to_string:
            out = "\n".join([", ".join(ln) for ln in out])
        return out

    @property
    def shape(self):
        return len(self.vocab),

    @property
    def size(self):
        return self.data.get_shape()[0]

    @property
    def mean(self):
        return self.data.mean()

    @property
    def average_sparsity(self):
        return np.mean(self.data.getnnz(axis=1)) / self.data.get_shape()[1]

    def compare_sparsity(self, other):
        x = np.array(self.data.getnnz(axis=1)).squeeze() / self.data.get_shape()[1]
        y = np.count_nonzero(other, axis=1) / self.data.get_shape()[1]
        out_string = "Average sparsity of training documents: {avg_spst_train}\n"
        out_string += "Average sparsity of generated documents: {avg_spst_gen}\n"
        out_string += "Empirical EMD: {emd_spst}\n"
        statistics = {
            "avg_spst_train": np.mean(x),
            "avg_spst_gen": np.mean(y),
            "emd_spst": data_emd(x, y)
        }
        return out_string.format(**statistics), statistics

    @property
    def average_length(self):
        return np.mean(self.data.sum(axis=1))

    def compare_length(self, other):
        x = np.array(self.data.sum(axis=1)).squeeze()
        y = np.sum(other, axis=1)
        out_string = "Average length of training documents: {avg_len_train}\n"
        out_string += "Average length of generated documents: {avg_len_gen}\n"
        out_string += "Empirical EMD: {emd_len}\n"
        statistics = {
            "avg_len_train": np.mean(x),
            "avg_len_gen": np.mean(y),
            "emd_len": data_emd(x, y),
        }
        return out_string.format(**statistics), statistics


def count_to_tfidf(arr: sp.csr_matrix):
    _arr = arr.copy().astype("float64", copy=False)
    df = np.bincount(arr.nonzero()[1], minlength=arr.shape[1])
    idf = np.log(arr.shape[0] / df)
    _arr.data /= np.maximum(arr.sum(axis=1).A1, 1)[arr.nonzero()[0]]
    _arr.data *= idf[arr.nonzero()[1]]
    return _arr


@register_dataset
class TwentyNews(BOWDataset):
    name = "20news"

    def __init__(self, root, subset, tf_idf=False, random_state=1234):
        data_home = os.path.join(root, "20news-bydate_py3")
        vocab_path = os.path.join(data_home, f"20news-bydate_vocab.pkl")
        if os.path.exists(vocab_path):
            with open(vocab_path, "rb") as f:
                vocab = pickle.load(f)
        else:
            _raw_contents = skds.fetch_20newsgroups(
                data_home=data_home, subset="all",
                shuffle=True, random_state=random_state,
                remove=("headers", "footers", "quotes"), return_X_y=False
            )["data"]
            _cv = CountVectorizer(
                input="content", strip_accents="ascii", lowercase=True, stop_words="english",
                analyzer="word", max_df=0.8, min_df=1e-3, max_features=65536)
            _data = _cv.fit_transform(_raw_contents)
            vocab = _cv.vocabulary_
            with open(vocab_path, "wb") as f:
                pickle.dump(vocab, f)
            _subset_path = os.path.join(data_home, f"20news-bydate_all.npz")
            sp.save_npz(_subset_path, _data, compressed=True)
        subset_path = os.path.join(data_home, f"20news-bydate_{subset}.npz")
        if os.path.exists(subset_path):
            data = sp.load_npz(subset_path)
        else:
            raw_contents = skds.fetch_20newsgroups(
                data_home=data_home, subset=subset,
                shuffle=True, random_state=random_state, return_X_y=False
            )["data"]
            cv = CountVectorizer(
                input="content", strip_accents="ascii", lowercase=True, vocabulary=vocab)
            data = cv.fit_transform(raw_contents)
            sp.save_npz(subset_path, data, compressed=True)
        if tf_idf:
            assert subset == "all"
            data = count_to_tfidf(data)
        self.tf_idf = tf_idf
        self.data = data
        self.vocab = self._sort_vocab(vocab)


@register_dataset
class NIPSPapers(BOWDataset):
    name = "nips"
    ZIPFILE = "nips-papers.zip"
    MD5 = "f6808b9eb03f8634adb4c484bb88d447"

    def __init__(self, root, tf_idf=False, **kwargs):
        self.root = root
        data_home = os.path.join(root, "nips-papers_py3")
        if not os.path.exists(data_home):
            os.makedirs(data_home)
        vocab_path = os.path.join(data_home, f"nips-papers_vocab.pkl")
        if os.path.exists(vocab_path):
            with open(vocab_path, "rb") as f:
                vocab = pickle.load(f)
            _data = None
        else:
            _raw_contents = self.load_data()
            _cv = CountVectorizer(
                input="content", strip_accents="ascii", lowercase=True, stop_words="english",
                analyzer="word", max_df=0.8, min_df=50, max_features=65536)
            _data = _cv.fit_transform(_raw_contents)
            vocab = _cv.vocabulary_
            with open(vocab_path, "wb") as f:
                pickle.dump(vocab, f)
        data_path = os.path.join(data_home, f"nips-papers_data.npz")
        if os.path.exists(data_path):
            data = sp.load_npz(data_path)
        else:
            data = _data
            sp.save_npz(data_path, _data, compressed=True)
        if tf_idf:
            data = count_to_tfidf(data)
        self.tf_idf = tf_idf
        self.data = data
        self.vocab = self._sort_vocab(vocab)

    def load_data(self):
        fpath = os.path.join(self.root, self.ZIPFILE)
        if not os.path.exists(fpath):
            raise FileNotFoundError("Dataset not found!")
        with open(fpath, "rb") as f:
            if not self.checkmd5(f.read()):
                logging.warning("MD5 Validation failed! Dataset might be corrupted!")
        with ZipFile(fpath, "r") as zf:
            txt_data = pd.read_csv(
                zf.open("papers.csv"),
                sep=",", header=0, index_col=0
            ).paper_text.tolist()
        return txt_data

    def checkmd5(self, data):
        return hashlib.md5(data).hexdigest() == self.MD5


def ml_preprocess(line):
    ln = line.decode("utf8").strip().split("::")
    return list(map(int, ln))


class MovieLensBase:
    name = None
    url = None
    md5 = None
    zip_folder = None
    ext = None
    dir = None
    delimiter = None
    header = False
    data_type = "other"

    def __init__(self, root, download=False, binary=False, **kwargs):
        self.root = root
        filename = os.path.basename(self.url)
        self.fpath = os.path.join(root, filename)
        assert os.path.exists(self.fpath) or download, "Dataset ZIP file does not exists! Please set download=True!"
        if download:
            self.download()
        self.data_dir = os.path.join(root, self.dir)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.data_path = os.path.join(self.data_dir, f"{filename.rsplit('.', maxsplit=1)[0]}_data.npz")
        self.data = self.load_data()
        self.binary = binary
        if self.binary:
            self.data.data.fill(1)

    def download(self):
        if not os.path.exists(self.fpath):
            wget.download(self.url, self.root)
        with open(self.fpath, "rb") as f:
            assert hashlib.md5(f.read()).hexdigest() == self.md5, "MD5 Validation failed! Dataset might be corrupted!"

    def load_data(self):
        if os.path.exists(self.data_path):
            data = sp.load_npz(self.data_path)
        else:
            with open(self.fpath, "rb") as f:
                with ZipFile(f, "r") as zf:
                    ratings = pd.Series(zf.open(f"{self.zip_folder}/ratings.{self.ext}", "r").readlines())
                    if self.header:
                        ratings = ratings[1:]
                    data = ratings.apply(lambda x: x.decode("utf8").strip()).str.split(self.delimiter, expand=True)
                    data.columns = "UserID::MovieID::Rating::Timestamp".split("::")
                    data = data.transform({"UserID": int, "MovieID": int, "Rating": float})
                    data = sp.csr_matrix((data["Rating"], (data["UserID"] - 1, data["MovieID"] - 1)))
            sp.save_npz(self.data_path, data, compressed=True)
        return data

    @property
    def shape(self):
        return self.data.get_shape()[1],

    @property
    def size(self):
        return self.data.get_shape()[0]

    @property
    def mean(self):
        return self.data.mean()

    @property
    def average_sparsity(self):
        return np.mean(self.data.getnnz(axis=1)) / self.data.get_shape()[1]

    def compare_sparsity(self, other):
        x = np.array(self.data.getnnz(axis=1)).squeeze() / self.data.get_shape()[1]
        y = np.count_nonzero(other, axis=1) / self.data.get_shape()[1]
        out_string = "Average sparsity of training documents: {avg_spst_train}\n"
        out_string += "Average sparsity of generated documents: {avg_spst_gen}\n"
        out_string += "Empirical EMD: {emd_spst}\n"
        statistics = {
            "avg_spst_train": np.mean(x),
            "avg_spst_gen": np.mean(y),
            "emd_spst": data_emd(x, y)
        }
        return out_string.format(**statistics), statistics

    @property
    def average_rating(self):
        return np.nanmean(np.array([
            np.mean(self.data.data[self.data.indptr[i]:self.data.indptr[i+1]])
            for i in range(len(self.data.indptr) - 1)]))

    def compare_rating(self, other):
        x = np.array([
            np.mean(self.data.data[self.data.indptr[i]:self.data.indptr[i+1]])
            for i in range(len(self.data.indptr) - 1)])
        np.nan_to_num(x, copy=False, nan=0.)
        x = x[x != 0.]
        nonzero = other != 0
        y = np.sum(other * nonzero, axis=1) / np.maximum(np.sum(nonzero, axis=1), 1)
        y = y[y != 0.]
        out_string = "Average ratings of training users: {avg_len_train}\n"
        out_string += "Average ratings of generated users: {avg_len_gen}\n"
        out_string += "Empirical EMD: {emd_rating}\n"
        statistics = {
            "avg_rating_train": np.mean(x),
            "avg_rating_gen": np.mean(y),
            "emd_rating": data_emd(x, y),
        }  # only users with at least one rating will be counted
        return out_string.format(**statistics), statistics

    def average_degree(self):
        return np.mean(self.data.sum(axis=0))

    def compare_degree(self, other):
        x = np.array(self.data.sum(axis=0)).squeeze()
        y = np.sum(other, axis=0)
        out_string = "Average movie ratings (in counts) of training users: {avg_deg_train}\n"
        out_string += "Average movie ratings (in counts) of generated users: {avg_deg_gen}\n"
        out_string += "Empirical EMD: {emd_deg}\n"
        statistics = {
            "avg_deg_train": np.mean(x),
            "avg_deg_gen": np.mean(y),
            "emd_deg": data_emd(x, y),
        }
        return out_string.format(**statistics), statistics


@register_dataset
class MovieLens1M(MovieLensBase):
    name = "ml_1m"
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    md5 = "c4d9eecfca2ab87c1945afe126590906"
    zip_folder = "ml-1m"
    ext = "dat"
    dir = "ml-1m_py3"
    delimiter = "::"
    header = False


@register_dataset
class MovieLens10M(MovieLensBase):
    name = "ml_10m"
    url = "https://files.grouplens.org/datasets/movielens/ml-10m.zip"
    md5 = "ce571fd55effeba0271552578f2648bd"
    zip_folder = "ml-10M100K"
    ext = "dat"
    dir = "ml-10m_py3"
    delimiter = "::"
    header = False


@register_dataset
class MovieLens20M(MovieLensBase):
    name = "ml_20m"
    url = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
    md5 = "cd245b17a1ae2cc31bb14903e1204af3"
    zip_folder = "ml-20m"
    ext = "csv"
    dir = "ml-20m_py3"
    delimiter = ","
    header = True


class DataLoader:
    sampler = None

    def __init__(self, dataset, batch_size, drop_last=True, shuffle=True, resample=False, random_state=1234):
        self.dataset = dataset
        self.data = dataset.data.astype(np.float32)
        self.total_size = self.dataset.size
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_batches = (
            math.floor if drop_last else math.ceil
        )(self.total_size / self.batch_size)
        self.shuffle = shuffle
        self.resample = resample
        self.random_state = random_state
        self.transform = safe_get(dataset, "transform")

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        inds = np.arange(self.total_size)
        if self.shuffle:
            self.random_state = (self.random_state + 7) % 2 ** 32  # 32-bit unsigned integer
            np.random.seed(self.random_state)
            np.random.shuffle(inds)
        for i in range(self.num_batches):
            end = (i + 1) * self.batch_size if (self.drop_last or i < self.num_batches - 1) else self.total_size
            x = self.data[inds[i * self.batch_size: end]]
            if isinstance(x, sp.csr_matrix):
                x = x.toarray(order="C")  # row-major
            if self.transform is not None:
                x = self.transform(x)
            yield torch.as_tensor(x).reshape((x.shape[0], -1))  # at least 2d

        if self.resample:
            self.dataset.resample()


def get_dataloader(
        dataset, batch_size,
        root=None, drop_last=True, shuffle=True, resample=False, random_state=1234,
        num_workers=0, pin_memory=False, distributed=False, dataset_configs=None, **sampler_kwargs
):
    data_configs = dataset_configs or DATASET_CONFIGS.get(dataset, None)
    if data_configs is None:
        data_configs = dict()
    if not issubclass(DATASET_DICT[dataset], BaseDataset):
        data_configs["root"] = root or data_configs.get("root", ROOT)
    is_image = isimage(dataset)
    if not is_image:
        data_configs["random_state"] = random_state
    dataset = DATASET_DICT[dataset](**data_configs)
    batch_size //= int(os.environ.get("WORLD_SIZE", "1"))
    dl_kwargs = dict(batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)
    if is_image:
        dl_kwargs.update({"num_workers": num_workers, "pin_memory": pin_memory})
        sampler = DistributedSampler(
            dataset=dataset, shuffle=True, drop_last=drop_last, **sampler_kwargs
        ) if distributed else None
        dl_kwargs["shuffle"] = sampler is None
        return DataLoader2D(dataset, sampler=sampler, **dl_kwargs)
    else:
        dl_kwargs.update({"resample": resample, "random_state": random_state})
        return DataLoader(dataset, **dl_kwargs)


if __name__ == "__main__":
    print("Registered integer-valued datasets:", INT_DATASETS)
    print("Registered real-valued datasets:", REAL_DATASETS)
    print("Registered image datasets:", IMAGE_DATASETS)
    print("Other datasets:", OTHER_DATASETS)
