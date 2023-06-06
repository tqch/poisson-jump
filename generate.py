import os
import re
import json
import math
import uuid
import time
import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from poisson_jump import *
from poisson_jump.datasets import DATASET_DICT
from tqdm import tqdm
import torch.multiprocessing as mp
from multiprocessing.sharedctypes import Synchronized


def progress_monitor(total, counter):
    pbar = tqdm(total=total)
    while pbar.n < total:
        if pbar.n < counter.value:  # non-blocking intended
            pbar.update(counter.value - pbar.n)
        time.sleep(0.1)


# noinspection DuplicatedCode
def generate(rank, args, counter=0):
    assert isinstance(counter, (Synchronized, int))

    is_leader = rank == 0
    config_file = args.config_path
    config_name = re.sub(r".json$", "", os.path.basename(args.config_path))
    try:
        with open(config_file, "r") as f:
            meta_configs = json.load(f)
    except FileNotFoundError:
        meta_configs = dict()
    diffusion_configs = meta_configs.get("diffusion", dict())
    if args.no_clip:
        diffusion_configs["clip_range"] = None
    model_configs = meta_configs.get("model", dict())
    dataset = meta_configs.get("dataset")
    dataset_configs = meta_configs.get("dataset_configs", dict())
    out_type = dataset_configs.get("out_type", "0-1")
    input_shape = DATASET_DICT[dataset].shape
    assert isimage(dataset), "Non-image data are not currently supported by generate.py!"
    exp_name = meta_configs.get("exp_name", config_name)

    diffusion_type = diffusion_configs.get("type", "ordinal_jump")
    decay_schedule = diffusion_configs.get("decay_schedule", "beta_linear")
    cont = diffusion_configs.get("continuous", True)
    timesteps = diffusion_configs.get("timesteps", 1000)
    pred_type = diffusion_configs.get("pred_type", "x_0")
    var_type = diffusion_configs.get("var_type", "fixed_small")
    loss_type = diffusion_configs.get("loss_type", "kl_simple")
    clip_range = diffusion_configs.get("clip_range", None)
    input_clip = diffusion_configs.get("input_clip", None)
    normalize = diffusion_configs.get("normalize", None)
    z_rescale = diffusion_configs.get("z_rescale", False)
    p_self_cond = diffusion_configs.get("p_self_cond", 0.)
    psnr = diffusion_configs.get("psnr", False)
    num_bits = diffusion_configs.get("num_bits", 8)
    try:
        if re.match(r"^bits(_[a-zA-Z0-9]+)?_jump$", diffusion_type) is not None:
            signal_stat = 1. if psnr else 0.5
        else:
            if psnr:
                signal_stat = DATASET_DICT[dataset].peak
            else:
                signal_stat = DATASET_DICT[dataset].mean_dict.get(out_type, 1.)
    except AttributeError:
        signal_stat = 1.
    diffusion_configs["lbd"] = diffusion_configs.get("lbd", "auto")
    diffusion_configs["signal_stat"] = diffusion_configs.get("signal_stat", signal_stat)
    schedule_kwargs = {k: diffusion_configs[k] for k in ["lbd", "signal_stat", ] +
                       [k for k in diffusion_configs if k.endswith(("start", "end"))]}
    schedule_dict, schedule_kwargs = get_decay_schedule(
        decay_schedule, timesteps=timesteps, return_function=cont, diffusion_type=diffusion_type, **schedule_kwargs)
    diffusion_configs.update(schedule_kwargs)
    lbd = diffusion_configs["lbd"]

    diffusion_kwargs = {
        "pred_type": pred_type,
        "var_type": var_type,
        "loss_type": loss_type,
        "lbd": lbd,
        "timesteps": timesteps,
        "clip_range": clip_range,
        "input_clip": input_clip,
        "normalize": normalize,
        "z_rescale": z_rescale,
        "p_self_cond": p_self_cond
    }
    diffusion_kwargs.update(schedule_dict)
    if diffusion_type.startswith("bits"):
        input_shape = (input_shape[0] * (num_bits or 8),) + input_shape[1:]
        diffusion_kwargs["num_bits"] = num_bits
    diffusion = DIFFUSION_DICT[diffusion_type](**diffusion_kwargs)

    device = torch.device(f"cuda:{rank}" if args.num_gpus > 1 else args.device)
    transform = model_configs.pop("transform", None)
    out_activation = model_configs.pop("out_activation", "none")
    if p_self_cond > 0:
        model_configs["in_channels"] *= 2
    model_configs["continuous_t"] = cont
    model_configs["resample_with_conv"] = model_configs.get("resample_with_conv", False)
    model_configs["resample_with_res"] = model_configs.get("resample_with_res", False)
    pre_transform, post_transform = get_transform(transform)
    if diffusion_type.endswith("jump"):
        if pred_type.startswith("eps"):
            post_transform = None
        elif transform != "normalize":
            post_transform = FuncChainer([get_activation(out_activation), post_transform])
    _model = UNet(**model_configs)
    model = ModelWrapper(_model, pre_transform, post_transform).to(device)

    chkpt_path = args.chkpt_path
    assert os.path.exists(chkpt_path)
    chkpt = torch.load(chkpt_path, map_location=device)
    chkpt = chkpt["ema"]["shadow"] if "ema" in chkpt else chkpt["model"]
    for k in tuple(chkpt.keys()):
        if k.startswith("module."):
            chkpt[k.split(".", maxsplit=1)[1]] = chkpt.pop(k)
    model.load_state_dict(chkpt)
    del chkpt

    model.requires_grad_(False)
    model.eval()

    def _save_image(arr, path):
        with Image.fromarray(arr, mode="RGB") as im:
            im.save(f"{path}/{uuid.uuid4()}.png")

    folder_name = os.path.basename(chkpt_path)[:-3]  # truncated at file extension
    save_dir = os.path.join(args.save_dir, exp_name, folder_name)
    if is_leader and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_image = partial(_save_image, path=save_dir)
    save_image_ = None
    if args.return_pred:
        save_dir_ = save_dir + "-pred_x_0"
        if is_leader and not os.path.exists(save_dir_):
            os.makedirs(save_dir_)
        save_image_ = partial(_save_image, path=save_dir_)

    local_total_size = args.local_total_size
    batch_size = args.batch_size
    if args.world_size > 1:
        if rank < args.total_size % args.world_size:
            local_total_size += 1
    local_num_batches = math.ceil(local_total_size / batch_size)
    shape = (batch_size,) + input_shape

    if is_leader:
        hps = {
            "diffusion": diffusion_configs,
            "model": model_configs,
        }
        print("Hyperparameter settings:\n" + dict2str(hps))
        print(f"Generating {args.total_size} image(s) on {args.world_size} GPU(s).")
        print(f"Batch size: {batch_size}", flush=True)

    if torch.backends.cudnn.is_available():  # noqa
        torch.backends.cudnn.benchmark = True  # noqa

    pbar = None
    if isinstance(counter, int):
        pbar = tqdm(total=local_num_batches)

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        for i in range(local_num_batches):
            if i == local_num_batches - 1:
                shape = (local_total_size - i * batch_size,) + input_shape
            z_T = {
                "jump": torch.zeros,
                "gaussian": torch.randn
            }[diffusion_type.split("_", maxsplit=1)[-1]](shape, device=device)
            x_0, pred_x_0 = diffusion.p_sample(model, z_T=z_T, return_pred=args.return_pred)
            x_0 = x_0.cpu()
            if dataset_configs.get("out_type", "0-1") == "norm":
                x_0.add_(1.).div_(2.)
            if not diffusion_type.startswith("bits"):
                x_0.mul_(255.)
            x_0 = x_0.round_().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
            pool.map(save_image, list(x_0))
            if args.return_pred:
                pred_x_0 = pred_x_0.cpu()
                if dataset_configs.get("out_type", "0-1") == "norm":
                    pred_x_0.add_(1.).div_(2.)
                if not diffusion_type.startswith("bits"):
                    pred_x_0.mul_(255.)
                pred_x_0 = pred_x_0.round_().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
                pool.map(save_image_, list(pred_x_0))
            if isinstance(counter, Synchronized):
                with counter.get_lock():
                    counter.value += 1
            else:
                pbar.update(1)


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--chkpt-path", required=True, type=str)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--total-size", default=50000, type=int)
    parser.add_argument("--return-pred", action="store_true")
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--num-gpus", default=1, type=int)
    parser.add_argument("--save-dir", default="./images/eval", type=str)
    parser.add_argument("--config-path", required=True, type=str)
    parser.add_argument("--max-workers", default=8, type=int)
    parser.add_argument("--no-clip", action="store_true")

    args = parser.parse_args()

    world_size = args.world_size = args.num_gpus or 1
    local_total_size = args.local_total_size = args.total_size // world_size
    batch_size = args.batch_size
    remainder = args.total_size % world_size
    num_batches = math.ceil((local_total_size + 1) / batch_size) * remainder
    num_batches += math.ceil(local_total_size / batch_size) * (world_size - remainder)
    args.num_batches = num_batches

    if world_size > 1:
        mp.set_start_method("spawn")
        counter = mp.Value("i", 0)
        mp.Process(target=progress_monitor, args=(num_batches, counter), daemon=True).start()
        mp.spawn(generate, args=(args, counter), nprocs=args.num_gpus)
    else:
        generate(0, args)


if __name__ == "__main__":
    main()
