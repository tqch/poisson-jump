import matplotlib.pyplot as plt
import matplotlib.scale as scale
import numpy as np
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid
from .utils.train import sym_sqrt_scale_functions, EMA, EncodeDict, TrainDict, SaveDict, DataDict


def train_step(train_dict: TrainDict, x, t, global_steps=1):
    loss = train_dict.diffusion.train_loss(
        train_dict.model, x.to(train_dict.device), t.to(train_dict.device)).mean()
    loss.div(train_dict.num_accum).backward()
    if global_steps % train_dict.num_accum == 0:
        if train_dict.grad_norm:
            nn.utils.clip_grad_norm_(
                train_dict.model.parameters(),
                max_norm=train_dict.grad_norm)
        train_dict.optimizer.step()
        if train_dict.scheduler is not None:
            train_dict.scheduler.step()
        if isinstance(train_dict.ema, EMA):
            train_dict.ema.update()
        train_dict.optimizer.zero_grad(set_to_none=True)
    loss = loss.detach()
    if dist.is_initialized():
        dist.reduce(loss, dst=0, op=dist.ReduceOp.SUM)
        loss.div_(dist.get_world_size())
    return loss.item()


def eval_step(
        train_dict: TrainDict,
        save_dict: SaveDict,
        data_dict: DataDict,
        e, rank, world_size, sample_seed):
    to_cpu = lambda x: x.cpu() if world_size <= 1 else x
    with train_dict.ema:
        if save_dict.eval_batch_size == 0:
            x_gen = to_cpu(train_dict.diffusion.p_sample(
                train_dict.model, z_T=save_dict.z_T.chunk(world_size, dim=0)[rank].to(train_dict.device),
                seed=sample_seed, return_pred=save_dict.use_pred)[int(save_dict.use_pred)].clone())
        else:
            x_gen = []
            for z in save_dict.z_T.chunk(world_size, dim=0)[rank].split(split_size=save_dict.eval_batch_size, dim=0):
                x_gen.append(to_cpu(train_dict.diffusion.p_sample(
                    train_dict.model, z_T=z.to(train_dict.device), seed=sample_seed, return_pred=save_dict.use_pred
                )[int(save_dict.use_pred)].clone()))
            x_gen = torch.cat(x_gen, dim=0)
    if world_size > 1:
        x_list = [torch.zeros_like(x_gen) for _ in range(world_size)]
        dist.all_gather(x_list, x_gen)
        x_gen = torch.cat(x_list, dim=0).cpu()
    img_path = os.path.join(save_dict.image_dir, f"{e + 1}.jpg")
    txt_path = os.path.join(save_dict.text_dir, f"{e + 1}.txt")
    if train_dict.is_leader:
        if data_dict.is_image:
            if hasattr(train_dict.trainloader.dataset, "out_type"):
                if train_dict.trainloader.dataset.out_type == "raw":
                    x_gen.div_(255.)
                elif train_dict.trainloader.dataset.out_type == "norm":
                    x_gen.add_(1.).div_(2.)
            x_gen.clamp_(min=0., max=1.)
            img_tensor = make_grid(x_gen, nrow=8).permute(1, 2, 0).numpy()
            if train_dict.is_leader:
                plt.imsave(img_path, img_tensor)
                if img_tensor.shape[-1] == 1:
                    img_tensor = img_tensor.squeeze(-1)
                    dataformats = "HW"
                else:
                    dataformats = "HWC"
                train_dict.writer.add_image("image", img_tensor, e + 1, dataformats=dataformats)
        else:
            x_gen = x_gen.squeeze().numpy()  # remove all the singular dimensions
            if data_dict.is_bow:
                x_gen = np.clip(x_gen, a_min=0., a_max=None)
                if not train_dict.trainloader.dataset.tf_idf:
                    x_gen = x_gen.round().astype("int")
                with open(txt_path, "w") as f:
                    out_string_spst, stats_spst = train_dict.trainloader.dataset.compare_sparsity(x_gen)
                    out_string_len, stats_len = train_dict.trainloader.dataset.compare_length(x_gen)
                    for stats in (stats_spst, stats_len):
                        for k, v in stats.items():
                            train_dict.writer.add_scalar(k, v, e + 1)
                    f.write(out_string_spst)
                    f.write(out_string_len)
                    f.write(f"Top {save_dict.topk} words:\n")
                    f.write(train_dict.trainloader.dataset.topk(
                        x_gen[:save_dict.ndocs], k=save_dict.topk, to_string=True))
            elif data_dict.is_ml:
                assert train_dict.trainloader.dataset.binary, "Non-binary case is not supported!"
                x_gen = np.clip(x_gen.round(), a_min=0, a_max=1)
                with open(txt_path, "w") as f:
                    out_string_spst, stats_spst = train_dict.trainloader.dataset.compare_sparsity(x_gen)
                    out_string_deg, stats_deg = train_dict.trainloader.dataset.compare_degree(x_gen)
                    for stats in (stats_spst, stats_deg):
                        for k, v in stats.items():
                            train_dict.writer.add_scalar(k, v, e + 1)
                    f.write(out_string_spst)
                    f.write(out_string_deg)
            else:
                if train_dict.trainloader.dataset.data_type == "int":
                    x_gen = x_gen.round()
                if train_dict.trainloader.dataset.name == "beta":
                    x_gen = np.clip(x_gen, a_min=0., a_max=1.)
                plt.figure(figsize=(6, 6), dpi=144)
                ax = plt.gca()
                if hasattr(train_dict.trainloader.dataset, "plot"):
                    scale_functions = train_dict.trainloader.dataset.plot(ax)
                    _range = plt.xlim()
                else:
                    scale_functions, _range = None, None
                if data_dict.is_real:
                    bins = max(save_dict.num_samples // 1000, 100)
                elif data_dict.dataset == "cat":
                    bins = np.linspace(-0.5, data_dict.input_shape[1] + 0.5, data_dict.input_shape[1] + 1)
                else:
                    bins = (save_dict.num_samples // 1000) if _range is None else \
                        np.arange(_range[0], _range[1] + 1, 1.0)
                ax.hist(x_gen, bins=bins, range=_range, density=True, label="Generated histogram")
                if save_dict.xsqrt:
                    plt.xscale(scale.FuncScale(ax, sym_sqrt_scale_functions()))
                if scale_functions is not None:
                    plt.yscale(scale.FuncScale(ax, scale_functions))
                plt.legend()
                plt.savefig(img_path)
                plt.close()
        train_dict.writer.flush()


def train_loop(
        encode_dict: EncodeDict = None,
        train_dict: TrainDict = None,
        save_dict: SaveDict = None,
        data_dict: DataDict = None):

    rank = 0
    world_size = 1
    sample_seed = train_dict.seed + 131071
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        sample_seed += rank

    global_steps = 0
    dry_run = bool(int(os.environ.get("DRY_RUN", "0")))
    if dry_run:
        train_dict.epochs = 1
        save_dict.chkpt_intv = 1
        save_dict.eval_intv = 1

    for e in range(train_dict.start_epoch, train_dict.epochs):

        if isinstance(train_dict.trainloader.sampler, DistributedSampler):
            train_dict.trainloader.sampler.set_epoch(e)

        with tqdm(
                train_dict.trainloader,
                desc=f"Epoch {e + 1}/{train_dict.epochs}", disable=not (train_dict.is_leader and train_dict.verbose)
        ) as pb:
            total_loss = 0
            total_count = 0
            train_dict.model.train()

            for i, x in enumerate(pb):
                global_steps += 1
                if isinstance(x, (list, tuple)):  # (data, label)
                    x = x[0]
                if encode_dict.continuous:
                    t = torch.rand(size=(x.shape[0],), dtype=torch.float64, device=train_dict.device)
                else:
                    t = torch.randint(encode_dict.timesteps, size=(x.shape[0], ), device=train_dict.device)
                loss = train_step(train_dict, x, t, global_steps=global_steps)
                total_loss += loss * x.shape[0]
                total_count += x.shape[0]
                pb.set_postfix({"loss": total_loss / total_count})
                if train_dict.is_leader:
                    train_dict.writer.add_scalar("train_loss", total_loss / total_count, e + 1)

                if dry_run:
                    break

        if (e + 1) % save_dict.eval_intv == 0:
            # logging & checkpointing are restricted to the primary process
            train_dict.model.eval()
            eval_step(train_dict, save_dict, data_dict, e, rank, world_size, sample_seed)

        if (e + 1) % save_dict.chkpt_intv == 0 and train_dict.is_leader:
            chkpt = dict()
            for k in ("model", "optimizer", "scheduler", "ema", "loss", "epoch"):
                if k == "epoch":
                    chkpt["epoch"] = e + 1
                if train_dict.get(k, None) is not None:
                    if k == "loss":
                        chkpt[k] = total_loss / total_count
                    elif isinstance(train_dict[k], (float, int)):
                        chkpt[k] = train_dict[k]
                    elif hasattr(train_dict[k], "state_dict"):
                        chkpt[k] = train_dict[k].state_dict()
                    else:
                        continue
            torch.save(chkpt, os.path.join(save_dict.chkpt_dir, f"{save_dict.exp_name}_{e + 1}.pt"))
            del chkpt

        if dist.is_available() and dist.is_initialized():
            dist.barrier()  # synchronize after evaluation

    train_dict.writer.close()
