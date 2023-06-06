import json
import math
import numpy as np
import os
import re
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from contextlib import nullcontext
from datetime import datetime
from poisson_jump import *
try:
    from tensorboardX import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = DummyWriter
from torch.optim import AdamW, lr_scheduler
from torch.distributed.elastic.multiprocessing import errors
from torch.nn.parallel import DistributedDataParallel as DDP


# noinspection DuplicatedCode
def train(rank: int, args, temp_dir=""):
    local_rank = rank = int(os.environ.get("RANK", str(rank)))
    is_leader = rank == 0

    mode = args.distributed_mode
    distributed = mode != "none"
    if distributed:
        assert dist.is_available()
        if mode == "mp_spawn":
            # multiprocessing spawn for single-node multi-gpu training
            # shared file-system initialization
            assert temp_dir, "Temporary directory cannot be empty!"
            init_file = os.path.join(temp_dir, ".torch_distributed_init")
            init_method = f"file://{init_file}"
            dist.init_process_group("nccl", init_method=init_method, rank=rank, world_size=args.num_gpus)
            local_rank = rank
            os.environ["WORLD_SIZE"] = str(args.num_gpus)
        elif mode in {"elastic", "slurm"}:
            # torch.distributed.elastic with C10d rendezvous backend by default uses TCP initialization
            world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
            rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
            dist.init_process_group("nccl", init_method="env://", world_size=world_size, rank=rank)
            local_world_size = int(os.environ.get("SLURM_GPUS_ON_NODE", "0")) or torch.cuda.device_count()
            local_rank = int(os.environ.get("LOCAL_RANK", "0")) or rank % local_world_size
            os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", str(world_size))
    else:
        try:
            local_rank = int(re.match(r"cuda:(\d)", args.device).group(1))
        except AttributeError:  # no match
            local_rank = 0

    if torch.cuda.is_available():  # check whether CUDA is available
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    config_file = args.config_path
    config_name = re.sub(r".json$", "", os.path.basename(args.config_path))
    try:
        with open(config_file, "r") as f:
            meta_configs = json.load(f)
    except FileNotFoundError:
        meta_configs = dict()
    diffusion_configs = meta_configs.get("diffusion", dict())
    model_configs = meta_configs.get("model", dict())
    trainer_configs = meta_configs.get("trainer", dict())
    dataset_configs = meta_configs.get("dataset_configs", dict())

    seed_all(args.seed)
    dataset = meta_configs.get("dataset", args.dataset)
    is_real = isreal(dataset)
    is_bow = issubclass(DATASET_DICT[dataset], BOWDataset)
    is_ml = issubclass(DATASET_DICT[dataset], MovieLensBase)
    is_image = isimage(dataset)
    exp_name = meta_configs.get("exp_name", config_name)

    def update_trainer(name1, name2=None):
        name2 = name2 or name1
        if name1 not in trainer_configs:
            trainer_configs[name1] = args.__getattribute__(name2)
        return trainer_configs[name1]

    batch_size = update_trainer("batch_size") // args.num_accum
    root = args.root
    if "~" in root:
        root = os.path.expanduser(root)
    elif "$" in root:
        root = os.path.expandvars(root)

    trainloader = get_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        root=root,
        drop_last=True,
        shuffle=True,
        resample=False,
        random_state=args.seed,
        num_workers=args.num_workers,
        pin_memory=True,
        distributed=distributed,
        dataset_configs=dataset_configs
    )
    input_shape = trainloader.dataset.shape

    def update_diffusion(name1, name2=None, default=None):  # lazy update: update only when the key is missing
        name2 = name2 or name1
        if name1 not in diffusion_configs:
            try:
                diffusion_configs[name1] = args.__getattribute__(name2)
            except AttributeError:
                return default
        return diffusion_configs[name1]

    diffusion_type = update_diffusion("type", "diffusion_type")
    decay_schedule = update_diffusion("decay_schedule")
    update_diffusion("logsnr_start")
    update_diffusion("logsnr_end")
    cont = update_diffusion("continuous", "cont")
    psnr = update_diffusion("psnr")
    if re.match(r"^bit(_[a-zA-Z0-9]+)?_jump$", diffusion_type) is not None:
        signal_stat = 1. if psnr else 0.5
    else:
        signal_stat = safe_get(trainloader.dataset, "peak" if psnr else "mean", None) or 1.
    diffusion_configs["signal_stat"] = signal_stat
    update_diffusion("lbd")
    timesteps = update_diffusion("timesteps")
    pred_type = update_diffusion("pred_type")
    var_type = update_diffusion("var_type")
    loss_type = update_diffusion("loss_type")
    clip_range = update_diffusion("clip_range")
    input_clip = update_diffusion("input_clip")
    normalize = update_diffusion("normalize")
    z_rescale = update_diffusion("z_rescale")
    p_self_cond = update_diffusion("p_self_cond")
    num_bits = update_diffusion("num_bits")

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
    if diffusion_type.endswith("jump"):
        if not args.no_check:
            # loss_type check
            assert diffusion_configs["loss_type"].startswith("kl"), "Non-KL loss type is not supported!"
        if diffusion_type.startswith("bit"):
            assert "num_bits" in diffusion_configs
            diffusion_kwargs["num_bits"] = num_bits

    diffusion = DIFFUSION_DICT[diffusion_type](**diffusion_kwargs)

    def update_model(name1, name2=None):
        name2 = name2 or name1
        if name1 not in model_configs:
            model_configs[name1] = args.__getattribute__(name2)
        return model_configs[name1]

    transform = update_model("transform")
    out_activation = update_model("out_activation")
    model_configs["continuous_t"] = cont
    if is_image:
        if p_self_cond > 0:
            model_configs["in_channels"] *= 2
        del model_configs["transform"]
        del model_configs["out_activation"]
        model_configs["resample_with_conv"] = model_configs.get("resample_with_conv", False)
        model_configs["resample_with_res"] = model_configs.get("resample_with_res", False)
        model_configs["scale_shift"] = model_configs.get("scale_shift", False)
        model = UNet(**model_configs)
    else:
        assert len(input_shape) == 1
        args.in_dim = input_shape[0]
        model_kwargs = ["in_dim", "base_dim", "multiplier", "num_layers", "drop_rate"]
        model_kwargs = {k: update_model(k) for k in model_kwargs}
        model = ConditionalMLP(**model_kwargs)

    pre_transform, post_transform = get_transform(transform)
    model_configs["transform"] = transform

    if diffusion_type.endswith("jump"):
        if pred_type.startswith("eps"):
            post_transform = None
        elif transform != "normalize":
            post_transform = FuncChainer([get_activation(out_activation), post_transform])
    model_configs["out_activation"] = out_activation

    model = ModelWrapper(model, pre_transform=pre_transform, post_transform=post_transform)
    model.to(device)

    if distributed and dist.is_initialized():  # Distributed Data Parallel
        model = DDP(model, device_ids=[local_rank, ])

    use_ema = update_trainer("use_ema")
    ema_decay = update_trainer("ema_decay")
    ema = nullcontext()
    if use_ema:  # Exponential Moving Averaging
        ema = EMA(model, decay=ema_decay)

    lr = update_trainer("lr")
    beta1 = update_trainer("beta1")
    beta2 = update_trainer("beta2")
    weight_decay = update_trainer("weight_decay")
    warmup = update_trainer("warmup")
    grad_norm = update_trainer("grad_norm")
    optimizer_kwargs = {"lr": lr, "betas": (beta1, beta2), "weight_decay": weight_decay}
    optimizer = AdamW(model.parameters(), **optimizer_kwargs)

    scheduler = lr_scheduler.LambdaLR(
        optimizer, lambda t: min((t + 1) / warmup, 1.0)) if warmup > 0 else None

    image_dir = os.path.join(args.image_dir, exp_name)
    text_dir = os.path.join(args.text_dir, exp_name)
    chkpt_dir = os.path.join(args.chkpt_dir, exp_name)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S%f")
    log_dir = os.path.join(args.log_dir, "_".join([exp_name, timestamp]))

    start_epoch = 0
    epochs = update_trainer("epochs")
    num_samples = update_trainer("eval_num_samples", "num_samples")
    eval_intv = update_trainer("eval_intv")
    eval_batch_size = update_trainer("eval_batch_size")
    use_pred = update_trainer("use_pred")
    chkpt_intv = update_trainer("chkpt_intv")
    topk = update_trainer("topk")
    ndocs = update_trainer("ndocs")
    xsqrt = update_trainer("xsqrt")

    chkpt_path = args.chkpt_path or os.path.join(chkpt_dir, f"{exp_name}.pt")
    if args.resume or distributed:
        if os.path.exists(chkpt_path):
            chkpt = torch.load(args.chkpt_path, map_location=device)
            start_epoch = resume_from_chkpt(chkpt, model=model, optimizer=optimizer, scheduler=scheduler, ema=ema)
            del chkpt

    writer = sys.stdout
    if is_leader:
        print(f"Dataset: {dataset}")
        if not is_bow and not os.path.exists(image_dir):
            os.makedirs(image_dir)
            print(f"Images are saved to {os.path.abspath(image_dir)}.")
        if is_bow and not os.path.exists(text_dir):
            os.makedirs(text_dir)
            print(f"Texts are saved to {os.path.abspath(text_dir)}.")
        for d in [chkpt_dir, log_dir]:
            if not os.path.exists(d):
                os.makedirs(d)
        print(f"Checkpoints are saved to {os.path.abspath(chkpt_dir)}.")
        print(f"Logs are saved to {os.path.abspath(log_dir)}.")

        writer = SummaryWriter(log_dir, write_to_disk=not args.no_log)
        hps = {
            "diffusion": diffusion_configs,
            "model": model_configs,
            "trainer": trainer_configs
        }
        hps_str = dict2str(hps)
        writer.add_text("hps", hps_str)
        print("Hyperparameter settings:\n" + hps_str)
        with open(os.path.join(log_dir, "exp_configs.json"), "w") as f:
            f.write(hps_str)

    num_samples = min(num_samples, 64) if is_image else num_samples
    _input_shape = (input_shape[0] * (num_bits or 8), ) + input_shape[1:]\
        if diffusion_type.startswith("bit") else input_shape
    mibs = math.ceil(np.prod((num_samples,) + _input_shape) / 2 ** 20)  # data size in mebibytes
    _device = torch.device("cpu") if mibs > args.mem_limit else device
    if diffusion_type.endswith("jump"):
        z_T = torch.zeros
    elif diffusion_type.endswith("gaussian"):
        z_T = torch.randn
    else:
        raise NotImplementedError(diffusion_type)
    z_T = z_T((num_samples,) + _input_shape, device=_device)

    encode_dict = EncodeDict().init(type=diffusion_type, lbd=lbd, timesteps=timesteps, continuous=cont)
    train_dict = TrainDict().init(
        trainloader=trainloader,
        diffusion=diffusion,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        num_accum=args.num_accum,
        grad_norm=grad_norm,
        writer=writer,
        start_epoch=start_epoch,
        epochs=epochs,
        is_leader=is_leader,
        device=device,
        verbose=args.verbose,
        seed=args.seed
    )
    save_dict = SaveDict().init(
        num_samples=num_samples,
        topk=topk,
        ndocs=ndocs,
        z_T=z_T,
        xsqrt=xsqrt,
        eval_intv=eval_intv,
        eval_batch_size=eval_batch_size,
        use_pred=use_pred,
        chkpt_intv=chkpt_intv,
        image_dir=image_dir,
        text_dir=text_dir,
        chkpt_dir=chkpt_dir,
        log_dir=log_dir,
        exp_name=exp_name
    )
    data_dict = DataDict().init(dataset=dataset, input_shape=input_shape, is_real=is_real,
                                is_bow=is_bow, is_ml=is_ml, is_image=is_image)
    train_loop(encode_dict=encode_dict, train_dict=train_dict, save_dict=save_dict, data_dict=data_dict)


# noinspection PyPep8
@errors.record
def main():
    from argparse import ArgumentParser

    def parse_seq(seq_str, sep=","):
        return tuple(float(s) if s.isdigit() else None for s in seq_str.split(sep))

    parser = ArgumentParser()
    parser.add_argument("--root", default="", type=str, help="datasets' root directory")
    parser.add_argument("--dataset", choices=DATASET_CONFIGS.keys(), default="poisson", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--diffusion-type", choices=DIFFUSION_DICT.keys(), default="ordinal_jump")
    parser.add_argument("--timesteps", default=100, type=int, help="number of encoding process")
    parser.add_argument("--cont", action="store_true", help="whether to use continuous-time training")
    parser.add_argument("--decay-schedule", default="beta_linear", help="schedule of signal decay")
    parser.add_argument("--logsnr-start", default=10., type=float)
    parser.add_argument("--logsnr-end", default=-10, type=float)
    parser.add_argument("--psnr", action="store_true")
    parser.add_argument("--beta-start", default=0.001, type=float)
    parser.add_argument("--beta-end", default=None, type=float)
    parser.add_argument("--lbd", default=None, type=float)
    parser.add_argument("--loss-type", choices=["mae", "mse", "huber", "kl", "kl_simple", "kl_alpha"], default="kl_simple", type=str)
    parser.add_argument("--clip-range", default=None, type=parse_seq)
    parser.add_argument("--input-clip", default=None, type=parse_seq)
    parser.add_argument("--normalize", default=None, type=parse_seq)
    parser.add_argument("--z-rescale", action="store_true")
    parser.add_argument("--p-self-cond", default=0., type=float)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--warmup", default=0, type=int)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.999, type=float)
    parser.add_argument("--weight-decay", default=0., type=float)
    parser.add_argument("--num-accum", default=1, type=int)
    parser.add_argument("--grad-norm", default=1.0, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--base-dim", default=128, type=int)
    parser.add_argument("--multiplier", nargs="+", default=1, type=int)
    parser.add_argument("--num-layers", default=3, type=int)
    parser.add_argument("--drop-rate", default=0., type=float)
    parser.add_argument("--config-path", required=True, type=str)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--num-samples", default=30000, type=int)
    parser.add_argument("--mem-limit", default=48, type=int)
    parser.add_argument("--use-pred", action="store_true")
    parser.add_argument("--image-dir", default="./images/train", type=str)
    parser.add_argument("--text-dir", default="./texts/train", type=str)
    parser.add_argument("--transform", choices=["none", "anscombe", "freeman-tukey", "normalize", "log"], default="none", type=str)
    parser.add_argument("--out-activation", choices=["none", "relu", "softplus"], default="none", type=str)
    parser.add_argument("--xsqrt", action="store_true")
    parser.add_argument("--topk", default=10, type=int)
    parser.add_argument("--ndocs", default=10, type=int)
    parser.add_argument("--eval-intv", default=10, type=int)
    parser.add_argument("--eval-batch-size", default=0, type=int)
    parser.add_argument("--chkpt-dir", default="./chkpts", type=str)
    parser.add_argument("--log-dir", default="./logs", type=str)
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--chkpt-intv", default=100, type=int)
    parser.add_argument("--chkpt-path", default="", type=str)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--ema-decay", default=0.9999, type=float)
    parser.add_argument("--distributed-mode", choices={"none", "mp_spawn", "elastic", "slurm"}, default="none", type=str)
    parser.add_argument("--num-gpus", default=1, type=int)
    parser.add_argument("--dry-run", action="store_true", help="run one mini-batch test")
    parser.add_argument("--verbose", action="store_true", help="whether to show progress bar")
    parser.add_argument("--no-check", action="store_true", help="no loss-type check")

    args = parser.parse_args()

    if args.dry_run:
        os.environ["DRY_RUN"] = "1"

    mode = args.distributed_mode
    if mode == "mp_spawn":
        mp.set_start_method("spawn")
        with tempfile.TemporaryDirectory() as temp_dir:
            mp.spawn(train, args=(args, temp_dir), nprocs=args.num_gpus)
    else:
        train(rank=0, args=args)


if __name__ == "__main__":
    main()
