import itertools
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import torch
import torch.multiprocessing as mp
from contextlib import nullcontext
from poisson_jump.datasets import *
from poisson_jump.diffusions import DIFFUSION_DICT
from poisson_jump.nets import ConditionalMLP
from poisson_jump.utils import *
from poisson_jump.utils.train import *
from poisson_jump.schedules import *
from poisson_jump.train_loop import train_loop
from scipy.stats import wasserstein_distance, gaussian_kde
from torch.optim import AdamW


def print_fn(*args, verbose=True, **print_kwargs):
    if verbose:
        print(*args, **print_kwargs)


def main(args):
    assert os.path.exists(args.config_path)
    with open(args.config_path, "r") as f:
        cfgs = json.load(f)
    args.master_seed = cfgs.get("seed", 1234)
    args.dataset = dataset = cfgs["dataset"]
    diff_cfgs = cfgs["diffusion"]
    args.is_real = isreal(dataset)
    args.is_bow = issubclass(DATASET_DICT[dataset], BOWDataset)
    args.is_ml = issubclass(DATASET_DICT[dataset], MovieLensBase)
    args.text_out = args.is_bow or args.is_ml
    args.is_image = isimage(dataset)
    args.is_beta = dataset == "beta"
    cfgs["trainer"]["batch_size"] = batch_size = cfgs["trainer"].get("batch_size", 1000)
    args.dataset_configs = cfgs.get("dataset_configs", None)
    args.dataset_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        root=args.root,
        drop_last=True,
        shuffle=True,
        resample=False,
        random_state=args.master_seed,
        num_workers=0,  # unused for non-image data
        pin_memory=True,
        distributed=False,
        dataset_configs=args.dataset_configs
    )
    args.diffusion_type = diff_cfgs.pop("type", "ordinal_jump")
    fixed = dict()
    tuning_list = []
    for k, v in diff_cfgs.items():
        if k not in ("clip_range", "input_clip", "normalize") and isinstance(v, (tuple, list)):
            tuning_list.append(k)
        else:
            fixed[k] = v
    tuning = tuple(
        dict(zip(tuning_list, prod))
        for prod in itertools.product(*[diff_cfgs[k] for k in tuning_list]))
    args.diff_fixed, args.diff_tuning = fixed, tuning
    args.world_size = len(tuning) or 1
    args.rank_offset = 0
    args.train_cfgs = cfgs["trainer"]
    args.model_cfgs = cfgs["model"]
    args.rel_path = os.path.relpath(os.path.dirname(args.config_path), "./configs")
    args.exp_name = cfgs.get("exp_name", os.path.basename(args.config_path)[:-5])
    args.exp_dir = os.path.join(args.exp_dir, args.rel_path, args.exp_name)
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
        print_fn(f"Experiment results are saved to {os.path.abspath(args.exp_dir)}.", verbose=args.verbose)

    if args.train:
        if args.world_size > 1:
            mp.set_start_method("spawn")
            if args.num_procs == 0:
                mp.spawn(subprocess_fn, args=(args,), nprocs=args.world_size)
            else:
                num_splits = math.ceil(args.world_size / args.num_procs)
                for i in range(num_splits):
                    nprocs = args.num_procs if (i != num_splits - 1) else (args.world_size % args.num_procs)
                    mp.spawn(subprocess_fn, args=(args,), nprocs=nprocs)
                    args.rank_offset += args.num_procs
        else:
            args.tuning = (dict(),)
            subprocess_fn(rank=0, args=args)
    else:
        results = []
        if args.text_out:
            fields = []
            pattern = r"Wasserstein distance (\(sparsity\)|\(length\)|\(degree\)):\s*(\d+\.\d+ \+/- \d+\.\d+)"
            for i in range(args.world_size):
                with open(os.path.join(args.exp_dir, f"hpset{i}.txt")) as f:
                    found = re.findall(pattern, f.read())
                    if i == 0:
                        fields.extend(list(map(lambda x: x[0], found)))
                    results.append(list(map(lambda x: x[1], found)))
            with open(os.path.join(args.exp_dir, "summary.csv"), "w") as f:
                f.write(",".join(["dataset", ] + tuning_list + [
                    "wasserstein_distance %s" % field for field in fields]) + "\n")
                for i, res in enumerate(results):
                    f.write(",".join([args.dataset, ] + [str(tuning[i][k]) for k in tuning_list] + res) + "\n")
        else:
            pattern = r"Wasserstein distance:\s*(\d+\.\d+ \+/- \d+\.\d+)"
            for i in range(args.world_size):
                with open(os.path.join(args.exp_dir, f"hpset{i}.txt")) as f:
                    results.append(re.search(pattern, f.read()).group(1))
            with open(os.path.join(args.exp_dir, "summary.csv"), "w") as f:
                f.write(",".join(["dataset", ] + tuning_list + ["wasserstein_distance"]) + "\n")
                for i, res in enumerate(results):
                    f.write(",".join([args.dataset, ] + [str(tuning[i][k]) for k in tuning_list] + [res, ]) + "\n")


def subprocess_fn(rank=0, args=None):
    rank += args.rank_offset

    trainloader = get_dataloader(**args.dataset_kwargs)
    input_shape = trainloader.dataset.shape
    try:
        args.is_int = trainloader.dataset.data_type == "int"
    except AttributeError:
        args.is_int = False
    try:
        args.is_bin = trainloader.dataset.binary
    except AttributeError:
        args.is_bin = False
    try:
        args.is_tfidf = trainloader.dataset.tf_idf
    except AttributeError:
        args.is_tfidf = False
    p_vals = trainloader.dataset.data
    data_size = trainloader.dataset.size
    signal_stat = trainloader.dataset.mean

    diff_kwargs = dict()
    diff_kwargs.update(args.diff_fixed)
    diff_kwargs.update(args.diff_tuning[rank])
    decay_schedule = diff_kwargs.pop("decay_schedule", "beta_linear")
    cont = diff_kwargs.get("continuous", False)
    if torch.cuda.is_available():
        device_id = args.device_id + rank % args.num_gpus
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device_id)
    else:
        device = "cpu"
    loss_type = diff_kwargs.get("loss_type", "kl_simple")
    pred_type = diff_kwargs.get("pred_type", "x_0")

    diff_kwargs.update({"loss_type": loss_type, "pred_type": pred_type, "continuous": cont})
    transform = args.model_cfgs.pop("transform", "none")
    out_activation = args.model_cfgs.pop("out_activation", "none")
    args.model_cfgs["continuous_t"] = cont
    if not args.no_check:
        if args.diffusion_type.endswith("jump"):
            assert loss_type.startswith("kl"), "Non-KL loss type is not supported!"
    pre_transform, post_transform = get_transform(transform)
    if args.diffusion_type.endswith("jump"):
        if pred_type.startswith("eps"):
            post_transform = None
        elif transform != "normalize":
            post_transform = FuncChainer([get_activation(out_activation), post_transform])
    schedule_dict, schedule_kwargs = get_decay_schedule(
        decay_schedule,
        return_function=cont, diffusion_type=args.diffusion_type, signal_stat=signal_stat, **diff_kwargs)
    diff_kwargs.update(schedule_kwargs)
    diffusion = DIFFUSION_DICT[args.diffusion_type](**diff_kwargs, **schedule_dict)

    pdfs = hists = None
    if args.is_bow:
        w_distances = {"spst": [], "len": []}
    elif args.is_ml:
        w_distances = {"spst": [], "deg": []}
    else:
        w_distances = []
        if args.is_real:
            pdfs = []
        if args.is_int:
            hists = []

    upper = xs = None

    if not args.text_out:
        fig = plt.figure()
        ax = plt.gca()
        trainloader.dataset.plot(ax)
        upper = plt.xlim()[1] - (0.05 if args.is_beta else 0.5)
        if args.is_real:
            xs = np.linspace(0, upper, 1000)
        else:
            upper = int(upper)
            xs = np.arange(upper + 1)
        plt.close(fig)

    train_cfgs = args.train_cfgs
    epochs = train_cfgs.get("epochs", args.epochs)
    eval_intv = train_cfgs.get("eval_intv", args.eval_intv)
    eval_batch_size = train_cfgs.get("eval_batch_size", args.eval_batch_size)
    xsqrt = not args.dataset.endswith("beta")
    lr = train_cfgs.get("lr", 600)
    betas = train_cfgs.get("betas", (0.9, 0.999))
    weight_decay = train_cfgs.get("weight_decay", 0)
    grad_norm = train_cfgs.get("grad_norm", None)
    num_samples = train_cfgs.get("eval_num_samples", args.eval_total_size)
    train_cfgs.update({
        "epochs": epochs, "lr": lr, "betas": betas, "weight_decay": weight_decay,
        "grad_norm": grad_norm, "eval_num_samples": num_samples, "eval_batch_size": eval_batch_size})

    rng = torch.Generator().manual_seed(args.master_seed)
    z_train = torch.empty((num_samples,) + input_shape)
    z_eval = torch.empty((data_size,) + input_shape)
    if args.diffusion_type.endswith("jump"):
        z_train.zero_()
        z_eval.zero_()
    elif args.diffusion_type.endswith("gaussian"):
        z_train.normal_(generator=rng)
        rng.manual_seed(args.master_seed)
        z_eval.normal_(generator=rng)
    else:
        raise NotImplementedError(args.diffusion_type)

    for i in range(args.num_runs):
        seed_all(args.seeds[i % len(args.seeds)])

        model = ConditionalMLP(**args.model_cfgs)
        model = ModelWrapper(model, pre_transform, post_transform)
        model.to(device)
        ema = nullcontext()

        optimizer = AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        scheduler = None

        chkpt_dir = f"{args.chkpt_dir}/{args.rel_path}/{args.exp_name}/hpset{rank}/run{i}"
        image_dir = f"{args.image_dir}/{args.rel_path}/{args.exp_name}/hpset{rank}/run{i}"
        text_dir = f"{args.text_dir}/{args.rel_path}/{args.exp_name}/hpset{rank}/run{i}"
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)
            print_fn(f"Checkpoints are saved to {os.path.abspath(chkpt_dir)}.", verbose=args.verbose)
        if not args.text_out and not os.path.exists(image_dir):
            os.makedirs(image_dir)
            print_fn(f"Images are saved to {os.path.abspath(image_dir)}.", verbose=args.verbose)
        if args.text_out and not os.path.exists(text_dir):
            os.makedirs(text_dir)
            print_fn(f"Texts are saved to {os.path.abspath(text_dir)}.", verbose=args.verbose)

        writer = DummyWriter()

        encode_dict = EncodeDict().init(
            type=args.diffusion_type, lbd=diff_kwargs["lbd"], timesteps=diff_kwargs["timesteps"], continuous=cont)
        train_dict = TrainDict().init(
            trainloader=trainloader,
            diffusion=diffusion,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            num_accum=1,
            grad_norm=grad_norm,
            writer=writer,
            start_epoch=0,
            epochs=epochs,
            is_leader=True,
            device=device,
            verbose=args.verbose,
            seed=args.master_seed
        )
        save_dict = SaveDict().init(
            num_samples=num_samples,
            topk=10,
            ndocs=10,
            z_T=z_train,
            xsqrt=xsqrt,
            eval_intv=eval_intv,
            eval_batch_size=eval_batch_size,
            use_pred=args.use_pred,
            chkpt_intv=epochs,
            image_dir=image_dir,
            text_dir=text_dir,
            chkpt_dir=chkpt_dir,
            log_dir=None,
            exp_name=args.exp_name
        )
        data_dict = DataDict().init(dataset=args.dataset, input_shape=input_shape, is_real=args.is_real,
                                    is_bow=args.is_bow, is_ml=args.is_ml, is_image=args.is_image)
        train_loop(encode_dict=encode_dict, train_dict=train_dict, save_dict=save_dict, data_dict=data_dict)

        if eval_batch_size != 0:
            x_gen = []
            for z in z_eval.split(split_size=eval_batch_size, dim=0):
                x_gen.append(diffusion.p_sample(
                    model, z.to(device), return_pred=args.use_pred
                )[int(args.use_pred)].clamp(min=0).cpu().numpy())
            x_gen = np.concatenate(x_gen, axis=0)
        else:
            x_gen = diffusion.p_sample(
                model, z_eval.to(device), return_pred=args.use_pred
            )[int(args.use_pred)].clamp(min=0).cpu().numpy()
        if not (args.is_real or args.is_tfidf):
            x_gen = x_gen.round()
        if args.is_beta or args.is_bin:
            x_gen = np.clip(x_gen, a_min=0, a_max=1)
        if args.is_bow:
            w_distances["spst"].append(trainloader.dataset.compare_sparsity(x_gen)[1]["emd_spst"])
            w_distances["len"].append(trainloader.dataset.compare_length(x_gen)[1]["emd_len"])
        elif args.is_ml:
            w_distances["spst"].append(trainloader.dataset.compare_sparsity(x_gen)[1]["emd_spst"])
            w_distances["deg"].append(trainloader.dataset.compare_degree(x_gen)[1]["emd_deg"])
        else:
            x_gen = np.maximum(x_gen.ravel(), 0)
            q_vals = x_gen
            w_distances.append(wasserstein_distance(p_vals, q_vals))
            if args.is_real:
                try:
                    if args.is_beta:
                        pdfs.append(gaussian_kde(np.concatenate([x_gen, 2 - x_gen, -x_gen])).pdf(xs) * 3)
                    else:
                        pdfs.append(gaussian_kde(np.concatenate([x_gen, -x_gen])).pdf(xs) * 2)
                except np.linalg.LinAlgError:
                    pass
            elif args.is_int:
                hists.append(np.bincount(x_gen.astype("int"), minlength=upper + 1)[:upper + 1] / len(x_gen))

    args.model_cfgs["transform"] = transform
    hps = {
        "diffusion": diff_kwargs,
        "model": args.model_cfgs,
        "trainer": train_cfgs
    }
    with open(os.path.join(args.exp_dir, f"hpset{rank}.txt"), "w") as f:
        if args.is_bow:
            f.write(f"Wasserstein distance (sparsity): {np.mean(w_distances['spst'])} +/- {np.std(w_distances['spst'])}\n")
            f.write(f"Wasserstein distance (length): {np.mean(w_distances['len'])} +/- {np.std(w_distances['len'])}\n")
        elif args.is_ml:
            f.write(f"Wasserstein distance (sparsity): {np.mean(w_distances['spst'])} +/- {np.std(w_distances['spst'])}\n")
            f.write(f"Wasserstein distance (degree): {np.mean(w_distances['deg'])} +/- {np.std(w_distances['deg'])}\n")
        else:
            f.write(f"Wasserstein distance: {np.mean(w_distances)} +/- {np.std(w_distances)}\n")
        f.write("Hyperparameters:\n")
        f.write(dict2str(hps))

    if args.num_runs > 1:
        if args.is_real and len(pdfs):
            pdf_stack = np.stack(pdfs)
            mu, sigma = pdf_stack.mean(axis=0), pdf_stack.std(axis=0)
            np.savez(os.path.join(args.exp_dir, f"hpset{rank}_kde_pdf.npz"), mu=mu, sigma=sigma)
        elif args.is_int:
            hist_stack = np.stack(hists)
            mu, sigma = hist_stack.mean(axis=0), hist_stack.std(axis=0)
            np.savez(os.path.join(args.exp_dir, f"hpset{rank}_hist_pdf.npz"), mu=mu, sigma=sigma)


if __name__ == "__main__":
    from argparse import ArgumentParser

    def parse_seeds(seeds):
        return tuple(map(int, seeds.split(",")))

    parser = ArgumentParser()
    parser.add_argument("--root", default="./data", type=str)
    parser.add_argument("--config-path", required=True, type=str)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--num-gpus", default=1, type=int)
    parser.add_argument("--num-procs", default=0, type=int)
    parser.add_argument("--device-id", default=0, type=int)
    parser.add_argument("--epochs", default=600, type=int)
    parser.add_argument("--eval-intv", default=100, type=int)
    parser.add_argument("--eval-batch-size", default=0, type=int)
    parser.add_argument("--eval-total-size", default=30000, type=int)
    parser.add_argument("--mem-limit", default=48, type=int)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--config-dir", default="./configs", type=str, help="starting directory of config files")
    parser.add_argument("--chkpt-dir", default="./chkpts", type=str)
    parser.add_argument("--exp-dir", default="./exps", type=str)
    parser.add_argument("--image-dir", default="./images", type=str)
    parser.add_argument("--text-dir", default="./texts", type=str)
    parser.add_argument("--num-runs", default=5, type=int)
    parser.add_argument("--seeds", default="468,970,123,527,234", type=parse_seeds)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--use-pred", action="store_true")

    args = parser.parse_args()

    main(args)
