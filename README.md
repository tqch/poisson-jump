<p align="center"><img alt="thinning" src="./assets/thinning.webp"></p>

---

# Learning to Jump: Thinning and Thickening Latent Counts for Generative Modeling

This repo contains the official PyTorch implementation for the ICML 2023 paper "Learning to Jump: Thinning and Thickening Latent Counts for Generative Modeling" [[arXiv]](https://arxiv.org/abs/2305.18375)

## Dependencies
see `requirements.txt`

## Dataset preparation
NeurIPS Papers: Please follow [[GitHub]](https://github.com/benhamner/nips-papers/tree/master) or directly download from [[Kaggle]](https://www.kaggle.com/datasets/benhamner/nips-papers). Rename the ZIP file as `nips-papers.zip` if necessary and move it to `data` folder within `poisson-jump`.


##  How to train JUMP models

### 1. Train on univariate toy data (example)

```shell
# train nbinom
python train_toy.py --config-path ./configs/synthetic/jump/nbinom_jump.json --train --num-runs 5 --num-gpus 1 --verbose
```

- `num-runs`: number of experiment runs
- `num-gpus`: number of GPUs available (used for hyperparameter sweeps)
- `train`: whether to train or to summarize existing experiment results
- `verbose`: print progress bar and other training logs

### 2. Train on image data (example)

```shell
# cifar-10 (1 GPU)
python train.py --config-path ./configs/cifar10_ordinal_jump.json --verbose
# cifar-10 (4 GPUs)
torchrun --standalone --nproc_per_node 4 --rdzv_backend c10d train.py --config-path ./configs/cifar10_ordinal_jump.json --distributed-mode elastic --num-gpus 4 --verbose
```

## Citation

```bibtex
@inproceedings{chen2023learning,
	title={Learning to Jump: Thinning and Thickening Latent Counts for Generative Modeling},
	author={Chen, Tianqi and Zhou, Mingyuan},
	booktitle={International Conference on Machine Learning (ICML)},
	year={2023},
}
```

## License

MIT license

---
<p align="center"><img alt="thickening" src="./assets/thickening.webp"></p>