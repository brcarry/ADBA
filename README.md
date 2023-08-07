# [Anti Distillation Backdoor Attacks](https://dl.acm.org/doi/pdf/10.1145/3474085.3475254)

This repository is an **un**official PyTorch implementation of:

[Anti Distillation Backdoor Attacks](https://dl.acm.org/doi/pdf/10.1145/3474085.3475254)(MM '21)

which is closed source.:fu:

## Requirements

- This codebase is written for python3 (used python 3.6 while implementing).
- We use Pytorch version of 1.8.2, 11.4 CUDA version.
- To install necessary python packages,

```shell
conda env create -f environment.yml
pip install -r requirements.txt
```

## How to Run Codes?

### Local Training

```shell
python3 main.py --type=pretrain  --lr=0.001 --model=res --dataset=cifar10 --partition=iid --seed=1 --local_ep=200  --alpha_backdoor=0.4 --txtpath=./saved/adba_result.txt
```

### Global Distillation

```shell
python3 main.py --type=distillation  --lr=0.001 --model=res --dataset=cifar10 --partition=iid --seed=1 --epochs=200  --alpha_backdoor=0.4 --txtpath=./saved/adba_result.txt
```

## Acknowledgement

This repository is based on the code of  [DENSE](https://github.com/zj-jayzhang/DENSE)(NeurIPS '22), and I also learn a lot from it when coding.🥰
