#!/bin/bash
set -uo pipefail

# Timeline: Phase 0 - original repository training example (legacy utility).
# Purpose: launch PGD-AT/TRADES/MART/RPAT baselines on four fixed GPU IDs and
# demonstrate the basic src/train.py CLI. It does not test BitCons and has no
# preflight, resume, manifest, or consolidated evaluation support.
# Status: historical example only; do not use for the current paper pipeline.
# Use run_rawc_bitcons_stage1_dual.sh for the current main experiment.

mkdir -p logs
desc="test"
nohup python src/train.py \
    --dataset cifar10 \
    --model resnet18 \
    --config pgd_at \
    --no-bitcons --no-bitcons_contrast \
    --desc ${desc} \
    --seed 42 \
    --gpu_id 0 \
    --out_dir outputs/ \
    > logs/cifar10_pgd_at_${desc}.out 2>&1 &

nohup python src/train.py \
    --dataset cifar10 \
    --model resnet18 \
    --config trades \
    --no-bitcons --no-bitcons_contrast \
    --desc ${desc} \
    --seed 42 \
    --gpu_id 1 \
    --out_dir outputs/ \
    > logs/cifar10_trades_${desc}.out 2>&1 &

nohup python src/train.py \
    --dataset cifar10 \
    --model resnet18 \
    --config mart \
    --no-bitcons --no-bitcons_contrast \
    --desc ${desc} \
    --seed 42 \
    --gpu_id 2 \
    --out_dir outputs/ \
    > logs/cifar10_mart_${desc}.out 2>&1 &

nohup python src/train.py \
    --dataset cifar10 \
    --model resnet18 \
    --config rpat \
    --no-bitcons --no-bitcons_contrast \
    --desc ${desc} \
    --seed 42 \
    --gpu_id 3 \
    --out_dir outputs/ \
    > logs/cifar10_rpat_${desc}.out 2>&1 &
