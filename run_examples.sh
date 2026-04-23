#!/bin/bash
mkdir -p logs
desc="test"
nohup python src/train.py \
    --dataset cifar10 \
    --model resnet18 \
    --config pgd_at \
    --desc ${desc} \
    --seed 42 \
    --gpu_id 0 \
    --out_dir outputs/ \
    > logs/cifar10_${desc}.out 2>&1 & 

nohup python src/train.py \
    --dataset cifar10 \
    --model resnet18 \
    --config trades \
    --desc ${desc} \
    --seed 42 \
    --gpu_id 1 \
    --out_dir outputs/ \
    > logs/cifar10_${desc}.out 2>&1 & 

nohup python src/train.py \
    --dataset cifar10 \
    --model resnet18 \
    --config mart \
    --desc ${desc} \
    --seed 42 \
    --gpu_id 2 \
    --out_dir outputs/ \
    > logs/cifar10_${desc}.out 2>&1 & 

nohup python src/train.py \
    --dataset cifar10 \
    --model resnet18 \
    --config rpat \
    --desc ${desc} \
    --seed 42 \
    --gpu_id 3 \
    --out_dir outputs/ \
    > logs/cifar10_${desc}.out 2>&1 & 