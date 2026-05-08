#!/bin/bash

# 定义可用资源和任务参数
GPUS=(0 3 4 7)
DATASETS=("cifar10" "cifar100" "tinynet")
# DATASETS=("tinynet")
# MODELS=("resnet18" "wrn28_10" "wrn34_10" "preactresnet18")
MODELS=("resnet18")
# CONFIGS=("pgd_at" "trades" "mart" "rpat")
CONFIGS=("trades" "mart" "rpat")
# CONFIGS=("pgd_at")
DESC="bitcons_true_contrast_true"
SEED=4243

RUN_TIME=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/train_${RUN_TIME}"
mkdir -p "${LOG_DIR}"

# 定义并行运行函数
run_task() {
    local dataset=$1
    local model=$2
    local config=$3
    local gpu_id=$4
    
    echo "[$(date)] 开始运行任务: dataset=${dataset} model=${model} config=${config} 在 GPU ${gpu_id} 上"
    
    # 执行实际的代码
    python src/train.py \
        --dataset ${dataset} \
        --model ${model} \
        --config ${config} \
        --desc ${DESC} \
        --seed ${SEED} \
        --gpu_id ${gpu_id} \
        --out_dir outputs/ \
        > "${LOG_DIR}/${dataset}_${model}_${config}_${DESC}.out" 2>&1
        
    echo "[$(date)] 任务完成: dataset=${dataset} model=${model} config=${config} (GPU ${gpu_id})"
}

# 数组用于跟踪每个 GPU 上正在运行的进程 PID
pids=()
for ((i=0; i<${#GPUS[@]}; i++)); do
    pids[i]=""
done

# 将所有组合展开成任务列表
TASKS=()
for d in "${DATASETS[@]}"; do
    for m in "${MODELS[@]}"; do
        for c in "${CONFIGS[@]}"; do
            TASKS+=("$d $m $c")
        done
    done
done

# 分配循环
for task in "${TASKS[@]}"; do
    # 解析出当前任务的 dataset, model, config
    read -r curr_dataset curr_model curr_config <<< "$task"
    
    assigned=false
    
    while [ "$assigned" = false ]; do
        # 遍历所有可用的 GPU
        for ((i=0; i<${#GPUS[@]}; i++)); do
            gpu_pid=${pids[$i]}
            
            # 如果对应的 PID 为空，或该 PID 的进程已经结束
            if [ -z "$gpu_pid" ] || ! kill -0 "$gpu_pid" 2>/dev/null; then
                # 空闲，安排执行
                run_task "$curr_dataset" "$curr_model" "$curr_config" "${GPUS[$i]}" &
                # 记录新的 PID
                pids[$i]=$!
                assigned=true
                break
            fi
        done
        
        # 如果没有分配出去，等待一段时间再尝试
        if [ "$assigned" = false ]; then
            sleep 10
        fi
    done
done

# 等待所有后台任务完成
echo "所有任务已分配完毕，等待后台任务执行结束..."
wait
echo "所有任务执行完成！"