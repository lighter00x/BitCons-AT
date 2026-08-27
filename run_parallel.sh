#!/bin/bash
set -uo pipefail

# 定义可用资源和任务参数
GPUS=(0 1 2 3)
# DATASETS=("cifar10" "cifar100" "tinynet")
DATASETS=("tinynet")
# MODELS=("resnet18" "wrn28_10" "wrn34_10" "preactresnet18")
MODELS=("resnet18")
CONFIGS=("pgd_at" "trades" "mart" "rpat")
# CONFIGS=("trades" "mart" "rpat")
# CONFIGS=("pgd_at")
BITCONS=false
BITCONS_CONTRAST=false
BITCONS_CE_WEIGHT=1.0
BITCONS_ALIGN_WEIGHT=1.0
DESC="baseline"
SEED=4243

if [ "$BITCONS_CONTRAST" = true ] && [ "$BITCONS" != true ]; then
    echo "BITCONS_CONTRAST=true requires BITCONS=true" >&2
    exit 1
fi

BITCONS_ARGS=(--no-bitcons --no-bitcons_contrast)
if [ "$BITCONS" = true ]; then
    BITCONS_ARGS=(
        --bitcons
        --no-bitcons_contrast
        --bitcons_ce_weight "$BITCONS_CE_WEIGHT"
        --bitcons_align_weight "$BITCONS_ALIGN_WEIGHT"
    )
fi
if [ "$BITCONS_CONTRAST" = true ]; then
    BITCONS_ARGS=(
        --bitcons
        --bitcons_contrast
        --bitcons_ce_weight "$BITCONS_CE_WEIGHT"
        --bitcons_align_weight "$BITCONS_ALIGN_WEIGHT"
    )
fi

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
    if python src/train.py \
        --dataset ${dataset} \
        --model ${model} \
        --config ${config} \
        --desc ${DESC} \
        "${BITCONS_ARGS[@]}" \
        --seed ${SEED} \
        --gpu_id ${gpu_id} \
        --out_dir outputs/ \
        > "${LOG_DIR}/${dataset}_${model}_${config}_${DESC}.out" 2>&1; then
        echo "[$(date)] 任务完成: dataset=${dataset} model=${model} config=${config} (GPU ${gpu_id})"
        return 0
    else
        local status=$?
        echo "[$(date)] 任务失败: dataset=${dataset} model=${model} config=${config} (GPU ${gpu_id}, exit=${status})" >&2
        return "$status"
    fi
}

# 数组用于跟踪每个 GPU 上正在运行的进程 PID
pids=()
FAILED=0
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

            if [ -n "$gpu_pid" ] && ! kill -0 "$gpu_pid" 2>/dev/null; then
                if ! wait "$gpu_pid"; then
                    FAILED=1
                fi
                pids[$i]=""
                gpu_pid=""
            fi

            # 如果对应的 PID 为空，则该 GPU 可以接收新任务
            if [ -z "$gpu_pid" ]; then
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
for pid in "${pids[@]}"; do
    if [ -n "$pid" ] && ! wait "$pid"; then
        FAILED=1
    fi
done

if [ "$FAILED" -ne 0 ]; then
    echo "部分训练任务失败，请检查 ${LOG_DIR} 下的日志。" >&2
    exit 1
fi

echo "所有任务执行完成！"
