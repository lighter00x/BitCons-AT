#!/bin/bash

OUTPUTS_DIR="/home/xq/projects/BitCons-AT/bitoutputs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGS_DIR="logs/eval_adv_$TIMESTAMP"
mkdir -p "$LOGS_DIR"

MASTER_LOG="$LOGS_DIR/batch_eval_main.log"

# 定义可用资源：8 张卡
GPUS=(0 1 2 3 4 5 6 7)

# 数组用于跟踪每个 GPU 上正在运行的进程 PID
pids=()
for ((i=0; i<${#GPUS[@]}; i++)); do
    pids[i]=""
done

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Scanning $OUTPUTS_DIR for experiment outputs..." | tee -a "$MASTER_LOG"

# 定义任务运行函数
run_eval_task() {
    local exp_name=$1
    local latest_seed_dir=$2
    local gpu_id=$3

    echo "==================================================" | tee -a "$MASTER_LOG"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting eval for: $exp_name on GPU $gpu_id" | tee -a "$MASTER_LOG"
    echo "Latest seed path : $latest_seed_dir" | tee -a "$MASTER_LOG"
    
    python src/eval.py \
        --bitcons-planes 0 1 2 \
        --bitcons-test \
        --gpu "$gpu_id" \
        --aa \
        --exp "$latest_seed_dir" \
        > "$LOGS_DIR/${exp_name}_eval.out" 2>&1
        
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Finished eval for $exp_name. Log saved to $LOGS_DIR/${exp_name}_eval.out" | tee -a "$MASTER_LOG"
}

# 遍历所有的实验参数组合文件夹
for EXP_DIR in "$OUTPUTS_DIR"/*; do
    if [ -d "$EXP_DIR" ]; then
        EXP_NAME=$(basename "$EXP_DIR")
        
        # 查找最新的 seed 文件夹
        LATEST_SEED_DIR=$(find "$EXP_DIR" -maxdepth 1 -type d -name "seed_*" | sort | tail -n 1)
        
        if [ -n "$LATEST_SEED_DIR" ]; then
            
            assigned=false
            
            while [ "$assigned" = false ]; do
                # 遍历所有可用的 GPU
                for ((i=0; i<${#GPUS[@]}; i++)); do
                    gpu_pid=${pids[$i]}
                    
                    # 如果对应的 PID 为空，或该 PID 的进程已经结束
                    if [ -z "$gpu_pid" ] || ! kill -0 "$gpu_pid" 2>/dev/null; then
                        # 空闲，安排执行进后台
                        run_eval_task "$EXP_NAME" "$LATEST_SEED_DIR" "${GPUS[$i]}" &
                        
                        # 记录新的 PID
                        pids[$i]=$!
                        assigned=true
                        break
                    fi
                done
                
                # 如果没有分配出去，等待一段时间再尝试
                if [ "$assigned" = false ]; then
                    sleep 5
                fi
            done
        fi
    fi
done

echo "[$(date +'%Y-%m-%d %H:%M:%S')] 所有任务已分配完毕，等待后台任务执行结束..." | tee -a "$MASTER_LOG"
wait

echo "==================================================" | tee -a "$MASTER_LOG"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] All queued evaluation tasks have been completed." | tee -a "$MASTER_LOG"
