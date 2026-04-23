#!/usr/bin/env bash
set -euo pipefail

# Robust launcher for evaluation jobs.
#
# Common usage:
#   ./run_eval.sh
#
# Evaluate on dual-GPU:
#   GPUS="0 1" PER_GPU_JOBS=1 ./run_eval.sh
#
# Single-GPU multi-process evaluation:
#   GPUS="0" PER_GPU_JOBS=2 ./run_eval.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DATASET="${DATASET:-cifar10}"
MODEL="${MODEL:-resnet18}"
DESC="${DESC:-a100_run}"
OUT_DIR="${OUT_DIR:-outputs}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/dataset}"

METHODS="${METHODS:-pgd_at trades mart rpat}"
PERTURBATION="${PERTURBATION:-none}"
CKPT="${CKPT:-best}"

GPUS="${GPUS:-0 1}"
PER_GPU_JOBS="${PER_GPU_JOBS:-1}"

ENABLE_AA="${ENABLE_AA:-1}"
ENABLE_BITCONS_TEST="${ENABLE_BITCONS_TEST:-0}"
BITCONS_PLANES="${BITCONS_PLANES:-3 4 5}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-logs/eval_${DESC}_${RUN_ID}}"
mkdir -p "$LOG_DIR"

# Split space-separated env values into arrays for loop scheduling.
read -r -a METHOD_ARR <<< "$METHODS"
read -r -a GPU_ARR <<< "$GPUS"
read -r -a BITCONS_ARR <<< "$BITCONS_PLANES"
read -r -a EXTRA_ARR <<< "$EXTRA_ARGS"

if [[ "${#GPU_ARR[@]}" -eq 0 ]]; then
    echo "[ERROR] GPUS is empty."
    exit 1
fi

if [[ "$PER_GPU_JOBS" -lt 1 ]]; then
    echo "[ERROR] PER_GPU_JOBS must be >= 1."
    exit 1
fi

declare -A GPU_RUNNING
declare -A PID_GPU
declare -A PID_TAG
declare -A PID_LOG

# GPU_RUNNING[gpu] = current number of active eval jobs on that GPU.
for gpu in "${GPU_ARR[@]}"; do
    GPU_RUNNING["$gpu"]=0
done

FAILED=0
TOTAL_JOBS=0

cleanup() {
    echo "[INFO] Caught interrupt, terminating child processes..."
    for pid in "${!PID_GPU[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    exit 130
}
trap cleanup INT TERM

reap_finished() {
    for pid in "${!PID_GPU[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            local gpu="${PID_GPU[$pid]}"
            local tag="${PID_TAG[$pid]}"
            local log_file="${PID_LOG[$pid]}"

            # wait returns the child exit code so we can mark failures.
            set +e
            wait "$pid"
            local status=$?
            set -e

            GPU_RUNNING["$gpu"]=$(( GPU_RUNNING["$gpu"] - 1 ))

            if [[ "$status" -eq 0 ]]; then
                echo "[DONE] $tag on GPU $gpu"
            else
                FAILED=1
                echo "[FAIL] $tag on GPU $gpu (exit=$status)"
                echo "       log: $log_file"
            fi

            unset PID_GPU["$pid"]
            unset PID_TAG["$pid"]
            unset PID_LOG["$pid"]
        fi
    done
}

pick_gpu() {
    while true; do
        # Pick the first GPU whose running jobs are below PER_GPU_JOBS.
        for gpu in "${GPU_ARR[@]}"; do
            if [[ "${GPU_RUNNING[$gpu]}" -lt "$PER_GPU_JOBS" ]]; then
                echo "$gpu"
                return
            fi
        done
        reap_finished
        sleep 3
    done
}

resolve_exp_name() {
    local method="$1"
    local base_name="${DATASET}_${MODEL}_${method}_${PERTURBATION}"
    local with_desc="${base_name}_${DESC}"

    # Prefer folder with desc suffix; fallback to folder without desc.
    if [[ -d "$OUT_DIR/$with_desc" ]]; then
        echo "$with_desc"
        return 0
    fi

    if [[ -d "$OUT_DIR/$base_name" ]]; then
        echo "$base_name"
        return 0
    fi

    return 1
}

launch_eval_job() {
    local method="$1"
    local gpu="$2"
    local exp_name="$3"

    local tag="${DATASET}_${MODEL}_${method}_${CKPT}"
    local log_file="${LOG_DIR}/${tag}_gpu${gpu}.log"

    # Build base eval command first, then append optional switches.
    local cmd=(
        python src/eval.py
        --exp "$exp_name"
        --ckpt "$CKPT"
        --gpu 0
        --data-dir "$DATA_DIR"
        "${EXTRA_ARR[@]}"
    )

    if [[ "$ENABLE_AA" == "1" ]]; then
        cmd+=(--aa)
    fi

    if [[ "$ENABLE_BITCONS_TEST" == "1" ]]; then
        cmd+=(--bitcons-test --bitcons-planes "${BITCONS_ARR[@]}")
    fi

    echo "[RUN ] $tag on GPU $gpu (exp=$exp_name)"

    # Same mapping rule as train script:
    # one visible physical GPU -> local GPU index 0 in the process.
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "${cmd[@]}" > "$log_file" 2>&1 &

    local pid=$!
    PID_GPU["$pid"]="$gpu"
    PID_TAG["$pid"]="$tag"
    PID_LOG["$pid"]="$log_file"
    GPU_RUNNING["$gpu"]=$(( GPU_RUNNING["$gpu"] + 1 ))
    TOTAL_JOBS=$(( TOTAL_JOBS + 1 ))
}

echo "[INFO] dataset=$DATASET model=$MODEL desc=$DESC ckpt=$CKPT"
echo "[INFO] methods=$METHODS"
echo "[INFO] gpus=$GPUS per_gpu_jobs=$PER_GPU_JOBS"
echo "[INFO] data_dir=$DATA_DIR"
echo "[INFO] enable_aa=$ENABLE_AA bitcons_test=$ENABLE_BITCONS_TEST"
echo "[INFO] logs=$LOG_DIR"

for method in "${METHOD_ARR[@]}"; do
    if ! exp_name="$(resolve_exp_name "$method")"; then
        echo "[WARN] Skip $method: experiment folder not found under $OUT_DIR"
        continue
    fi

    # Block until at least one GPU has a free slot.
    gpu="$(pick_gpu)"
    launch_eval_job "$method" "$gpu" "$exp_name"
    reap_finished
done

if [[ "$TOTAL_JOBS" -eq 0 ]]; then
    echo "[SUMMARY] No evaluation jobs launched."
    exit 1
fi

while [[ "${#PID_GPU[@]}" -gt 0 ]]; do
    reap_finished
    sleep 3
done

if [[ "$FAILED" -ne 0 ]]; then
    echo "[SUMMARY] Finished with failures. Please check logs under $LOG_DIR"
    exit 1
fi

echo "[SUMMARY] All $TOTAL_JOBS evaluation jobs completed successfully."
echo "[SUMMARY] Logs: $LOG_DIR"