#!/usr/bin/env bash
set -Eeuo pipefail

# Timeline: Phase 5 - current RA-WC-BitCons Stage-1 screen, single-GPU variant.
# Purpose: isolate the contribution chain PGD Base -> BitMax-only -> risk-
# adaptive worst-case BitCons under one seed and one training budget.
# Status: current-method fallback for one-GPU machines. On this dual-A100 host,
# prefer run_rawc_bitcons_stage1_dual.sh, which runs the same causal comparison
# faster and writes consolidated status/results tables.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-bit}"
DATASET="${DATASET:-cifar10}"
MODEL="${MODEL:-resnet18}"
SEED="${SEED:-4243}"
EPOCHS="${EPOCHS:-110}"
GPU="${GPU:-0}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/dataset/data}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/outputs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
DESC="${DESC:-rawc_bitcons_stage1_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/rawc_bitcons_stage1_${RUN_ID}}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
RUN_AA="${RUN_AA:-1}"
DRY_RUN="${DRY_RUN:-0}"

# tag|config|bitcons
TASKS=(
    "corrected_pgd_base|pgd_at|false"
    "bitmax_only|bitcons_at|false"
    "rawc_bitcons|bitcons_at|true"
)

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

experiment_dir() {
    local tag="$1"
    local config="$2"
    local bitcons="$3"
    echo "$OUT_DIR/${DATASET}_${MODEL}_${config}_none_bitcons_${bitcons}_contrast_false_${DESC}/seed_${SEED}_${RUN_ID}_${tag}"
}

write_manifest() {
    printf 'tag\tconfig\tbitcons\texperiment_dir\n' > "$LOG_DIR/manifest.tsv"
    local task tag config bitcons exp_dir
    for task in "${TASKS[@]}"; do
        IFS='|' read -r tag config bitcons <<< "$task"
        exp_dir="$(experiment_dir "$tag" "$config" "$bitcons")"
        printf '%s\t%s\t%s\t%s\n' \
            "$tag" "$config" "$bitcons" "$exp_dir" \
            >> "$LOG_DIR/manifest.tsv"
    done
}

preflight() {
    [[ "$EPOCHS" -eq 110 ]] || die "Stage 1 requires EPOCHS=110"
    [[ "$RUN_AA" == "0" || "$RUN_AA" == "1" ]] || die "RUN_AA must be 0 or 1"
    [[ -f "$DATA_DIR/cifar-10-batches-py/data_batch_1" ]] \
        || die "Local CIFAR-10 data is incomplete under $DATA_DIR"
    command -v conda >/dev/null 2>&1 || die "conda is not available"

    CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n "$CONDA_ENV" \
        python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
    if [[ "$RUN_AA" == "1" ]]; then
        conda run --no-capture-output -n "$CONDA_ENV" python -c \
            "from autoattack import AutoAttack; print('AutoAttack: OK')"
    fi
    MPLCONFIGDIR=/tmp/bitcons-mpl conda run --no-capture-output -n "$CONDA_ENV" \
        python -m unittest discover -s tests -q
}

run_task() {
    local task="$1"
    local tag config bitcons exp_dir final_ckpt train_log eval_log status_file
    IFS='|' read -r tag config bitcons <<< "$task"
    exp_dir="$(experiment_dir "$tag" "$config" "$bitcons")"
    final_ckpt="$exp_dir/checkpoints/final_model.pt"
    train_log="$LOG_DIR/${tag}_train_gpu${GPU}.log"
    eval_log="$LOG_DIR/${tag}_eval_best_gpu${GPU}.log"
    status_file="$LOG_DIR/status/${tag}.tsv"
    printf 'running\t%s\ttraining gpu=%s\n' "$(date '+%F %T')" "$GPU" > "$status_file"

    if [[ -s "$final_ckpt" ]]; then
        echo "[SKIP] $tag training already complete"
    else
        [[ ! -d "$exp_dir" ]] || die "Partial experiment exists: $exp_dir"
        local train_cmd=(
            conda run --no-capture-output -n "$CONDA_ENV" python src/train.py
            --dataset "$DATASET" --data_dir "$DATA_DIR" --model "$MODEL"
            --config "$config" --desc "$DESC" --epochs "$EPOCHS" --seed "$SEED"
            --gpu_id 0 --num_workers "$NUM_WORKERS" --out_dir "$OUT_DIR"
            --exp_name "${RUN_ID}_${tag}" --perturbation none
            --no-bitcons_contrast
        )
        if [[ "$bitcons" == "true" ]]; then
            train_cmd+=(--bitcons)
        else
            train_cmd+=(--no-bitcons)
        fi

        echo "[TRAIN][GPU $GPU] $tag"
        CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/bitcons-mpl \
            "${train_cmd[@]}" > "$train_log" 2>&1
        [[ -s "$final_ckpt" ]] || die "$tag did not produce final_model.pt"
    fi

    local result_file="$exp_dir/eval_results_best.txt"
    if [[ ! -s "$result_file" ]]; then
        printf 'running\t%s\tevaluating best gpu=%s\n' "$(date '+%F %T')" "$GPU" > "$status_file"
        local eval_cmd=(
            conda run --no-capture-output -n "$CONDA_ENV" python src/eval.py
            --exp "$exp_dir" --ckpt best --gpu 0 --data-dir "$DATA_DIR"
            --batch-size "$EVAL_BATCH_SIZE" --num-workers "$NUM_WORKERS"
        )
        [[ "$RUN_AA" == "1" ]] && eval_cmd+=(--aa)
        echo "[EVAL][GPU $GPU] $tag best"
        CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
            "${eval_cmd[@]}" > "$eval_log" 2>&1
        [[ -s "$result_file" ]] || die "$tag best evaluation produced no result"
    fi

    printf 'complete\t%s\tgpu=%s\n' "$(date '+%F %T')" "$GPU" > "$status_file"
    echo "[DONE][GPU $GPU] $tag"
}

mkdir -p "$LOG_DIR/status"
write_manifest
if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY RUN] No training or evaluation will be started."
    sed -n '1,10p' "$LOG_DIR/manifest.tsv"
    exit 0
fi

preflight
echo "[RA-WC-BITCONS] run_id=$RUN_ID tasks=${#TASKS[@]} gpu=$GPU aa=$RUN_AA"
for task in "${TASKS[@]}"; do
    run_task "$task"
done
echo "[RA-WC-BITCONS] complete; logs: $LOG_DIR"
