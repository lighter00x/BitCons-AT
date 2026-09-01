#!/usr/bin/env bash
set -Eeuo pipefail

# Timeline: Phase 2 - low-bit auxiliary-stream screening (historical study).
# Purpose: move fragile planes from the original higher-bit choice to P01/P012,
# sweep smaller alignment weights, and separate Mask-CE/Align/Contrast effects.
# It showed that weaker low-bit alignment avoided some collapse but still did
# not beat the standard robust baseline consistently.
# Status: historical reproduction only; superseded by threat-ball candidates
# and risk-adaptive consistency. Each GPU runs one job at a time.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-bit}"
DATASET="${DATASET:-cifar10}"
MODEL="${MODEL:-resnet18}"
SEED="${SEED:-4243}"
EPOCHS="${EPOCHS:-110}"
GPUS="${GPUS:-0 1}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/dataset/data}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/outputs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
DESC="${DESC:-lowbit_screen_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/lowbit_screen_${RUN_ID}}"
NUM_WORKERS="${NUM_WORKERS:-4}"
RUN_BEST_EVAL="${RUN_BEST_EVAL:-0}"
RUN_AA="${RUN_AA:-0}"
DRY_RUN="${DRY_RUN:-0}"

WARMUP=60
CONTRAST_LAM=0.001

# tag|bitcons|planes|alpha|mask_ce|align|contrast
TASKS=(
    "pgd_at_base|0|-|0|0|0|0"
    "align_p012_a010|1|0,1,2|0.10|0|1|0"
    "align_p012_a025|1|0,1,2|0.25|0|1|0"
    "align_p012_a050|1|0,1,2|0.50|0|1|0"
    "core_p012_a010|1|0,1,2|0.10|1|1|0"
    "core_p012_a025|1|0,1,2|0.25|1|1|0"
    "align_p01_a025|1|0,1|0.25|0|1|0"
    "full_p012_a025|1|0,1,2|0.25|1|1|1"
)

read -r -a GPU_ARR <<< "$GPUS"

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

experiment_dir() {
    local tag="$1"
    local bitcons="$2"
    local contrast="$3"
    local bitcons_name="false"
    local contrast_name="false"
    [[ "$bitcons" == "1" ]] && bitcons_name="true"
    [[ "$contrast" == "1" ]] && contrast_name="true"
    echo "$OUT_DIR/${DATASET}_${MODEL}_pgd_at_none_bitcons_${bitcons_name}_contrast_${contrast_name}_${DESC}/seed_${SEED}_${RUN_ID}_${tag}"
}

preflight() {
    [[ "$EPOCHS" -eq 110 ]] || die "Screening requires EPOCHS=110"
    [[ "$RUN_BEST_EVAL" == "0" || "$RUN_BEST_EVAL" == "1" ]] \
        || die "RUN_BEST_EVAL must be 0 or 1"
    [[ "$RUN_AA" == "0" || "$RUN_AA" == "1" ]] || die "RUN_AA must be 0 or 1"
    [[ "${#GPU_ARR[@]}" -eq 2 ]] || die "Exactly two GPU IDs are required"
    [[ "${GPU_ARR[0]}" != "${GPU_ARR[1]}" ]] || die "GPU IDs must differ"
    [[ -d "$DATA_DIR" ]] || die "Dataset directory does not exist: $DATA_DIR"
    command -v conda >/dev/null 2>&1 || die "conda is not available"

    SCREEN_GPU_IDS="$GPUS" conda run --no-capture-output -n "$CONDA_ENV" python -c \
        "import os, torch; ids=[int(v) for v in os.environ['SCREEN_GPU_IDS'].split()]; assert torch.cuda.is_available(); assert max(ids) < torch.cuda.device_count(); print('GPUs:', ids)"
    conda run --no-capture-output -n "$CONDA_ENV" \
        python -m unittest discover -s tests -q
}

write_manifest() {
    local manifest="$LOG_DIR/manifest.tsv"
    printf 'tag\tbitcons\tplanes\talpha\tmask_ce\talign\tcontrast\texperiment_dir\n' > "$manifest"
    local task tag bitcons planes alpha mask_ce align contrast exp_dir
    for task in "${TASKS[@]}"; do
        IFS='|' read -r tag bitcons planes alpha mask_ce align contrast <<< "$task"
        exp_dir="$(experiment_dir "$tag" "$bitcons" "$contrast")"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$tag" "$bitcons" "$planes" "$alpha" "$mask_ce" "$align" \
            "$contrast" "$exp_dir" >> "$manifest"
    done
}

run_task() {
    local gpu="$1"
    local task="$2"
    local tag bitcons planes alpha mask_ce align contrast
    IFS='|' read -r tag bitcons planes alpha mask_ce align contrast <<< "$task"

    local exp_dir final_ckpt train_log status_file
    exp_dir="$(experiment_dir "$tag" "$bitcons" "$contrast")"
    final_ckpt="$exp_dir/checkpoints/final_model.pt"
    train_log="$LOG_DIR/${tag}_train_gpu${gpu}.log"
    status_file="$LOG_DIR/status/${tag}.tsv"
    printf 'running\t%s\tgpu=%s\n' "$(date '+%F %T')" "$gpu" > "$status_file"

    if [[ -s "$final_ckpt" ]]; then
        echo "[SKIP][GPU $gpu] $tag training already complete"
    else
        [[ ! -d "$exp_dir" ]] \
            || die "Partial experiment exists; use a new RUN_ID: $exp_dir"

        local train_cmd=(
            conda run --no-capture-output -n "$CONDA_ENV" python src/train.py
            --dataset "$DATASET" --data_dir "$DATA_DIR" --model "$MODEL"
            --config pgd_at --desc "$DESC" --epochs "$EPOCHS" --seed "$SEED"
            --gpu_id 0 --num_workers "$NUM_WORKERS" --out_dir "$OUT_DIR"
            --exp_name "${RUN_ID}_${tag}" --perturbation none
        )
        if [[ "$bitcons" == "1" ]]; then
            local planes_spaced="${planes//,/ }"
            local plane_arr
            read -r -a plane_arr <<< "$planes_spaced"
            train_cmd+=(
                --bitcons --bitcons_planes "${plane_arr[@]}"
                --bitcons_alpha "$alpha" --bitcons_warmup "$WARMUP"
                --bitcons_ce_weight "$mask_ce"
                --bitcons_align_weight "$align"
                --bitcons_contrast_lam "$CONTRAST_LAM"
            )
        else
            train_cmd+=(--no-bitcons)
        fi
        if [[ "$contrast" == "1" ]]; then
            train_cmd+=(--bitcons_contrast)
        else
            train_cmd+=(--no-bitcons_contrast)
        fi

        echo "[TRAIN][GPU $gpu] $tag"
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/bitcons-mpl \
            "${train_cmd[@]}" > "$train_log" 2>&1
        [[ -s "$final_ckpt" ]] || die "$tag did not produce final_model.pt"
    fi

    if [[ "$RUN_BEST_EVAL" == "1" ]]; then
        local eval_log="$LOG_DIR/${tag}_eval_best_gpu${gpu}.log"
        local eval_cmd=(
            conda run --no-capture-output -n "$CONDA_ENV" python src/eval.py
            --exp "$exp_dir" --ckpt best --gpu 0 --data-dir "$DATA_DIR"
            --num-workers "$NUM_WORKERS"
        )
        [[ "$RUN_AA" == "1" ]] && eval_cmd+=(--aa)
        echo "[EVAL][GPU $gpu] $tag best"
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
            "${eval_cmd[@]}" > "$eval_log" 2>&1
    fi

    printf 'complete\t%s\tgpu=%s\n' "$(date '+%F %T')" "$gpu" > "$status_file"
    echo "[DONE][GPU $gpu] $tag"
}

worker() {
    local gpu="$1"
    local offset="$2"
    local index
    for ((index=offset; index<${#TASKS[@]}; index+=${#GPU_ARR[@]})); do
        run_task "$gpu" "${TASKS[$index]}"
    done
}

mkdir -p "$LOG_DIR/status"
write_manifest
if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY RUN] No training or evaluation will be started."
    sed -n '1,20p' "$LOG_DIR/manifest.tsv"
    exit 0
fi

preflight
echo "[SCREEN] run_id=$RUN_ID tasks=${#TASKS[@]} gpus=$GPUS best_eval=$RUN_BEST_EVAL aa=$RUN_AA"

pids=()
for index in "${!GPU_ARR[@]}"; do
    worker "${GPU_ARR[$index]}" "$index" &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
[[ "$failed" -eq 0 ]] || die "At least one screening worker failed"
echo "[SCREEN] complete; logs: $LOG_DIR"
