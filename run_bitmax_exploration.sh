#!/usr/bin/env bash
set -Eeuo pipefail

# Timeline: Phase 4 - BitMax candidate-search precursor to current BitCons.
# Purpose: compare PGD with P01/P012 worst low-bit candidates and candidate
# counts K=2/K=4, while projecting every candidate into the clean-centered
# L_inf ball. This tests whether discrete bit risk exists beyond standard PGD.
# Status: supporting inner-optimizer exploration. BitMax is not the paper's
# replacement for BitCons; the current main comparison is RA-WC-BitCons.
# Four runs use two workers, one task per GPU at a time.

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
DESC="${DESC:-bitmax_explore_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/bitmax_explore_${RUN_ID}}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
BITMAX_REFINE_STEPS="${BITMAX_REFINE_STEPS:-2}"
RUN_AA="${RUN_AA:-1}"
DRY_RUN="${DRY_RUN:-0}"

# tag|config|planes|candidate_count
TASKS=(
    "corrected_pgd_base|pgd_at|-|0"
    "bitmax_p01_k2|bitmax_at|0,1|2"
    "bitmax_p012_k2|bitmax_at|0,1,2|2"
    "bitmax_p012_k4|bitmax_at|0,1,2|4"
)

read -r -a GPU_ARR <<< "$GPUS"

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

experiment_dir() {
    local tag="$1"
    local config="$2"
    echo "$OUT_DIR/${DATASET}_${MODEL}_${config}_none_bitcons_false_contrast_false_${DESC}/seed_${SEED}_${RUN_ID}_${tag}"
}

write_manifest() {
    printf 'tag\tconfig\tplanes\tcandidates\texperiment_dir\n' > "$LOG_DIR/manifest.tsv"
    local task tag config planes candidates exp_dir
    for task in "${TASKS[@]}"; do
        IFS='|' read -r tag config planes candidates <<< "$task"
        exp_dir="$(experiment_dir "$tag" "$config")"
        printf '%s\t%s\t%s\t%s\t%s\n' \
            "$tag" "$config" "$planes" "$candidates" "$exp_dir" \
            >> "$LOG_DIR/manifest.tsv"
    done
}

preflight() {
    [[ "$EPOCHS" -eq 110 ]] || die "BitMax exploration requires EPOCHS=110"
    [[ "$RUN_AA" == "0" || "$RUN_AA" == "1" ]] || die "RUN_AA must be 0 or 1"
    [[ "${#GPU_ARR[@]}" -eq 2 ]] || die "Exactly two GPU IDs are required"
    [[ "${GPU_ARR[0]}" != "${GPU_ARR[1]}" ]] || die "GPU IDs must differ"
    [[ -f "$DATA_DIR/cifar-10-batches-py/data_batch_1" ]] \
        || die "Local CIFAR-10 data is incomplete under $DATA_DIR"
    command -v conda >/dev/null 2>&1 || die "conda is not available"

    BITMAX_GPU_IDS="$GPUS" conda run --no-capture-output -n "$CONDA_ENV" python -c \
        "import os, torch; ids=[int(v) for v in os.environ['BITMAX_GPU_IDS'].split()]; assert torch.cuda.is_available(); assert max(ids) < torch.cuda.device_count(); print('GPUs:', ids)"
    if [[ "$RUN_AA" == "1" ]]; then
        conda run --no-capture-output -n "$CONDA_ENV" python -c \
            "from autoattack import AutoAttack; print('AutoAttack: OK')"
    fi
    MPLCONFIGDIR=/tmp/bitcons-mpl conda run --no-capture-output -n "$CONDA_ENV" \
        python -m unittest discover -s tests -q
}

run_task() {
    local gpu="$1"
    local task="$2"
    local tag config planes candidates
    IFS='|' read -r tag config planes candidates <<< "$task"

    local exp_dir final_ckpt train_log eval_log status_file
    exp_dir="$(experiment_dir "$tag" "$config")"
    final_ckpt="$exp_dir/checkpoints/final_model.pt"
    train_log="$LOG_DIR/${tag}_train_gpu${gpu}.log"
    eval_log="$LOG_DIR/${tag}_eval_best_gpu${gpu}.log"
    status_file="$LOG_DIR/status/${tag}.tsv"
    printf 'running\t%s\ttraining gpu=%s\n' "$(date '+%F %T')" "$gpu" > "$status_file"

    if [[ -s "$final_ckpt" ]]; then
        echo "[SKIP][GPU $gpu] $tag training already complete"
    else
        [[ ! -d "$exp_dir" ]] || die "Partial experiment exists: $exp_dir"
        local train_cmd=(
            conda run --no-capture-output -n "$CONDA_ENV" python src/train.py
            --dataset "$DATASET" --data_dir "$DATA_DIR" --model "$MODEL"
            --config "$config" --desc "$DESC" --epochs "$EPOCHS" --seed "$SEED"
            --gpu_id 0 --num_workers "$NUM_WORKERS" --out_dir "$OUT_DIR"
            --exp_name "${RUN_ID}_${tag}" --perturbation none
            --no-bitcons --no-bitcons_contrast
        )
        if [[ "$config" == "bitmax_at" ]]; then
            local planes_spaced="${planes//,/ }"
            local plane_arr
            read -r -a plane_arr <<< "$planes_spaced"
            train_cmd+=(
                --bitmax_planes "${plane_arr[@]}"
                --bitmax_candidates "$candidates"
                --bitmax_refine_steps "$BITMAX_REFINE_STEPS"
            )
        fi

        echo "[TRAIN][GPU $gpu] $tag"
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/bitcons-mpl \
            "${train_cmd[@]}" > "$train_log" 2>&1
        [[ -s "$final_ckpt" ]] || die "$tag did not produce final_model.pt"
    fi

    local result_file="$exp_dir/eval_results_best.txt"
    if [[ ! -s "$result_file" ]]; then
        printf 'running\t%s\tevaluating best gpu=%s\n' "$(date '+%F %T')" "$gpu" > "$status_file"
        local eval_cmd=(
            conda run --no-capture-output -n "$CONDA_ENV" python src/eval.py
            --exp "$exp_dir" --ckpt best --gpu 0 --data-dir "$DATA_DIR"
            --batch-size "$EVAL_BATCH_SIZE" --num-workers "$NUM_WORKERS"
        )
        [[ "$RUN_AA" == "1" ]] && eval_cmd+=(--aa)
        echo "[EVAL][GPU $gpu] $tag best"
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
            "${eval_cmd[@]}" > "$eval_log" 2>&1
        [[ -s "$result_file" ]] || die "$tag best evaluation produced no result"
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
    sed -n '1,10p' "$LOG_DIR/manifest.tsv"
    exit 0
fi

preflight
echo "[BITMAX] run_id=$RUN_ID tasks=${#TASKS[@]} gpus=$GPUS aa=$RUN_AA"

pids=()
for index in "${!GPU_ARR[@]}"; do
    worker "${GPU_ARR[$index]}" "$index" &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
[[ "$failed" -eq 0 ]] || die "At least one BitMax worker failed"
echo "[BITMAX] complete; logs: $LOG_DIR"
