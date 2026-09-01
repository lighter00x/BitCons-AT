#!/usr/bin/env bash
set -Eeuo pipefail

# Timeline: Phase 6 - candidate-family/discrepancy BitCons validation.
# Purpose: fix the near-zero BitMax/gate rates observed in Phase 5 by searching
# P0/P01/P012, refining the strongest bit seed, and gating BitCons with the
# normalized bit/PGD predictive discrepancy. The two causal variants run with
# identical budgets; the completed Phase-5 PGD Base is reused as reference.
# Status: current follow-up experiment, two GPUs with one task per GPU.

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
DESC="${DESC:-bitcons_family_stage2_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/bitcons_family_stage2_${RUN_ID}}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
DRY_RUN="${DRY_RUN:-0}"
STAGE="${STAGE:-all}"

STATUS_DIR="$LOG_DIR/status"
MANIFEST="$LOG_DIR/manifest.tsv"
SUMMARY="$LOG_DIR/status.tsv"
RESULTS="$LOG_DIR/results.tsv"
CONFIG="bitcons_family_at"

# tag|bitcons
TASKS=(
    "family_bitmax_only|false"
    "family_disc_bitcons|true"
)
read -r -a GPU_ARR <<< "$GPUS"

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

experiment_dir() {
    local tag="$1"
    local bitcons="$2"
    echo "$OUT_DIR/${DATASET}_${MODEL}_bitcons_at_none_bitcons_${bitcons}_contrast_false_${DESC}/seed_${SEED}_${RUN_ID}_${tag}"
}

record_status() {
    printf '%s\t%s\t%s\t%s\n' \
        "$1" "$2" "$(date '+%F %T')" "$3" > "$STATUS_DIR/$1.tsv"
}

write_manifest() {
    printf 'tag\tconfig\tbitcons\tpurpose\texperiment_dir\n' > "$MANIFEST"
    local task tag bitcons purpose
    for task in "${TASKS[@]}"; do
        IFS='|' read -r tag bitcons <<< "$task"
        purpose="family search robust CE"
        [[ "$bitcons" == "true" ]] \
            && purpose="family search plus discrepancy-gated BitCons"
        printf '%s\t%s\t%s\t%s\t%s\n' \
            "$tag" "$CONFIG" "$bitcons" "$purpose" \
            "$(experiment_dir "$tag" "$bitcons")" >> "$MANIFEST"
    done
}

preflight() {
    [[ "$EPOCHS" -eq 110 ]] || die "Phase 6 requires EPOCHS=110"
    [[ "$STAGE" == "all" || "$STAGE" == "train" || "$STAGE" == "eval" ]] \
        || die "STAGE must be all, train, or eval"
    [[ "${#GPU_ARR[@]}" -eq 2 ]] || die "Exactly two GPU IDs are required"
    [[ "${GPU_ARR[0]}" != "${GPU_ARR[1]}" ]] || die "GPU IDs must differ"
    [[ -f "$DATA_DIR/cifar-10-batches-py/data_batch_1" ]] \
        || die "CIFAR-10 data is incomplete under $DATA_DIR"
    command -v conda >/dev/null 2>&1 || die "conda is not available"

    FAMILY_GPU_IDS="$GPUS" conda run --no-capture-output -n "$CONDA_ENV" \
        python -c \
        "import os,torch; ids=[int(x) for x in os.environ['FAMILY_GPU_IDS'].split()]; assert torch.cuda.is_available() and max(ids)<torch.cuda.device_count(); print('GPUs:',ids)"
    conda run --no-capture-output -n "$CONDA_ENV" python -c \
        "from autoattack import AutoAttack; print('AutoAttack: OK')"
    MPLCONFIGDIR=/tmp/bitcons-mpl conda run --no-capture-output -n "$CONDA_ENV" \
        python -m unittest discover -s tests -q
}

run_eval() {
    local gpu="$1" tag="$2" exp_dir="$3"
    local result_file="$exp_dir/eval_results_best.txt"
    local eval_log="$LOG_DIR/${tag}_eval_best_gpu${gpu}.log"
    if [[ -s "$result_file" ]]; then
        echo "[SKIP][GPU $gpu] $tag evaluation complete"
        return 0
    fi
    record_status "$tag" running "evaluating best gpu=$gpu"
    echo "[EVAL][GPU $gpu] $tag full attack suite"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
        conda run --no-capture-output -n "$CONDA_ENV" python src/eval.py \
        --exp "$exp_dir" --ckpt best --gpu 0 --data-dir "$DATA_DIR" \
        --batch-size "$EVAL_BATCH_SIZE" --num-workers "$NUM_WORKERS" \
        --all-attacks > "$eval_log" 2>&1
    [[ -s "$result_file" ]] || die "$tag produced no evaluation result"
}

run_task() {
    local gpu="$1" task="$2"
    local tag bitcons exp_dir final_ckpt train_log
    IFS='|' read -r tag bitcons <<< "$task"
    exp_dir="$(experiment_dir "$tag" "$bitcons")"
    final_ckpt="$exp_dir/checkpoints/final_model.pt"
    train_log="$LOG_DIR/${tag}_train_gpu${gpu}.log"
    record_status "$tag" running "gpu=$gpu"

    if [[ "$STAGE" != "eval" ]]; then
        if [[ -s "$final_ckpt" ]]; then
            echo "[SKIP][GPU $gpu] $tag training complete"
        else
            [[ ! -d "$exp_dir" ]] \
                || die "Partial experiment exists; use a new RUN_ID: $exp_dir"
            local train_cmd=(
                conda run --no-capture-output -n "$CONDA_ENV" python src/train.py
                --dataset "$DATASET" --data_dir "$DATA_DIR" --model "$MODEL"
                --config "$CONFIG" --desc "$DESC" --epochs "$EPOCHS"
                --seed "$SEED" --gpu_id 0 --num_workers "$NUM_WORKERS"
                --out_dir "$OUT_DIR" --exp_name "${RUN_ID}_${tag}"
                --perturbation none --no-bitcons_contrast
            )
            [[ "$bitcons" == "true" ]] \
                && train_cmd+=(--bitcons) || train_cmd+=(--no-bitcons)
            echo "[TRAIN][GPU $gpu] $tag"
            CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
                MPLCONFIGDIR=/tmp/bitcons-mpl \
                "${train_cmd[@]}" > "$train_log" 2>&1
            [[ -s "$final_ckpt" ]] || die "$tag produced no final checkpoint"
        fi
    elif [[ ! -s "$final_ckpt" ]]; then
        die "Cannot evaluate $tag; checkpoint is missing"
    fi

    [[ "$STAGE" == "train" ]] || run_eval "$gpu" "$tag" "$exp_dir"
    record_status "$tag" complete "gpu=$gpu"
    echo "[DONE][GPU $gpu] $tag"
}

metric_from_result() {
    awk -F: -v metric="$2" '
        index($1,metric){v=$2;gsub(/^[[:space:]]+|[[:space:]%]+$/, "", v);print v;exit}
    ' "$1"
}

build_outputs() {
    printf 'tag\tstatus\ttimestamp\tdetail\n' > "$SUMMARY"
    printf 'tag\tbitcons\tnatural\tpgd10\tpgd20\tpgd50\tcw\taa\texperiment_dir\n' \
        > "$RESULTS"
    local task tag bitcons exp_dir result_file
    for task in "${TASKS[@]}"; do
        IFS='|' read -r tag bitcons <<< "$task"
        [[ -f "$STATUS_DIR/$tag.tsv" ]] && cat "$STATUS_DIR/$tag.tsv" >> "$SUMMARY"
        exp_dir="$(experiment_dir "$tag" "$bitcons")"
        result_file="$exp_dir/eval_results_best.txt"
        [[ -s "$result_file" ]] || continue
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$tag" "$bitcons" \
            "$(metric_from_result "$result_file" 'Natural Accuracy')" \
            "$(metric_from_result "$result_file" 'PGD-10 Accuracy')" \
            "$(metric_from_result "$result_file" 'PGD-20 Accuracy')" \
            "$(metric_from_result "$result_file" 'PGD-50 Accuracy')" \
            "$(metric_from_result "$result_file" 'C&W Accuracy')" \
            "$(metric_from_result "$result_file" 'AutoAttack Accuracy')" \
            "$exp_dir" >> "$RESULTS"
    done
}

mkdir -p "$LOG_DIR" "$STATUS_DIR"
write_manifest
echo "[FAMILY-BITCONS] run_id=$RUN_ID stage=$STAGE gpus=$GPUS"
if [[ "$DRY_RUN" == "1" ]]; then
    column -t -s $'\t' "$MANIFEST" 2>/dev/null || cat "$MANIFEST"
    echo "[DRY RUN] No GPU work started."
    exit 0
fi

preflight
pids=()
for index in "${!GPU_ARR[@]}"; do
    run_task "${GPU_ARR[$index]}" "${TASKS[$index]}" &
    pids+=("$!")
done
failed=0
set +e
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
set -e
build_outputs
[[ "$failed" -eq 0 ]] || die "At least one Phase-6 worker failed"
echo "[SUMMARY] Status: $SUMMARY"
echo "[SUMMARY] Results: $RESULTS"
