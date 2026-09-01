#!/usr/bin/env bash
set -Eeuo pipefail

# Timeline: Phase 3.5 - quantization causal control after the BPDA study.
# Purpose: separate the effect of ordinary 8-bit rounding from clearing low bit
# planes, using an adaptive BPDA evaluation under the same CIFAR-10 protocol.
# Status: supporting control only. Run it when the matching control result is
# missing; quantization/BPDA do not replace BitCons in the current paper method.

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
DESC="${DESC:-quantize_only_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/quantize_only_${RUN_ID}}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
RUN_AA="${RUN_AA:-1}"
DRY_RUN="${DRY_RUN:-0}"

TAG="quantize_only"
CONFIG="quantize_at"
EXP_DIR="$OUT_DIR/${DATASET}_${MODEL}_${CONFIG}_none_bitcons_false_contrast_false_${DESC}/seed_${SEED}_${RUN_ID}_${TAG}"
STATUS_FILE="$LOG_DIR/status/${TAG}.tsv"
TRAIN_LOG="$LOG_DIR/${TAG}_train_gpu${GPU}.log"
EVAL_LOG="$LOG_DIR/${TAG}_eval_best_gpu${GPU}.log"

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

preflight() {
    [[ "$EPOCHS" -eq 110 ]] || die "Quantize-only exploration requires EPOCHS=110"
    [[ "$RUN_AA" == "0" || "$RUN_AA" == "1" ]] || die "RUN_AA must be 0 or 1"
    [[ "$GPU" =~ ^[0-9]+$ ]] || die "GPU must be a non-negative integer"
    [[ -f "$DATA_DIR/cifar-10-batches-py/data_batch_1" ]] \
        || die "Local CIFAR-10 data is incomplete under $DATA_DIR"
    [[ ! -d "$EXP_DIR" ]] || die "Experiment already exists: $EXP_DIR"
    command -v conda >/dev/null 2>&1 || die "conda is not available"

    QUANT_GPU="$GPU" conda run --no-capture-output -n "$CONDA_ENV" python -c \
        "import os, torch; gpu=int(os.environ['QUANT_GPU']); assert torch.cuda.is_available(); assert gpu < torch.cuda.device_count(); print('GPU:', gpu)"
    if [[ "$RUN_AA" == "1" ]]; then
        conda run --no-capture-output -n "$CONDA_ENV" python -c \
            "from autoattack import AutoAttack; print('AutoAttack: OK')"
    fi
    MPLCONFIGDIR=/tmp/bitcons-mpl conda run --no-capture-output -n "$CONDA_ENV" \
        python -m unittest discover -s tests -q
}

mkdir -p "$LOG_DIR/status"
printf '%s\n' "$$" > "$LOG_DIR/launcher.pid"
printf 'tag\tconfig\ttransform\tadaptive_attack\texperiment_dir\n' > "$LOG_DIR/manifest.tsv"
printf '%s\t%s\t%s\t%s\t%s\n' \
    "$TAG" "$CONFIG" "round_8bit_only" "true" "$EXP_DIR" \
    >> "$LOG_DIR/manifest.tsv"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY RUN] No training or evaluation will be started."
    cat "$LOG_DIR/manifest.tsv"
    exit 0
fi

preflight
printf 'running\t%s\ttraining gpu=%s\n' "$(date '+%F %T')" "$GPU" > "$STATUS_FILE"
echo "[QUANTIZE] run_id=$RUN_ID gpu=$GPU aa=$RUN_AA"
echo "[TRAIN][GPU $GPU] $TAG"

CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/bitcons-mpl \
    conda run --no-capture-output -n "$CONDA_ENV" python src/train.py \
    --dataset "$DATASET" --data_dir "$DATA_DIR" --model "$MODEL" \
    --config "$CONFIG" --desc "$DESC" --epochs "$EPOCHS" --seed "$SEED" \
    --gpu_id 0 --num_workers "$NUM_WORKERS" --out_dir "$OUT_DIR" \
    --exp_name "${RUN_ID}_${TAG}" --perturbation none \
    --no-bitcons --no-bitcons_contrast > "$TRAIN_LOG" 2>&1

[[ -s "$EXP_DIR/checkpoints/final_model.pt" ]] \
    || die "$TAG did not produce final_model.pt"

printf 'running\t%s\tevaluating best gpu=%s\n' "$(date '+%F %T')" "$GPU" > "$STATUS_FILE"
eval_cmd=(
    conda run --no-capture-output -n "$CONDA_ENV" python src/eval.py
    --exp "$EXP_DIR" --ckpt best --gpu 0 --data-dir "$DATA_DIR"
    --batch-size "$EVAL_BATCH_SIZE" --num-workers "$NUM_WORKERS"
)
[[ "$RUN_AA" == "1" ]] && eval_cmd+=(--aa)
echo "[EVAL][GPU $GPU] $TAG best (adaptive BPDA)"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
    "${eval_cmd[@]}" > "$EVAL_LOG" 2>&1

[[ -s "$EXP_DIR/eval_results_best.txt" ]] \
    || die "$TAG best evaluation produced no result"
printf 'complete\t%s\tgpu=%s\n' "$(date '+%F %T')" "$GPU" > "$STATUS_FILE"
echo "[QUANTIZE] complete; logs: $LOG_DIR"
