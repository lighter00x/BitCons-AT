#!/usr/bin/env bash
set -Eeuo pipefail

# Timeline: Phase 5 - current RA-WC-BitCons Stage-1 screen (recommended).
# Purpose: determine whether risk-gated BitCons adds robust accuracy beyond both
# corrected PGD-AT and the BitMax-only inner optimizer. Only a stable gain over
# BitMax-only demonstrates an independent BitCons contribution.
# Status: current recommended launcher; run ID rawc_stage1_s4243_20260831_175427
# was started on 2026-08-31. It performs ordinary differentiable inference and
# the full Clean/PGD-10/20/50/C&W/AutoAttack evaluation on best checkpoints.
#
# Two-GPU scheduling:
# GPU 0: BitMax-only -> corrected PGD base
# GPU 1: RA-WC-BitCons

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
DESC="${DESC:-rawc_bitcons_stage1_${RUN_ID}}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/rawc_bitcons_stage1_${RUN_ID}}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
DRY_RUN="${DRY_RUN:-0}"
STAGE="${STAGE:-all}"

STATUS_DIR="$LOG_DIR/status"
MANIFEST="$LOG_DIR/manifest.tsv"
SUMMARY="$LOG_DIR/status.tsv"
RESULTS="$LOG_DIR/results.tsv"

# Ordering is intentional. Strided assignment gives worker 0 the two tasks
# at indices 0 and 2, while worker 1 runs the main method at index 1.
# tag|config|bitcons
TASKS=(
    "bitmax_only|bitcons_at|false"
    "rawc_bitcons|bitcons_at|true"
    "corrected_pgd_base|pgd_at|false"
)

read -r -a GPU_ARR <<< "$GPUS"

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage: ./run_rawc_bitcons_stage1_dual.sh

Environment overrides:
  GPUS="0 1"          Exactly two physical GPU IDs
  RUN_ID=<id>         Stable ID used for resume/evaluation
  STAGE=all           all | train | eval
  DRY_RUN=1           Print the manifest without GPU work

Examples:
  DRY_RUN=1 ./run_rawc_bitcons_stage1_dual.sh
  RUN_ID=rawc_s4243 ./run_rawc_bitcons_stage1_dual.sh
  RUN_ID=rawc_s4243 STAGE=eval ./run_rawc_bitcons_stage1_dual.sh
EOF
}

experiment_dir() {
    local tag="$1"
    local config="$2"
    local bitcons="$3"
    echo "$OUT_DIR/${DATASET}_${MODEL}_${config}_none_bitcons_${bitcons}_contrast_false_${DESC}/seed_${SEED}_${RUN_ID}_${tag}"
}

record_status() {
    local tag="$1"
    local status="$2"
    local detail="$3"
    printf '%s\t%s\t%s\t%s\n' \
        "$tag" "$status" "$(date '+%F %T')" "$detail" \
        > "$STATUS_DIR/${tag}.tsv"
}

write_manifest() {
    printf 'tag\tconfig\tbitcons\texperiment_dir\n' > "$MANIFEST"
    local task tag config bitcons exp_dir
    for task in "${TASKS[@]}"; do
        IFS='|' read -r tag config bitcons <<< "$task"
        exp_dir="$(experiment_dir "$tag" "$config" "$bitcons")"
        printf '%s\t%s\t%s\t%s\n' \
            "$tag" "$config" "$bitcons" "$exp_dir" >> "$MANIFEST"
    done
}

preflight() {
    [[ "$EPOCHS" -eq 110 ]] || die "Stage 1 requires EPOCHS=110"
    [[ "$STAGE" == "all" || "$STAGE" == "train" || "$STAGE" == "eval" ]] \
        || die "STAGE must be all, train, or eval"
    [[ "${#GPU_ARR[@]}" -eq 2 ]] || die "Exactly two GPU IDs are required"
    [[ "${GPU_ARR[0]}" != "${GPU_ARR[1]}" ]] || die "GPU IDs must differ"
    [[ -f "$DATA_DIR/cifar-10-batches-py/data_batch_1" ]] \
        || die "Local CIFAR-10 data is incomplete under $DATA_DIR"
    command -v conda >/dev/null 2>&1 || die "conda is not available"

    RAWC_GPU_IDS="$GPUS" conda run --no-capture-output -n "$CONDA_ENV" \
        python -c \
        "import os, torch; ids=[int(v) for v in os.environ['RAWC_GPU_IDS'].split()]; assert torch.cuda.is_available(); assert max(ids) < torch.cuda.device_count(); print('GPUs:', ids)"
    conda run --no-capture-output -n "$CONDA_ENV" python -c \
        "from autoattack import AutoAttack; print('AutoAttack: OK')"
    MPLCONFIGDIR=/tmp/bitcons-mpl conda run --no-capture-output -n "$CONDA_ENV" \
        python -m unittest discover -s tests -q
}

run_eval() {
    local gpu="$1"
    local tag="$2"
    local exp_dir="$3"
    local result_file="$exp_dir/eval_results_best.txt"
    local eval_log="$LOG_DIR/${tag}_eval_best_gpu${gpu}.log"

    if [[ -s "$result_file" ]]; then
        echo "[SKIP][GPU $gpu] $tag best evaluation already complete"
        return 0
    fi

    record_status "$tag" "running" "evaluating best gpu=$gpu"
    local eval_cmd=(
        conda run --no-capture-output -n "$CONDA_ENV" python src/eval.py
        --exp "$exp_dir" --ckpt best --gpu 0 --data-dir "$DATA_DIR"
        --batch-size "$EVAL_BATCH_SIZE" --num-workers "$NUM_WORKERS"
        --all-attacks
    )

    echo "[EVAL][GPU $gpu] $tag best (full attack suite)"
    if ! CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
        "${eval_cmd[@]}" > "$eval_log" 2>&1; then
        echo "[FAIL][GPU $gpu] $tag evaluation; log: $eval_log" >&2
        return 1
    fi
    [[ -s "$result_file" ]] || {
        echo "[FAIL][GPU $gpu] $tag produced no evaluation result" >&2
        return 1
    }
}

run_task() {
    local gpu="$1"
    local task="$2"
    local tag config bitcons exp_dir final_ckpt train_log
    IFS='|' read -r tag config bitcons <<< "$task"
    exp_dir="$(experiment_dir "$tag" "$config" "$bitcons")"
    final_ckpt="$exp_dir/checkpoints/final_model.pt"
    train_log="$LOG_DIR/${tag}_train_gpu${gpu}.log"

    echo "[TASK][GPU $gpu] $tag config=$config bitcons=$bitcons"
    record_status "$tag" "running" "gpu=$gpu"

    if [[ "$STAGE" != "eval" ]]; then
        if [[ -s "$final_ckpt" ]]; then
            echo "[SKIP][GPU $gpu] $tag training already complete"
        else
            if [[ -d "$exp_dir" ]]; then
                echo "[FAIL][GPU $gpu] Partial experiment exists: $exp_dir" >&2
                record_status "$tag" "failed" "partial training directory"
                return 1
            fi

            local train_cmd=(
                conda run --no-capture-output -n "$CONDA_ENV" python src/train.py
                --dataset "$DATASET" --data_dir "$DATA_DIR" --model "$MODEL"
                --config "$config" --desc "$DESC" --epochs "$EPOCHS"
                --seed "$SEED" --gpu_id 0 --num_workers "$NUM_WORKERS"
                --out_dir "$OUT_DIR" --exp_name "${RUN_ID}_${tag}"
                --perturbation none --no-bitcons_contrast
            )
            if [[ "$bitcons" == "true" ]]; then
                train_cmd+=(--bitcons)
            else
                train_cmd+=(--no-bitcons)
            fi

            echo "[TRAIN][GPU $gpu] $tag"
            if ! CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
                MPLCONFIGDIR=/tmp/bitcons-mpl \
                "${train_cmd[@]}" > "$train_log" 2>&1; then
                echo "[FAIL][GPU $gpu] $tag training; log: $train_log" >&2
                record_status "$tag" "failed" "training"
                return 1
            fi
            if [[ ! -s "$final_ckpt" ]]; then
                echo "[FAIL][GPU $gpu] $tag produced no final checkpoint" >&2
                record_status "$tag" "failed" "missing final checkpoint"
                return 1
            fi
        fi
    elif [[ ! -s "$final_ckpt" ]]; then
        echo "[FAIL][GPU $gpu] Cannot evaluate $tag; checkpoint missing" >&2
        record_status "$tag" "failed" "checkpoint missing"
        return 1
    fi

    if [[ "$STAGE" != "train" ]]; then
        if ! run_eval "$gpu" "$tag" "$exp_dir"; then
            record_status "$tag" "failed" "best evaluation"
            return 1
        fi
    fi

    record_status "$tag" "complete" "gpu=$gpu"
    echo "[DONE][GPU $gpu] $tag"
}

worker() {
    local worker_index="$1"
    local gpu="${GPU_ARR[$worker_index]}"
    local failed=0
    local index
    for ((index=worker_index; index<${#TASKS[@]}; index+=${#GPU_ARR[@]})); do
        run_task "$gpu" "${TASKS[$index]}" || failed=1
    done
    return "$failed"
}

build_summary() {
    printf 'tag\tstatus\ttimestamp\tdetail\n' > "$SUMMARY"
    local task tag rest status_file
    for task in "${TASKS[@]}"; do
        IFS='|' read -r tag rest <<< "$task"
        status_file="$STATUS_DIR/${tag}.tsv"
        if [[ -f "$status_file" ]]; then
            cat "$status_file" >> "$SUMMARY"
        else
            printf '%s\tnot_started\t-\t-\n' "$tag" >> "$SUMMARY"
        fi
    done
}

metric_from_result() {
    local result_file="$1"
    local metric="$2"
    local value
    value="$(awk -F ':' -v metric="$metric" '
        index($1, metric) {
            value=$2
            gsub(/^[[:space:]]+|[[:space:]%]+$/, "", value)
            print value
            exit
        }
    ' "$result_file")"
    echo "${value:-NA}"
}

build_results() {
    printf 'tag\tconfig\tbitcons\tnatural\tpgd10\tpgd20\tpgd50\tcw\taa\texperiment_dir\n' \
        > "$RESULTS"
    local task tag config bitcons exp_dir result_file
    local natural pgd10 pgd20 pgd50 cw aa
    for task in "${TASKS[@]}"; do
        IFS='|' read -r tag config bitcons <<< "$task"
        exp_dir="$(experiment_dir "$tag" "$config" "$bitcons")"
        result_file="$exp_dir/eval_results_best.txt"
        [[ -s "$result_file" ]] || continue
        natural="$(metric_from_result "$result_file" 'Natural Accuracy')"
        pgd10="$(metric_from_result "$result_file" 'PGD-10 Accuracy')"
        pgd20="$(metric_from_result "$result_file" 'PGD-20 Accuracy')"
        pgd50="$(metric_from_result "$result_file" 'PGD-50 Accuracy')"
        cw="$(metric_from_result "$result_file" 'C&W Accuracy')"
        aa="$(metric_from_result "$result_file" 'AutoAttack Accuracy')"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$tag" "$config" "$bitcons" "$natural" "$pgd10" "$pgd20" \
            "$pgd50" "$cw" "$aa" "$exp_dir" >> "$RESULTS"
    done
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi
[[ "$#" -eq 0 ]] || die "Unknown argument: $1 (use --help)"

mkdir -p "$LOG_DIR" "$STATUS_DIR"
write_manifest

echo "[RA-WC-BITCONS] run_id=$RUN_ID stage=$STAGE gpus=$GPUS tasks=${#TASKS[@]}"
echo "[RA-WC-BITCONS] logs=$LOG_DIR"
if [[ "$DRY_RUN" == "1" ]]; then
    column -t -s $'\t' "$MANIFEST" 2>/dev/null || cat "$MANIFEST"
    echo "[DRY RUN] No training or evaluation was started."
    exit 0
fi

preflight
pids=()
for worker_index in "${!GPU_ARR[@]}"; do
    worker "$worker_index" &
    pids+=("$!")
done

failed=0
set +e
for pid in "${pids[@]}"; do
    wait "$pid"
    [[ "$?" -eq 0 ]] || failed=1
done
set -e

build_summary
build_results
if [[ "$failed" -ne 0 ]]; then
    echo "[SUMMARY] Stage 1 finished with failures: $SUMMARY" >&2
    exit 1
fi

echo "[SUMMARY] All tasks completed successfully."
echo "[SUMMARY] Status:  $SUMMARY"
echo "[SUMMARY] Results: $RESULTS"
