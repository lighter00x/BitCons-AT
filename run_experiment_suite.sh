#!/usr/bin/env bash
set -Eeuo pipefail

# Timeline: Phase 1 - original BitCons Base/Core/Full study (completed history).
# Purpose: compare the legacy masking-stream BitCons across PGD-AT, TRADES,
# MART, and RPAT, plus the PGD-AT 2^3 Mask-CE/Align/Contrast ablation. This
# experiment established that the original unconditional auxiliary objectives
# did not improve standard-input robustness and could cause robust collapse.
# Status: retained for exact reproduction; it is not the current paper method.
# Two workers run in parallel; each GPU runs one train/eval job at a time.

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
DESC="${DESC:-paper_suite_${RUN_ID}}"
STAGE="${STAGE:-all}"                  # all | train | eval
RUN_AA="${RUN_AA:-1}"                  # 1 includes AutoAttack
RUN_FINAL_EVAL="${RUN_FINAL_EVAL:-0}"  # 1 also evaluates final_model.pt
DRY_RUN="${DRY_RUN:-0}"                # 1 prints the matrix only
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
BITCONS_PLANES="${BITCONS_PLANES:-0 1 2}"
BITCONS_ALPHA="${BITCONS_ALPHA:-0.25}"
BITCONS_WARMUP="${BITCONS_WARMUP:-60}"
BITCONS_CONTRAST_LAM="${BITCONS_CONTRAST_LAM:-0.001}"

LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/suite_${RUN_ID}}"
STATUS_DIR="$LOG_DIR/status"
MANIFEST="$LOG_DIR/manifest.tsv"
SUMMARY="$LOG_DIR/summary.tsv"
RESULTS="$LOG_DIR/results.tsv"

read -r -a GPU_ARR <<< "$GPUS"

# category|tag|method|mask_ce|align|contrast
# Base/Core/Full for four host methods form the comparison experiment.
# The additional PGD-AT rows complete all 2^3 component combinations.
TASKS=(
    "comparison|pgd_at_base|pgd_at|0|0|0"
    "comparison|pgd_at_core|pgd_at|1|1|0"
    "comparison|pgd_at_full|pgd_at|1|1|1"
    "comparison|trades_base|trades|0|0|0"
    "comparison|trades_core|trades|1|1|0"
    "comparison|trades_full|trades|1|1|1"
    "comparison|mart_base|mart|0|0|0"
    "comparison|mart_core|mart|1|1|0"
    "comparison|mart_full|mart|1|1|1"
    "comparison|rpat_base|rpat|0|0|0"
    "comparison|rpat_core|rpat|1|1|0"
    "comparison|rpat_full|rpat|1|1|1"
    "ablation|pgd_at_mask_only|pgd_at|1|0|0"
    "ablation|pgd_at_align_only|pgd_at|0|1|0"
    "ablation|pgd_at_contrast_only|pgd_at|0|0|1"
    "ablation|pgd_at_mask_contrast|pgd_at|1|0|1"
    "ablation|pgd_at_align_contrast|pgd_at|0|1|1"
)

usage() {
    cat <<'EOF'
Usage: ./run_experiment_suite.sh

Default: CIFAR-10, ResNet18, seed 4243, 110 epochs, GPUs 0 and 1.
The suite runs 12 comparison jobs and completes the 8-row PGD-AT ablation.
By default, only the best checkpoint receives the complete evaluation.

Useful overrides:
  RUN_ID=my_run ./run_experiment_suite.sh
  STAGE=train ./run_experiment_suite.sh
  RUN_ID=my_run STAGE=eval ./run_experiment_suite.sh
  RUN_ID=my_run STAGE=eval RUN_FINAL_EVAL=1 ./run_experiment_suite.sh
  GPUS="2 3" RUN_ID=my_run ./run_experiment_suite.sh
  DRY_RUN=1 ./run_experiment_suite.sh

Reusing RUN_ID skips completed training and evaluation outputs. A partial
training directory without final_model.pt is rejected; use a new RUN_ID so
that metrics from separate attempts are never mixed.
EOF
}

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

experiment_dir() {
    local tag="$1"
    local method="$2"
    local mask_ce="$3"
    local align="$4"
    local contrast="$5"
    local bitcons="false"
    local contrast_name="false"

    if [[ "$mask_ce" == "1" || "$align" == "1" || "$contrast" == "1" ]]; then
        bitcons="true"
    fi
    if [[ "$contrast" == "1" ]]; then
        contrast_name="true"
    fi

    echo "$OUT_DIR/${DATASET}_${MODEL}_${method}_none_bitcons_${bitcons}_contrast_${contrast_name}_${DESC}/seed_${SEED}_${RUN_ID}_${tag}"
}

write_manifest() {
    printf 'category\ttag\tmethod\tmask_ce\talign\tcontrast\texperiment_dir\n' > "$MANIFEST"
    local task category tag method mask_ce align contrast exp_dir
    for task in "${TASKS[@]}"; do
        IFS='|' read -r category tag method mask_ce align contrast <<< "$task"
        exp_dir="$(experiment_dir "$tag" "$method" "$mask_ce" "$align" "$contrast")"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$category" "$tag" "$method" "$mask_ce" "$align" "$contrast" "$exp_dir" >> "$MANIFEST"
    done
}

preflight() {
    [[ "$STAGE" == "all" || "$STAGE" == "train" || "$STAGE" == "eval" ]] \
        || die "STAGE must be all, train, or eval (got: $STAGE)"
    [[ "$RUN_AA" == "0" || "$RUN_AA" == "1" ]] || die "RUN_AA must be 0 or 1"
    [[ "$RUN_FINAL_EVAL" == "0" || "$RUN_FINAL_EVAL" == "1" ]] \
        || die "RUN_FINAL_EVAL must be 0 or 1"
    [[ "$EPOCHS" -eq 110 ]] || die "This paper suite requires EPOCHS=110"
    [[ "${#GPU_ARR[@]}" -eq 2 ]] || die "Exactly two GPU IDs are required (got: $GPUS)"
    [[ "${GPU_ARR[0]}" != "${GPU_ARR[1]}" ]] || die "GPU IDs must be different"
    local gpu
    for gpu in "${GPU_ARR[@]}"; do
        [[ "$gpu" =~ ^[0-9]+$ ]] || die "GPU ID must be a non-negative integer (got: $gpu)"
    done
    [[ -d "$DATA_DIR" ]] || die "Dataset directory does not exist: $DATA_DIR"
    case "$DATASET" in
        cifar10)
            [[ -f "$DATA_DIR/cifar-10-batches-py/data_batch_1" ]] \
                || die "Local CIFAR-10 files are incomplete under: $DATA_DIR"
            ;;
        cifar100)
            [[ -f "$DATA_DIR/cifar-100-python/train" ]] \
                || die "Local CIFAR-100 files are incomplete under: $DATA_DIR"
            ;;
        tinynet)
            [[ -f "$DATA_DIR/tiny-imagenet-200/wnids.txt" ]] \
                || die "Local TinyImageNet files are incomplete under: $DATA_DIR"
            ;;
        *)
            die "Unsupported DATASET for this suite: $DATASET"
            ;;
    esac
    command -v conda >/dev/null 2>&1 || die "conda is not available"

    SUITE_GPU_IDS="$GPUS" conda run --no-capture-output -n "$CONDA_ENV" python -c \
        "import os, torch; ids=[int(v) for v in os.environ['SUITE_GPU_IDS'].split()]; count=torch.cuda.device_count(); assert torch.cuda.is_available(), 'CUDA is not available'; assert max(ids) < count, f'GPU IDs {ids} invalid for {count} visible GPUs'; print('PyTorch', torch.__version__, '| visible GPUs', count, '| selected', ids)"

    if [[ "$RUN_AA" == "1" ]]; then
        conda run --no-capture-output -n "$CONDA_ENV" python -c \
            "from autoattack import AutoAttack; print('AutoAttack import: OK')"
    fi

    MPLCONFIGDIR=/tmp/bitcons-mpl conda run --no-capture-output -n "$CONDA_ENV" \
        python -m unittest discover -s tests -q
}

record_status() {
    local tag="$1"
    local status="$2"
    local detail="$3"
    printf '%s\t%s\t%s\t%s\n' "$tag" "$status" "$(date '+%F %T')" "$detail" \
        > "$STATUS_DIR/${tag}.tsv"
}

run_eval() {
    local gpu="$1"
    local tag="$2"
    local exp_dir="$3"
    local ckpt="$4"
    local result_file="$exp_dir/eval_results_${ckpt}.txt"
    local log_file="$LOG_DIR/${tag}_eval_${ckpt}_gpu${gpu}.log"

    if [[ -s "$result_file" ]]; then
        echo "[SKIP][GPU $gpu] $tag $ckpt evaluation already complete"
        return 0
    fi

    local eval_cmd=(
        conda run --no-capture-output -n "$CONDA_ENV" python src/eval.py
        --exp "$exp_dir"
        --ckpt "$ckpt"
        --gpu 0
        --data-dir "$DATA_DIR"
        --batch-size "$EVAL_BATCH_SIZE"
        --num-workers "$NUM_WORKERS"
    )
    if [[ "$RUN_AA" == "1" ]]; then
        eval_cmd+=(--aa)
    fi

    echo "[EVAL][GPU $gpu] $tag checkpoint=$ckpt"
    if ! CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
        "${eval_cmd[@]}" > "$log_file" 2>&1; then
        echo "[FAIL][GPU $gpu] $tag $ckpt evaluation; log: $log_file" >&2
        return 1
    fi
    if [[ ! -s "$result_file" ]]; then
        echo "[FAIL][GPU $gpu] $tag $ckpt produced no result file; log: $log_file" >&2
        return 1
    fi
}

run_task() {
    local gpu="$1"
    local task="$2"
    local category tag method mask_ce align contrast
    IFS='|' read -r category tag method mask_ce align contrast <<< "$task"

    local exp_dir
    exp_dir="$(experiment_dir "$tag" "$method" "$mask_ce" "$align" "$contrast")"
    local final_ckpt="$exp_dir/checkpoints/final_model.pt"
    local train_log="$LOG_DIR/${tag}_train_gpu${gpu}.log"
    local bitcons=0
    if [[ "$mask_ce" == "1" || "$align" == "1" || "$contrast" == "1" ]]; then
        bitcons=1
    fi

    echo "[TASK][GPU $gpu] $tag ($category, method=$method, modules=$mask_ce/$align/$contrast)"
    record_status "$tag" "running" "gpu=$gpu"

    if [[ "$STAGE" != "eval" ]]; then
        if [[ -s "$final_ckpt" ]]; then
            echo "[SKIP][GPU $gpu] $tag training already complete"
        else
            if [[ -d "$exp_dir" ]]; then
                echo "[FAIL][GPU $gpu] Partial experiment exists: $exp_dir" >&2
                echo "       Use a new RUN_ID; partial metrics must not be mixed." >&2
                record_status "$tag" "failed" "partial training directory"
                return 1
            fi

            local train_cmd=(
                conda run --no-capture-output -n "$CONDA_ENV" python src/train.py
                --dataset "$DATASET"
                --data_dir "$DATA_DIR"
                --model "$MODEL"
                --config "$method"
                --desc "$DESC"
                --epochs "$EPOCHS"
                --seed "$SEED"
                --gpu_id 0
                --num_workers "$NUM_WORKERS"
                --out_dir "$OUT_DIR"
                --exp_name "${RUN_ID}_${tag}"
                --perturbation none
            )

            if [[ "$bitcons" == "1" ]]; then
                read -r -a bitcons_plane_arr <<< "$BITCONS_PLANES"
                train_cmd+=(
                    --bitcons
                    --bitcons_planes "${bitcons_plane_arr[@]}"
                    --bitcons_alpha "$BITCONS_ALPHA"
                    --bitcons_warmup "$BITCONS_WARMUP"
                    --bitcons_contrast_lam "$BITCONS_CONTRAST_LAM"
                    --bitcons_ce_weight "$mask_ce"
                    --bitcons_align_weight "$align"
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
            if ! CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/bitcons-mpl \
                "${train_cmd[@]}" > "$train_log" 2>&1; then
                echo "[FAIL][GPU $gpu] $tag training; log: $train_log" >&2
                record_status "$tag" "failed" "training"
                return 1
            fi
            if [[ ! -s "$final_ckpt" ]]; then
                echo "[FAIL][GPU $gpu] $tag produced no final checkpoint; log: $train_log" >&2
                record_status "$tag" "failed" "missing final checkpoint"
                return 1
            fi
        fi
    elif [[ ! -s "$final_ckpt" ]]; then
        echo "[FAIL][GPU $gpu] Cannot evaluate $tag; checkpoint missing: $final_ckpt" >&2
        record_status "$tag" "failed" "checkpoint missing"
        return 1
    fi

    if [[ "$STAGE" != "train" ]]; then
        if ! run_eval "$gpu" "$tag" "$exp_dir" best; then
            record_status "$tag" "failed" "best evaluation"
            return 1
        fi
        if [[ "$RUN_FINAL_EVAL" == "1" ]]; then
            if ! run_eval "$gpu" "$tag" "$exp_dir" final; then
                record_status "$tag" "failed" "final evaluation"
                return 1
            fi
        fi
    fi

    record_status "$tag" "complete" "gpu=$gpu"
    echo "[DONE][GPU $gpu] $tag"
}

worker() {
    local worker_index="$1"
    local gpu="${GPU_ARR[$worker_index]}"
    local failed=0
    local i

    for ((i=worker_index; i<${#TASKS[@]}; i+=${#GPU_ARR[@]})); do
        if ! run_task "$gpu" "${TASKS[$i]}"; then
            failed=1
        fi
    done
    return "$failed"
}

build_summary() {
    printf 'tag\tstatus\ttimestamp\tdetail\n' > "$SUMMARY"
    local task category tag rest status_file
    for task in "${TASKS[@]}"; do
        IFS='|' read -r category tag rest <<< "$task"
        status_file="$STATUS_DIR/${tag}.tsv"
        if [[ -f "$status_file" ]]; then
            cat "$status_file" >> "$SUMMARY"
        else
            printf '%s\t%s\t%s\t%s\n' "$tag" "not_started" "-" "-" >> "$SUMMARY"
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
    printf 'category\ttag\tmethod\tmask_ce\talign\tcontrast\tcheckpoint\tnatural\tpgd10\tpgd20\tpgd50\tcw\taa\texperiment_dir\n' > "$RESULTS"
    local task category tag method mask_ce align contrast exp_dir ckpt result_file
    local natural pgd10 pgd20 pgd50 cw aa
    for task in "${TASKS[@]}"; do
        IFS='|' read -r category tag method mask_ce align contrast <<< "$task"
        exp_dir="$(experiment_dir "$tag" "$method" "$mask_ce" "$align" "$contrast")"
        for ckpt in best final; do
            result_file="$exp_dir/eval_results_${ckpt}.txt"
            if [[ ! -s "$result_file" ]]; then
                continue
            fi
            natural="$(metric_from_result "$result_file" 'Natural Accuracy')"
            pgd10="$(metric_from_result "$result_file" 'PGD-10 Accuracy')"
            pgd20="$(metric_from_result "$result_file" 'PGD-20 Accuracy')"
            pgd50="$(metric_from_result "$result_file" 'PGD-50 Accuracy')"
            cw="$(metric_from_result "$result_file" 'C&W Accuracy')"
            aa="$(metric_from_result "$result_file" 'AutoAttack Accuracy')"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$category" "$tag" "$method" "$mask_ce" "$align" "$contrast" \
                "$ckpt" "$natural" "$pgd10" "$pgd20" "$pgd50" "$cw" "$aa" "$exp_dir" \
                >> "$RESULTS"
        done
    done
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi
[[ "$#" -eq 0 ]] || die "Unknown argument: $1 (use --help)"

mkdir -p "$LOG_DIR" "$STATUS_DIR"
write_manifest

echo "[SUITE] run_id=$RUN_ID dataset=$DATASET model=$MODEL seed=$SEED epochs=$EPOCHS"
echo "[SUITE] stage=$STAGE gpus=$GPUS conda_env=$CONDA_ENV tasks=${#TASKS[@]} final_eval=$RUN_FINAL_EVAL"
echo "[SUITE] logs=$LOG_DIR"

if [[ "$DRY_RUN" == "1" ]]; then
    column -t -s $'\t' "$MANIFEST" 2>/dev/null || cat "$MANIFEST"
    echo "[DRY RUN] No training or evaluation was started."
    exit 0
fi

preflight

pids=()
for ((worker_index=0; worker_index<${#GPU_ARR[@]}; worker_index++)); do
    worker "$worker_index" &
    pids+=("$!")
done

failed=0
set +e
for pid in "${pids[@]}"; do
    wait "$pid"
    status=$?
    if [[ "$status" -ne 0 ]]; then
        failed=1
    fi
done
set -e

build_summary
build_results
if [[ "$failed" -ne 0 ]]; then
    echo "[SUMMARY] Suite finished with failures: $SUMMARY" >&2
    echo "[SUMMARY] Available results: $RESULTS" >&2
    exit 1
fi

echo "[SUMMARY] All ${#TASKS[@]} tasks completed successfully."
echo "[SUMMARY] Manifest: $MANIFEST"
echo "[SUMMARY] Status:   $SUMMARY"
echo "[SUMMARY] Results:  $RESULTS"
