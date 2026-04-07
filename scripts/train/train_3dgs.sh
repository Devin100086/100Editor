#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DRY_RUN=0
USE_EVAL=1

usage() {
    cat <<'EOF'
Usage:
  bash scripts/train/train_3dgs.sh [--dry-run] [--no-eval] [scene_path] [output_dir] [checkpoint_iter]

Examples:
  bash scripts/train/train_3dgs.sh
  bash scripts/train/train_3dgs.sh /data/scene output/scene 10000
  bash scripts/train/train_3dgs.sh --dry-run /data/scene output/scene
EOF
}

ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --no-eval)
            USE_EVAL=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

SCENE_PATH="${ARGS[0]:-/home/wucunqi/Desktop/datasets/yuseung}"
OUTPUT_DIR="${ARGS[1]:-output/yuseung}"
CHECKPOINT_ITER="${ARGS[2]:-7000}"

log() {
    printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

if [[ ! -d "${SCENE_PATH}" ]]; then
    log "ERROR: scene_path does not exist: ${SCENE_PATH}"
    exit 1
fi

CMD=(python src/trainer/origin/train.py
    -s "${SCENE_PATH}"
    --checkpoint_iterations "${CHECKPOINT_ITER}"
    -m "${OUTPUT_DIR}"
)
if [[ "${USE_EVAL}" -eq 1 ]]; then
    CMD+=(--eval)
fi

log "Starting 3DGS training"
log "scene_path      : ${SCENE_PATH}"
log "output_dir      : ${OUTPUT_DIR}"
log "checkpoint_iter : ${CHECKPOINT_ITER}"
log "eval            : ${USE_EVAL}"
log "CMD: ${CMD[*]}"

if [[ "${DRY_RUN}" -eq 0 ]]; then
    "${CMD[@]}"
    log "Training finished"
else
    log "Dry-run mode enabled: training command was not executed."
fi
