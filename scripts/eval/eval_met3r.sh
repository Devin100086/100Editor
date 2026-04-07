#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DRY_RUN=0
USE_CPU=0
VERBOSE_PAIRS=0

usage() {
    cat <<'EOF'
Usage:
  bash scripts/eval/eval_met3r.sh [--dry-run] [--cpu] [--verbose-pairs] <image_dir> [distance] [img_size]

Examples:
  bash scripts/eval/eval_met3r.sh /path/to/rendered_origin cosine 256
  bash scripts/eval/eval_met3r.sh --cpu /path/to/rendered_origin ssim 0
EOF
}

ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --cpu)
            USE_CPU=1
            shift
            ;;
        --verbose-pairs)
            VERBOSE_PAIRS=1
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

if [[ ${#ARGS[@]} -lt 1 ]]; then
    usage
    exit 1
fi

IMAGE_DIR="${ARGS[0]}"
DISTANCE="${ARGS[1]:-cosine}"
IMG_SIZE="${ARGS[2]:-256}"

log() {
    printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

if [[ ! -d "${IMAGE_DIR}" ]]; then
    log "ERROR: image_dir does not exist: ${IMAGE_DIR}"
    exit 1
fi

CMD=(python src/eval/metrics/eval_met3r.py
    --image-dir "${IMAGE_DIR}"
    --distance "${DISTANCE}"
    --img-size "${IMG_SIZE}"
)
if [[ "${USE_CPU}" -eq 1 ]]; then
    CMD+=(--cpu)
fi
if [[ "${VERBOSE_PAIRS}" -eq 1 ]]; then
    CMD+=(--verbose-pairs)
fi

log "Starting MEt3R evaluation"
log "image_dir      : ${IMAGE_DIR}"
log "distance       : ${DISTANCE}"
log "img_size       : ${IMG_SIZE}"
log "cpu            : ${USE_CPU}"
log "verbose_pairs  : ${VERBOSE_PAIRS}"
log "CMD: ${CMD[*]}"

if [[ "${DRY_RUN}" -eq 0 ]]; then
    "${CMD[@]}"
    log "MEt3R evaluation finished"
else
    log "Dry-run mode enabled: evaluation command was not executed."
fi
