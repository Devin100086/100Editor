#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DRY_RUN=0

usage() {
    cat <<'EOF'
Usage:
  bash scripts/eval/eval_image.sh [--dry-run] <origin_image_dir> <edited_image_dir> <origin_prompt> <target_prompt>

Examples:
  bash scripts/eval/eval_image.sh \
    /path/to/rendered_origin \
    /path/to/rendered_edited \
    "a photo of an outdoor garden" \
    "a photo of an outdoor garden in winter"
EOF
}

ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
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

if [[ ${#ARGS[@]} -lt 4 ]]; then
    usage
    exit 1
fi

ORIGIN_IMAGE_DIR="${ARGS[0]}"
EDITED_IMAGE_DIR="${ARGS[1]}"
CLIP_PROMPT_ORIGIN="${ARGS[2]}"
CLIP_PROMPT_TARGET="${ARGS[3]}"

log() {
    printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

if [[ ! -d "${ORIGIN_IMAGE_DIR}" ]]; then
    log "ERROR: origin_image_dir does not exist: ${ORIGIN_IMAGE_DIR}"
    exit 1
fi
if [[ ! -d "${EDITED_IMAGE_DIR}" ]]; then
    log "ERROR: edited_image_dir does not exist: ${EDITED_IMAGE_DIR}"
    exit 1
fi

CMD=(python src/eval/metrics/eval_image.py
    --clip_prompt_origin "${CLIP_PROMPT_ORIGIN}"
    --clip_prompt_target "${CLIP_PROMPT_TARGET}"
    --origin_image_dir "${ORIGIN_IMAGE_DIR}"
    --edited_image_dir "${EDITED_IMAGE_DIR}"
)

log "Starting image evaluation"
log "origin_dir    : ${ORIGIN_IMAGE_DIR}"
log "edited_dir    : ${EDITED_IMAGE_DIR}"
log "origin_prompt : ${CLIP_PROMPT_ORIGIN}"
log "target_prompt : ${CLIP_PROMPT_TARGET}"
log "CMD: ${CMD[*]}"

if [[ "${DRY_RUN}" -eq 0 ]]; then
    "${CMD[@]}"
    log "Evaluation finished"
else
    log "Dry-run mode enabled: evaluation command was not executed."
fi
