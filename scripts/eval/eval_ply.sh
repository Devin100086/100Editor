#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DRY_RUN=0

usage() {
    cat <<'EOF'
Usage:
  bash scripts/eval/eval_ply.sh [--dry-run] <origin_gs_source.ply> <edited_gs_source.ply> <colmap_dir> <origin_prompt> <target_prompt> [use_original_resolution(0|1)]

Examples:
  bash scripts/eval/eval_ply.sh \
    /path/to/origin.ply \
    /path/to/edited.ply \
    /path/to/colmap \
    "a photo of a man" \
    "a photo of an iron man" \
    1
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

if [[ ${#ARGS[@]} -lt 5 ]]; then
    usage
    exit 1
fi

ORIGIN_GS_SOURCE="${ARGS[0]}"
EDITED_GS_SOURCE="${ARGS[1]}"
COLMAP_DIR="${ARGS[2]}"
CLIP_PROMPT_ORIGIN="${ARGS[3]}"
CLIP_PROMPT_TARGET="${ARGS[4]}"
USE_ORIGIN="${ARGS[5]:-1}"

log() {
    printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

if [[ ! -f "${ORIGIN_GS_SOURCE}" ]]; then
    log "ERROR: origin_gs_source does not exist: ${ORIGIN_GS_SOURCE}"
    exit 1
fi
if [[ ! -f "${EDITED_GS_SOURCE}" ]]; then
    log "ERROR: edited_gs_source does not exist: ${EDITED_GS_SOURCE}"
    exit 1
fi
if [[ ! -d "${COLMAP_DIR}" ]]; then
    log "ERROR: colmap_dir does not exist: ${COLMAP_DIR}"
    exit 1
fi
if [[ "${USE_ORIGIN}" != "0" && "${USE_ORIGIN}" != "1" ]]; then
    log "ERROR: use_original_resolution must be 0 or 1, got: ${USE_ORIGIN}"
    exit 1
fi

CMD=(python src/eval/metrics/eval_ply.py
    --clip_prompt_origin "${CLIP_PROMPT_ORIGIN}"
    --clip_prompt_target "${CLIP_PROMPT_TARGET}"
    --origin_gs_source "${ORIGIN_GS_SOURCE}"
    --edited_gs_source "${EDITED_GS_SOURCE}"
    --colmap_dir "${COLMAP_DIR}"
    --use_original_resolution "${USE_ORIGIN}"
)

log "Starting PLY CLIP evaluation"
log "origin_gs_source : ${ORIGIN_GS_SOURCE}"
log "edited_gs_source : ${EDITED_GS_SOURCE}"
log "colmap_dir       : ${COLMAP_DIR}"
log "origin_prompt    : ${CLIP_PROMPT_ORIGIN}"
log "target_prompt    : ${CLIP_PROMPT_TARGET}"
log "use_origin       : ${USE_ORIGIN}"
log "CMD: ${CMD[*]}"

if [[ "${DRY_RUN}" -eq 0 ]]; then
    "${CMD[@]}"
    log "PLY evaluation finished"
else
    log "Dry-run mode enabled: evaluation command was not executed."
fi
