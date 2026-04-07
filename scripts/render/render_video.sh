#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DRY_RUN=0

usage() {
    cat <<'EOF'
Usage:
  bash scripts/render/render_video.sh [--dry-run] <gs_source.ply> <colmap_dir> [save_video.mp4] [use_original_resolution(0|1)] [render_path_mode(0|1)]

Notes:
  - render_path_mode=1 will pass "--render_path True" to Python.
  - render_path_mode=0 keeps default spiral trajectory.
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

if [[ ${#ARGS[@]} -lt 2 ]]; then
    usage
    exit 1
fi

GS_SOURCE="${ARGS[0]}"
COLMAP_DIR="${ARGS[1]}"
SAVE_PATH="${ARGS[2]:-${REPO_ROOT}/runtime/render/output.mp4}"
USE_ORIGIN="${ARGS[3]:-0}"
RENDER_PATH_MODE="${ARGS[4]:-0}"

log() {
    printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

if [[ ! -f "${GS_SOURCE}" ]]; then
    log "ERROR: gs_source file does not exist: ${GS_SOURCE}"
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
if [[ "${RENDER_PATH_MODE}" != "0" && "${RENDER_PATH_MODE}" != "1" ]]; then
    log "ERROR: render_path_mode must be 0 or 1, got: ${RENDER_PATH_MODE}"
    exit 1
fi

mkdir -p "$(dirname "${SAVE_PATH}")"

CMD=(python src/eval/renderers/render_video.py
    --gs_source "${GS_SOURCE}"
    --colmap_dir "${COLMAP_DIR}"
    --save_dir "${SAVE_PATH}"
    --use_original_resolution "${USE_ORIGIN}"
)
# render_video.py currently defines --render_path with type=bool,
# so we only pass it when explicitly enabled.
if [[ "${RENDER_PATH_MODE}" == "1" ]]; then
    CMD+=(--render_path True)
fi

log "Starting video rendering"
log "gs_source       : ${GS_SOURCE}"
log "colmap_dir      : ${COLMAP_DIR}"
log "save_video      : ${SAVE_PATH}"
log "use_origin      : ${USE_ORIGIN}"
log "render_path     : ${RENDER_PATH_MODE}"
log "CMD: ${CMD[*]}"

if [[ "${DRY_RUN}" -eq 0 ]]; then
    "${CMD[@]}"
    log "Video rendering finished"
else
    log "Dry-run mode enabled: render command was not executed."
fi
