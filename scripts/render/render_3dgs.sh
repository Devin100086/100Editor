#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DRY_RUN=0

usage() {
    cat <<'EOF'
Usage:
  bash scripts/render/render_3dgs.sh [--dry-run] <scene_path> <model_path> [iteration] [skip_train(0|1)] [skip_test(0|1)]

Examples:
  bash scripts/render/render_3dgs.sh /path/to/scene /path/to/output/model
  bash scripts/render/render_3dgs.sh /path/to/scene /path/to/output/model 7000 0 1
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

SCENE_PATH="${ARGS[0]}"
MODEL_PATH="${ARGS[1]}"
ITERATION="${ARGS[2]:--1}"
SKIP_TRAIN="${ARGS[3]:-0}"
SKIP_TEST="${ARGS[4]:-0}"

log() {
    printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

if [[ ! -d "${SCENE_PATH}" ]]; then
    log "ERROR: scene_path does not exist: ${SCENE_PATH}"
    exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
    log "ERROR: model_path does not exist: ${MODEL_PATH}"
    exit 1
fi
if [[ "${SKIP_TRAIN}" != "0" && "${SKIP_TRAIN}" != "1" ]]; then
    log "ERROR: skip_train must be 0 or 1, got: ${SKIP_TRAIN}"
    exit 1
fi
if [[ "${SKIP_TEST}" != "0" && "${SKIP_TEST}" != "1" ]]; then
    log "ERROR: skip_test must be 0 or 1, got: ${SKIP_TEST}"
    exit 1
fi

CMD=(python src/trainer/origin/render.py
    -s "${SCENE_PATH}"
    -m "${MODEL_PATH}"
    --iteration "${ITERATION}"
)
if [[ "${SKIP_TRAIN}" == "1" ]]; then
    CMD+=(--skip_train)
fi
if [[ "${SKIP_TEST}" == "1" ]]; then
    CMD+=(--skip_test)
fi

log "Starting 3DGS render"
log "scene_path   : ${SCENE_PATH}"
log "model_path   : ${MODEL_PATH}"
log "iteration    : ${ITERATION}"
log "skip_train   : ${SKIP_TRAIN}"
log "skip_test    : ${SKIP_TEST}"
log "CMD: ${CMD[*]}"

if [[ "${DRY_RUN}" -eq 0 ]]; then
    "${CMD[@]}"
    log "3DGS render finished"
else
    log "Dry-run mode enabled: render command was not executed."
fi
