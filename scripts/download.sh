#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNTIME_CACHE_DIR="${REPO_ROOT}/runtime/.cache"
SAM2_DIR="${RUNTIME_CACHE_DIR}/sam2"
DPT_DIR="${RUNTIME_CACHE_DIR}/dpt"
SAM2_FILE="${SAM2_DIR}/sam2.1_hiera_large.pt"
DPT_FILE="${DPT_DIR}/omnidata_dpt_depth_v2.ckpt"

DRY_RUN=0

usage() {
    cat <<'EOF'
Usage:
  bash scripts/download.sh [--dry-run]

Description:
  Download required runtime checkpoints/models into runtime/.cache.
EOF
}

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -ne 0 ]]; then
    usage
    exit 1
fi

log() {
    printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

require_cmd() {
    if ! command -v "$1" &> /dev/null; then
        log "ERROR: command not found: $1"
        exit 1
    fi
}

run() {
    log "CMD: $*"
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        "$@"
    fi
}

log "Preparing runtime cache under: ${RUNTIME_CACHE_DIR}"
mkdir -p "${SAM2_DIR}" "${DPT_DIR}"

log "Step 1/4: checking required commands"
if [[ "${DRY_RUN}" -eq 0 ]]; then
    require_cmd wget
    require_cmd huggingface-cli
else
    log "Dry-run mode: skip command availability checks."
fi

log "Step 2/4: downloading SAM2 checkpoint"
run wget -O "${SAM2_FILE}" \
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"

log "Step 3/4: downloading DPT checkpoint"
run wget -O "${DPT_FILE}" \
    "https://huggingface.co/thegenerativegeneration/omnidata/resolve/main/omnidata_dpt_depth_v2.ckpt?download=true"

log "Step 4/4: downloading BrushEdit + LeftRefill assets"
run huggingface-cli download TencentARC/BrushEdit \
    --include "brushnetX/*" \
    --local-dir "${RUNTIME_CACHE_DIR}"
run huggingface-cli download TencentARC/BrushEdit \
    --include "base_model/*" \
    --local-dir "${RUNTIME_CACHE_DIR}/brushnetX"
run huggingface-cli download Devin100086/100Editor \
    --include "LeftRefill/*" \
    --local-dir "${RUNTIME_CACHE_DIR}"

log "Done. Downloaded assets are available in: ${RUNTIME_CACHE_DIR}"
if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "Dry-run mode enabled: no files were actually downloaded."
fi
