#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "${RUNPOD_PROJECT_ROOT}"

: "${RUNPOD_API_KEY:?Export RUNPOD_API_KEY before running this script}"

readonly image="${RUNPOD_IMAGE:-ghcr.io/tascaenzo/llm-lab-cuda:latest}"
readonly gpu="${RUNPOD_GPU_ID:-NVIDIA GeForce RTX 4090}"
readonly pod_name="${RUNPOD_POD_NAME:-llm-lab-cuda-train}"
readonly volume_gb="${RUNPOD_VOLUME_GB:-40}"
readonly pod_environment="$(runpod_pod_environment_json)"

runpod_require_positive_integer RUNPOD_VOLUME_GB "${volume_gb}"

if ! command -v runpodctl >/dev/null 2>&1; then
    printf 'Install runpodctl first, then retry: https://docs.runpod.io/runpodctl/overview\n' >&2
    exit 2
fi

if ! runpodctl user >/dev/null; then
    printf 'RunPod authentication failed. Run ./deploy/runpod/setup.sh once, then retry.\n' >&2
    exit 2
fi

runpodctl pod create \
    --name "${pod_name}" \
    --gpu-id "${gpu}" \
    --gpu-count 1 \
    --image "${image}" \
    --container-disk-in-gb 30 \
    --volume-in-gb "${volume_gb}" \
    --volume-mount-path /workspace \
    --ports "22/tcp" \
    --ssh \
    --env "${pod_environment}"
