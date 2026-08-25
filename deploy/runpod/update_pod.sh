#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "${RUNPOD_PROJECT_ROOT}"

: "${RUNPOD_API_KEY:?Set RUNPOD_API_KEY in deploy/runpod/.env before running this script}"
: "${RUNPOD_POD_ID:?Set RUNPOD_POD_ID in deploy/runpod/.env before running this script}"

readonly image="${RUNPOD_IMAGE:-ghcr.io/tascaenzo/llm-lab-cuda:latest}"
readonly pod_environment="$(runpod_pod_environment_json)"

if ! command -v runpodctl >/dev/null 2>&1; then
    printf 'Install runpodctl first, then retry: https://docs.runpod.io/runpodctl/overview\n' >&2
    exit 2
fi

if ! runpodctl user >/dev/null; then
    printf 'RunPod authentication failed. Run ./deploy/runpod/setup.sh once, then retry.\n' >&2
    exit 2
fi

printf 'RunPod pipeline: updating Pod %s from %s. Stop an active training session first.\n' \
    "${RUNPOD_POD_ID}" "${RUNPOD_CONFIG_FILE}" >&2
exec runpodctl pod update "${RUNPOD_POD_ID}" --image "${image}" --env "${pod_environment}"
