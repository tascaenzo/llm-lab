#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "${RUNPOD_PROJECT_ROOT}"

: "${RUNPOD_SSH_HOST:?Set RUNPOD_SSH_HOST to user@host}"
: "${RUNPOD_SSH_PORT:?Set RUNPOD_SSH_PORT to the RunPod SSH port}"

readonly remote_root="${RUNPOD_REMOTE_ROOT:-/workspace/llm-lab}"
readonly model_dir=artifacts/models/italiano-base-75m
readonly profile_dir=artifacts/benchmarks/runpod

mkdir -p "${model_dir}"
rsync --archive --compress --partial --append-verify --progress \
    -e "ssh -p ${RUNPOD_SSH_PORT}" \
    "${RUNPOD_SSH_HOST}:${remote_root}/${model_dir}/" \
    "${model_dir}/"

# The diagnostic profile is optional, so do not turn a normal checkpoint
# download into an error when it has never been requested for this Pod.
if ssh -p "${RUNPOD_SSH_PORT}" "${RUNPOD_SSH_HOST}" \
    "test -d '${remote_root}/${profile_dir}'"; then
    mkdir -p "${profile_dir}"
    rsync --archive --compress --partial --append-verify --progress \
        -e "ssh -p ${RUNPOD_SSH_PORT}" \
        "${RUNPOD_SSH_HOST}:${remote_root}/${profile_dir}/" \
        "${profile_dir}/"
fi
