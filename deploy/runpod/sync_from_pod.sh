#!/usr/bin/env bash
set -euo pipefail

: "${RUNPOD_SSH_HOST:?Set RUNPOD_SSH_HOST to user@host}"
: "${RUNPOD_SSH_PORT:?Set RUNPOD_SSH_PORT to the RunPod SSH port}"

readonly remote_root="${RUNPOD_REMOTE_ROOT:-/workspace/llm-lab}"
readonly model_dir=artifacts/models/italiano-base-75m

mkdir -p "${model_dir}"
rsync --archive --compress --partial --append-verify --progress \
    -e "ssh -p ${RUNPOD_SSH_PORT}" \
    "${RUNPOD_SSH_HOST}:${remote_root}/${model_dir}/" \
    "${model_dir}/"
