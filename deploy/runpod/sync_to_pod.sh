#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "${RUNPOD_PROJECT_ROOT}"

: "${RUNPOD_SSH_HOST:?Set RUNPOD_SSH_HOST to user@host}"
: "${RUNPOD_SSH_PORT:?Set RUNPOD_SSH_PORT to the RunPod SSH port}"

readonly remote_root="${RUNPOD_REMOTE_ROOT:-/workspace/llm-lab}"
readonly model_dir=artifacts/models/italiano-base-75m
readonly dataset_dir=data/derived/italiano-v3/lm
readonly ssh_identity_file="${RUNPOD_SSH_IDENTITY_FILE:-${HOME}/.runpod/ssh/runpodctl-ssh-key}"

if [[ ! -f "${ssh_identity_file}" ]]; then
    printf 'RunPod pipeline: SSH identity file does not exist: %s\n' "${ssh_identity_file}" >&2
    printf 'Run ./deploy/runpod/setup.sh, or set RUNPOD_SSH_IDENTITY_FILE.\n' >&2
    exit 2
fi

readonly ssh_transport="$(printf 'ssh -i %q -o IdentitiesOnly=yes -p %q' "${ssh_identity_file}" "${RUNPOD_SSH_PORT}")"
readonly ssh_options=(-i "${ssh_identity_file}" -o IdentitiesOnly=yes -p "${RUNPOD_SSH_PORT}")

ssh "${ssh_options[@]}" "${RUNPOD_SSH_HOST}" \
    "mkdir -p '${remote_root}/${dataset_dir}' '${remote_root}/${model_dir}'"

runpod_rsync \
    -e "${ssh_transport}" \
    "${dataset_dir}/" \
    "${RUNPOD_SSH_HOST}:${remote_root}/${dataset_dir}/"
runpod_rsync \
    -e "${ssh_transport}" \
    "${model_dir}/latest.llmckpt" \
    "${model_dir}/best.llmckpt" \
    "${model_dir}/best.llmckpt.metrics.json" \
    "${model_dir}/training.jsonl" \
    "${RUNPOD_SSH_HOST}:${remote_root}/${model_dir}/"

ssh "${ssh_options[@]}" "${RUNPOD_SSH_HOST}" \
    "mkdir -p '${remote_root}' && touch '${remote_root}/.llm-lab-inputs-ready'"
