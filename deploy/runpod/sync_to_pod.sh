#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "${RUNPOD_PROJECT_ROOT}"

: "${RUNPOD_SSH_HOST:?Set RUNPOD_SSH_HOST to user@host}"
: "${RUNPOD_SSH_PORT:?Set RUNPOD_SSH_PORT to the RunPod SSH port}"

readonly remote_root="${RUNPOD_REMOTE_ROOT:-/workspace/llm-lab}"
readonly model_dir=artifacts/models/italiano-base-75m
readonly dataset_dir=data/derived/italiano-v3/lm

rsync --archive --compress --partial --append-verify --progress \
    -e "ssh -p ${RUNPOD_SSH_PORT}" \
    "${dataset_dir}/" \
    "${RUNPOD_SSH_HOST}:${remote_root}/${dataset_dir}/"
rsync --archive --compress --partial --append-verify --progress \
    -e "ssh -p ${RUNPOD_SSH_PORT}" \
    "${model_dir}/latest.llmckpt" \
    "${model_dir}/best.llmckpt" \
    "${model_dir}/best.llmckpt.metrics.json" \
    "${model_dir}/training.jsonl" \
    "${RUNPOD_SSH_HOST}:${remote_root}/${model_dir}/"

ssh -p "${RUNPOD_SSH_PORT}" "${RUNPOD_SSH_HOST}" \
    "mkdir -p '${remote_root}' && touch '${remote_root}/.llm-lab-inputs-ready'"
