#!/usr/bin/env bash
set -euo pipefail

readonly project_dir=/opt/llm-lab
readonly workspace_dir="${LLM_LAB_WORKSPACE:-/workspace/llm-lab}"
readonly train_steps="${LLM_LAB_TRAIN_STEPS:?Set LLM_LAB_TRAIN_STEPS before starting the Pod}"
readonly checkpoint_every="${LLM_LAB_CHECKPOINT_EVERY:-1000}"
readonly validation_every="${LLM_LAB_VALIDATION_EVERY:-1000}"
readonly validation_batches="${LLM_LAB_VALIDATION_BATCHES:-100}"
readonly dataset="${workspace_dir}/data/derived/italiano-v3/lm/italiano-v3.train.llmdat"
readonly validation_dataset="${workspace_dir}/data/derived/italiano-v3/lm/italiano-v3.validation.llmdat"
readonly model_dir="${workspace_dir}/artifacts/models/italiano-base-75m"
readonly checkpoint="${model_dir}/latest.llmckpt"
readonly best_checkpoint="${model_dir}/best.llmckpt"
readonly training_log="${model_dir}/training.jsonl"
readonly ready_marker="${workspace_dir}/.llm-lab-inputs-ready"

require_file() {
    if [[ ! -f "$1" ]]; then
        printf 'RunPod pipeline: required file is missing: %s\n' "$1" >&2
        exit 2
    fi
}

if ! [[ "${train_steps}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'RunPod pipeline: LLM_LAB_TRAIN_STEPS must be a positive integer.\n' >&2
    exit 2
fi

nvidia-smi
if [[ ! -f "${ready_marker}" ]]; then
    printf 'RunPod pipeline: waiting for input upload marker %s\n' "${ready_marker}" >&2
    printf 'Run deploy/runpod/sync_to_pod.sh, then stop and restart this Pod.\n' >&2
    exec sleep infinity
fi
require_file "${dataset}"
require_file "${validation_dataset}"
require_file "${checkpoint}"
mkdir -p "${model_dir}"

# A real NVIDIA device is the missing gate for the CUDA backend. This is quick
# and validates CPU/CUDA numerical parity before any checkpoint is modified.
ctest --test-dir "${project_dir}/build/cuda-release" \
    --output-on-failure -R '^runtime\\.cuda_backend$'

exec "${project_dir}/build/cuda-release/llm-lab" model train \
    "${dataset}" \
    "${train_steps}" \
    --backend cuda \
    --resume "${checkpoint}" \
    --checkpoint "${checkpoint}" \
    --checkpoint-every "${checkpoint_every}" \
    --validation "${validation_dataset}" \
    --validation-every "${validation_every}" \
    --validation-batches "${validation_batches}" \
    --best-checkpoint "${best_checkpoint}" \
    --log "${training_log}"
