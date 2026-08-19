#!/usr/bin/env bash
set -euo pipefail

: "${RUNPOD_API_KEY:?Export RUNPOD_API_KEY before running this script}"

readonly image="${RUNPOD_IMAGE:-ghcr.io/tascaenzo/llm-lab-cuda:latest}"
readonly gpu="${RUNPOD_GPU_ID:-NVIDIA GeForce RTX 4090}"
readonly name="${RUNPOD_POD_NAME:-llm-lab-cuda-train}"
readonly volume_gb="${RUNPOD_VOLUME_GB:-40}"
readonly train_steps="${LLM_LAB_TRAIN_STEPS:-1000}"
readonly checkpoint_every="${LLM_LAB_CHECKPOINT_EVERY:-1000}"
readonly validation_every="${LLM_LAB_VALIDATION_EVERY:-1000}"
readonly validation_batches="${LLM_LAB_VALIDATION_BATCHES:-100}"

if ! command -v runpodctl >/dev/null 2>&1; then
    printf 'Install runpodctl first, then retry: https://docs.runpod.io/runpodctl/overview\n' >&2
    exit 2
fi

runpodctl config --apiKey "${RUNPOD_API_KEY}"
runpodctl pod create \
    --name "${name}" \
    --gpu-id "${gpu}" \
    --gpu-count 1 \
    --image "${image}" \
    --container-disk-in-gb 30 \
    --volume-in-gb "${volume_gb}" \
    --volume-mount-path /workspace \
    --ports "22/tcp" \
    --ssh \
    --env "{\"LLM_LAB_TRAIN_STEPS\":\"${train_steps}\",\"LLM_LAB_CHECKPOINT_EVERY\":\"${checkpoint_every}\",\"LLM_LAB_VALIDATION_EVERY\":\"${validation_every}\",\"LLM_LAB_VALIDATION_BATCHES\":\"${validation_batches}\"}"
