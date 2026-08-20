#!/usr/bin/env bash
set -euo pipefail

readonly project_dir=/opt/llm-lab
readonly backend="${LLM_LAB_BACKEND:-cuda}"
readonly workspace_dir="${LLM_LAB_WORKSPACE:-/workspace/llm-lab}"
readonly train_steps="${LLM_LAB_TRAIN_STEPS:?Set LLM_LAB_TRAIN_STEPS before starting the Pod}"
readonly checkpoint_every="${LLM_LAB_CHECKPOINT_EVERY:-1000}"
readonly validation_every="${LLM_LAB_VALIDATION_EVERY:-1000}"
readonly validation_batches="${LLM_LAB_VALIDATION_BATCHES:-100}"
readonly profile_cuda="${LLM_LAB_PROFILE_CUDA:-0}"
readonly dataset="${workspace_dir}/data/derived/italiano-v3/lm/italiano-v3.train.llmdat"
readonly validation_dataset="${workspace_dir}/data/derived/italiano-v3/lm/italiano-v3.validation.llmdat"
readonly model_dir="${workspace_dir}/artifacts/models/italiano-base-75m"
readonly checkpoint="${model_dir}/latest.llmckpt"
readonly best_checkpoint="${model_dir}/best.llmckpt"
readonly training_log="${model_dir}/training.jsonl"
readonly ready_marker="${workspace_dir}/.llm-lab-inputs-ready"
readonly profile_dir="${workspace_dir}/artifacts/benchmarks/runpod"
readonly preflight_marker="${workspace_dir}/.llm-lab-cuda-preflight"
readonly profile_marker="${workspace_dir}/.llm-lab-cuda-profile"
readonly cli="${project_dir}/build/cuda-release/llm-lab"
readonly cuda_test="${project_dir}/build/cuda-release/tests/runtime_cuda_backend_test"
readonly benchmark="${project_dir}/build/cuda-release/utils/benchmarks/runtime_benchmark"

require_file() {
    if [[ ! -f "$1" ]]; then
        printf 'RunPod pipeline: required file is missing: %s\n' "$1" >&2
        exit 2
    fi
}

start_ssh() {
    mkdir -p /run/sshd /root/.ssh
    chmod 0700 /root/.ssh
    if [[ -n "${PUBLIC_KEY:-}" ]]; then
        printf '%s\n' "${PUBLIC_KEY}" > /root/.ssh/authorized_keys
        chmod 0600 /root/.ssh/authorized_keys
    else
        printf 'RunPod pipeline: PUBLIC_KEY is empty; SSH file transfer is unavailable.\n' >&2
    fi
    /usr/sbin/sshd
}

marker_matches() {
    [[ -f "$1" ]] && [[ "$(<"$1")" == "$2" ]]
}

write_marker() {
    local marker_path="$1"
    local marker_value="$2"
    local temporary_marker="${marker_path}.tmp"
    printf '%s\n' "${marker_value}" > "${temporary_marker}"
    mv "${temporary_marker}" "${marker_path}"
}

require_positive_integer() {
    local variable_name="$1"
    local value="$2"
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
        printf 'RunPod pipeline: %s must be a positive integer.\n' "${variable_name}" >&2
        exit 2
    fi
}

require_positive_integer LLM_LAB_TRAIN_STEPS "${train_steps}"
require_positive_integer LLM_LAB_CHECKPOINT_EVERY "${checkpoint_every}"
require_positive_integer LLM_LAB_VALIDATION_EVERY "${validation_every}"
require_positive_integer LLM_LAB_VALIDATION_BATCHES "${validation_batches}"
if [[ "${backend}" != "cuda" ]]; then
    printf 'RunPod pipeline: LLM_LAB_BACKEND must be cuda for this image.\n' >&2
    exit 2
fi
if [[ "${profile_cuda}" != "0" && "${profile_cuda}" != "1" ]]; then
    printf 'RunPod pipeline: LLM_LAB_PROFILE_CUDA must be 0 or 1.\n' >&2
    exit 2
fi

start_ssh
nvidia-smi
if [[ ! -f "${ready_marker}" ]]; then
    printf 'RunPod pipeline: waiting for input upload marker %s\n' "${ready_marker}" >&2
    printf 'Run deploy/runpod/sync_to_pod.sh, then stop and restart this Pod.\n' >&2
    exec sleep infinity
fi
require_file "${dataset}"
require_file "${validation_dataset}"
require_file "${checkpoint}"
require_file "${cli}"
require_file "${cuda_test}"
require_file "${benchmark}"
mkdir -p "${model_dir}" "${profile_dir}"

readonly build_key="$(sha256sum "${cli}" "${cuda_test}" | sha256sum | cut -d' ' -f1)"

# Run the numerical suite and one disposable real trainer step once per image.
# The input checkpoint is hashed before and after to prove that the preflight
# did not consume or rewrite the saved training progress.
if ! marker_matches "${preflight_marker}" "${build_key}"; then
    ctest --test-dir "${project_dir}/build/cuda-release" \
        --output-on-failure --no-tests=error -R '^runtime\.cuda_backend$'

    readonly preflight_timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    readonly preflight_work_dir="$(mktemp -d /tmp/llm-lab-cuda-preflight.XXXXXX)"
    readonly preflight_checkpoint="${preflight_work_dir}/preflight.llmckpt"
    readonly preflight_log="${profile_dir}/cuda-preflight-${preflight_timestamp}.jsonl"
    readonly preflight_summary="${profile_dir}/cuda-preflight-${preflight_timestamp}.summary.json"
    readonly checkpoint_hash_before="$(sha256sum "${checkpoint}" | cut -d' ' -f1)"
    cleanup_preflight() {
        rm -f "${preflight_checkpoint}" "${preflight_checkpoint}.metrics.json"
        rmdir "${preflight_work_dir}" 2>/dev/null || true
    }
    trap cleanup_preflight EXIT
    printf 'RunPod pipeline: running one disposable CUDA trainer step...\n' >&2
    "${cli}" model train \
        "${dataset}" \
        1 \
        --resume "${checkpoint}" \
        --checkpoint "${preflight_checkpoint}" \
        --checkpoint-every 1 \
        --log "${preflight_log}" \
        > "${preflight_summary}"
    readonly checkpoint_hash_after="$(sha256sum "${checkpoint}" | cut -d' ' -f1)"
    if [[ "${checkpoint_hash_before}" != "${checkpoint_hash_after}" ]]; then
        printf 'RunPod pipeline: preflight changed the source checkpoint; aborting.\n' >&2
        exit 3
    fi
    cleanup_preflight
    trap - EXIT
    write_marker "${preflight_marker}" "${build_key}"
    printf 'RunPod pipeline: CUDA trainer preflight passed; source checkpoint unchanged.\n' >&2
fi

if [[ "${profile_cuda}" == "1" ]]; then
    readonly profile_key="${build_key}:$(sha256sum "${project_dir}/utils/benchmarks/profile_model.py" | cut -d' ' -f1)"
    if ! marker_matches "${profile_marker}" "${profile_key}"; then
        readonly profile_file="${profile_dir}/cuda-profile-$(date -u +%Y%m%dT%H%M%SZ).txt"
        printf 'RunPod pipeline: profiling one representative CUDA training update...\n' >&2
        python3 "${project_dir}/utils/benchmarks/profile_model.py" \
            --benchmark "${benchmark}" \
            > "${profile_file}"
        write_marker "${profile_marker}" "${profile_key}"
        printf 'RunPod pipeline: CUDA profile saved to %s\n' "${profile_file}" >&2
    else
        printf 'RunPod pipeline: CUDA profile already completed for this image; skipped.\n' >&2
    fi
fi

exec "${cli}" model train \
    "${dataset}" \
    "${train_steps}" \
    --resume "${checkpoint}" \
    --checkpoint "${checkpoint}" \
    --checkpoint-every "${checkpoint_every}" \
    --validation "${validation_dataset}" \
    --validation-every "${validation_every}" \
    --validation-batches "${validation_batches}" \
    --best-checkpoint "${best_checkpoint}" \
    --log "${training_log}"
