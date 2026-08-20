#!/usr/bin/env bash

# Shared setup for commands launched on the developer machine. RunPod values
# are deliberately isolated from the root local-runtime .env file.
readonly RUNPOD_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly RUNPOD_CONFIG_FILE="${RUNPOD_ENV_FILE:-${RUNPOD_PROJECT_ROOT}/deploy/runpod/.env}"
if [[ -f "${RUNPOD_CONFIG_FILE}" ]]; then
    # Preserve variables already exported by the caller: process environment
    # has higher precedence than values stored in the local dotenv file.
    exported_names=()
    exported_values=()
    while IFS= read -r exported_name; do
        exported_names+=("${exported_name}")
        exported_values+=("${!exported_name}")
    done < <(compgen -e)
    set -a
    # shellcheck disable=SC1091
    source "${RUNPOD_CONFIG_FILE}"
    set +a
    for ((index = 0; index < ${#exported_names[@]}; ++index)); do
        printf -v "${exported_names[index]}" '%s' "${exported_values[index]}"
        export "${exported_names[index]}"
    done
    unset exported_name exported_names exported_values index
else
    printf 'RunPod pipeline: environment file does not exist: %s\n' "${RUNPOD_CONFIG_FILE}" >&2
    printf 'Create it with: cp deploy/runpod/.env.example deploy/runpod/.env\n' >&2
    return 2
fi

runpod_require_positive_integer() {
    local variable_name="$1"
    local value="$2"
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
        printf 'RunPod pipeline: %s must be a positive integer.\n' "${variable_name}" >&2
        return 2
    fi
}

runpod_pod_environment_json() {
    local names=(LLM_LAB_BACKEND LLM_LAB_TRAIN_STEPS LLM_LAB_CHECKPOINT_EVERY
                 LLM_LAB_VALIDATION_EVERY LLM_LAB_VALIDATION_BATCHES LLM_LAB_CUDA_TF32
                 LLM_LAB_PROFILE_CUDA)
    local index name value separator=''
    printf '{'
    for ((index = 0; index < ${#names[@]}; ++index)); do
        name="${names[index]}"
        value="${!name-}"
        if [[ -z "${value}" ]]; then
            printf 'RunPod pipeline: %s is missing from %s.\n' "${name}" \
                "${RUNPOD_CONFIG_FILE}" >&2
            return 2
        fi
        if [[ "${value}" == *['"\\'$'\n'$'\r']* ]]; then
            printf 'RunPod pipeline: %s contains a character not supported in Pod configuration.\n' \
                "${name}" >&2
            return 2
        fi
        printf '%s"%s":"%s"' "${separator}" "${name}" "${value}"
        separator=','
    done
    printf '}'
}

runpod_rsync() {
    # Checkpoints are rewritten atomically but retain the same byte length. Do
    # not use --append here: old macOS rsync then treats a stale equal-size file
    # as complete and silently skips the new checkpoint.
    rsync --archive --compress --partial --progress "$@"
}

runpod_rsync_resume_immutable_upload() {
    if rsync --append-verify --version >/dev/null 2>&1; then
        rsync --archive --compress --partial --append-verify --progress "$@"
        return
    fi

    # macOS ships rsync 2.6.9, which lacks --append-verify. --append keeps
    # interrupted large transfers resumable; SSH already protects each block.
    printf 'RunPod pipeline: rsync lacks --append-verify; using --append for immutable upload.\n' >&2
    rsync --archive --compress --partial --append --progress "$@"
}
