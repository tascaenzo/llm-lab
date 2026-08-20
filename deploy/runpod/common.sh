#!/usr/bin/env bash

# Shared setup for commands launched on the developer machine. Values from the
# ignored root .env are exported so the RunPod scripts work without a manual
# series of `export` commands.
readonly RUNPOD_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly RUNPOD_ENV_FILE="${LLM_LAB_ENV_FILE:-${RUNPOD_PROJECT_ROOT}/.env}"
if [[ -f "${RUNPOD_ENV_FILE}" ]]; then
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
    source "${RUNPOD_ENV_FILE}"
    set +a
    for ((index = 0; index < ${#exported_names[@]}; ++index)); do
        printf -v "${exported_names[index]}" '%s' "${exported_values[index]}"
        export "${exported_names[index]}"
    done
    unset exported_name exported_names exported_values index
elif [[ -n "${LLM_LAB_ENV_FILE:-}" ]]; then
    printf 'RunPod pipeline: environment file does not exist: %s\n' "${RUNPOD_ENV_FILE}" >&2
    return 2
fi

runpod_require_positive_integer() {
    local name="$1"
    local value="$2"
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
        printf 'RunPod pipeline: %s must be a positive integer.\n' "${name}" >&2
        return 2
    fi
}
