#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "${RUNPOD_PROJECT_ROOT}"

: "${RUNPOD_API_KEY:?Set RUNPOD_API_KEY in deploy/runpod/.env before running this script}"

if ! command -v runpodctl >/dev/null 2>&1; then
    printf 'Install runpodctl first, then retry: https://docs.runpod.io/runpodctl/overview\n' >&2
    exit 2
fi

# runpodctl v2 reads RUNPOD_API_KEY directly. Doctor validates the key and
# configures or registers the local SSH key needed by the rsync workflow.
exec runpodctl doctor
