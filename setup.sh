#!/usr/bin/env bash
set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: ./setup.sh /path/to/cholec-tinytools" >&2
    exit 1
fi

DATA_ROOT="$1"
if [ ! -d "$DATA_ROOT/train" ] || [ ! -d "$DATA_ROOT/validation" ]; then
    echo "Dataset root must contain train/ and validation/: $DATA_ROOT" >&2
    exit 1
fi

export DATA_ROOT
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
exec ./run_all.sh
