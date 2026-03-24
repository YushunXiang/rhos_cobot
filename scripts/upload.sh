#!/bin/bash
set -e

if [ $# -ne 2 ]; then
    echo "Usage: $0 <DATA_DIR> <TASK_NAME>"
    echo "Example: $0 ocl_data turn_on_tap"
    exit 1
fi

DATA_DIR="$1"
TASK_NAME="$2"

cd "$DATA_DIR"
7z a -mmt=on "${TASK_NAME}.7z" "${TASK_NAME}"
obsutil cp "${TASK_NAME}.7z" "obs://sai.liyl/xiangyushun/${TASK_NAME}.7z"
rm "${TASK_NAME}.7z"
