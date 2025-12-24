#!/bin/bash
# Dataset/Model repo ID format: username/policy-robot-MM-DD-YYYY

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd ${PROJECT_ROOT}
exec uv run policy/act/train.py \
    --repo-id jliu6718/act-so101-12-24-2025 \
    --output-dir ${PROJECT_ROOT}/model \
    --batch-size 32 \
    --steps 10000 \
    --seed 42 \
    --push