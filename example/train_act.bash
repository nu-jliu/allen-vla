#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd ${PROJECT_ROOT}
exec uv run policy/act/train_act.py \
    --repo-id jliu6718/lerobot-so101-act \
    --output-dir ${PROJECT_ROOT}/model \
    --batch-size 32 \
    --steps 10000 \
    --seed 42 \
    --push \
    --dataset-root ${PROJECT_ROOT}/data