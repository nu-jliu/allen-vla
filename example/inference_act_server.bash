#!/bin/bash
# Checkpoint format: {username}/{policy}-{robot}-{task}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd ${PROJECT_ROOT}
exec uv run policy/act/inference_server.py \
    --checkpoint jliu6718/act-so101-place_brick \
    --host 0.0.0.0 \
    --port 8000 \
    --device cuda \
    --task place_brick