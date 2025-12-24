#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd ${PROJECT_ROOT}
exec uv run policy/act/inference_server.py \
    --checkpoint jliu6718/lerobot-so101-act \
    --host 0.0.0.0 \
    --port 8000 \
    --device cuda