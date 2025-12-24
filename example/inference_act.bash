#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd ${PROJECT_ROOT}
exec uv run policy/act/inference.py \
    --checkpoint jliu6718/lerobot-so101-act \
    --robot-port /dev/ttyACM0 \
    --camera-index 0 \
    --repo-id jliu6718/eval_lerobot-so101-act \
    --robot-id my_follower \
    --camera-name front \
    --camera-width 640 \
    --camera-height 480 \
    --camera-fps 30 \
    --fps 30 \
    --root ${PROJECT_ROOT}/data \
    --push-to-hub \
    --no-display