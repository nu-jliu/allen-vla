#!/bin/bash
# Evaluation dataset will be pushed to: jliu6718/eval_act-so101-place_brick
# Repo-ID format: {username}/eval_{policy}-{robot}-{task}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd ${PROJECT_ROOT}
exec uv run policy/act/inference.py \
    --checkpoint jliu6718/act-so101-place_brick \
    --robot-port /dev/ttyACM0 \
    --camera-index 0 \
    --username jliu6718 \
    --policy-type act \
    --robot-type so101 \
    --task place_brick \
    --robot-id my_follower \
    --camera-name front \
    --camera-width 640 \
    --camera-height 480 \
    --camera-fps 30 \
    --fps 30 \
    --root ${PROJECT_ROOT}/data \
    --push-to-hub \
    --no-display