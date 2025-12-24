#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd ${PROJECT_ROOT}
exec uv run data_collection/collect.py \
    --leader-port /dev/ttyACM1 \
    --leader-id my_leader \
    --follower-port /dev/ttyACM0 \
    --follower-id my_follower \
    --repo-id jliu6718/lerobot-so101-act \
    --hz 30 \
    --push \
    --camera-index 0 \
    --camera-width 640 \
    --camera-height 480 \
    --root ${PROJECT_ROOT}/data 