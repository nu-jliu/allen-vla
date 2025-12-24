#!/bin/bash
# Dataset will be pushed to: jliu6718/act-so101-MM-DD-YYYY

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd ${PROJECT_ROOT}
exec uv run data_collection/collect.py \
    --leader-port /dev/ttyACM1 \
    --leader-id my_leader \
    --follower-port /dev/ttyACM0 \
    --follower-id my_follower \
    --username jliu6718 \
    --policy-type act \
    --robot-type so101 \
    --hz 30 \
    --push \
    --camera-index 0 \
    --camera-width 640 \
    --camera-height 480 \
    --root ${PROJECT_ROOT}/data \
    --port 1234