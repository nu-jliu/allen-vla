#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd ${PROJECT_ROOT}
exec uv run policy/diffusion/inference_client.py \
    --robot-port /dev/ttyACM0 \
    --robot-id my_follower \
    --camera-index 0 \
    --camera-name front \
    --camera-width 640 \
    --camera-height 480 \
    --camera-fps 30 \
    --server-host 192.168.100.146 \
    --server-port 8000
