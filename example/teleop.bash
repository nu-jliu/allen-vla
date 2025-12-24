#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd ${PROJECT_ROOT}
exec uv run teleop/teleop.py \
    --leader-port /dev/ttyACM1 \
    --leader-id my_leader \
    --follower-port /dev/ttyACM0 \
    --follower-id my_follower \
    --frequency 30.0