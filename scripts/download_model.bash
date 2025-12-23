#!/bin/bash

# Download script to sync model directory from remote Jetson machine
# Usage: ./scripts/download_model.bash <username> <hostname>
# Example: ./scripts/download_model.bash allen jetson

set -e  # Exit on error

# Check if both arguments are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: Username and hostname required"
    echo "Usage: $0 <username> <hostname>"
    echo "Example: $0 allen jetson"
    exit 1
fi

USERNAME="$1"
HOSTNAME="$2"
SSH_TARGET="${USERNAME}@${HOSTNAME}"
REMOTE_DIR="/home/${USERNAME}/.ws/vla_ws/model/checkpoints/last"

# Get absolute path to project root (parent of script directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOCAL_DIR="${PROJECT_ROOT}/model/"

echo "================================================"
echo "Downloading model from: ${SSH_TARGET}:${REMOTE_DIR}"
echo "To local directory: ${LOCAL_DIR}"
echo "================================================"

# Use rsync to sync files from remote to local
# -a: archive mode (preserves permissions, timestamps, etc.)
# -v: verbose
# -z: compress during transfer
# --progress: show progress
# --delete: delete files locally that don't exist on remote
# --mkpath: create local directory path if it doesn't exist
rsync -avzL --progress --delete --mkpath \
    "${SSH_TARGET}:${REMOTE_DIR}" "${LOCAL_DIR}"

echo "================================================"
echo "Model download completed successfully!"
echo "================================================"
