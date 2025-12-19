#!/bin/bash

# Deploy script to sync files to remote Jetson machine
# Usage: ./scripts/deploy_jetson.bash <user@host>
# Example: ./scripts/deploy_jetson.bash allen@jetson

set -e  # Exit on error

# Check if SSH argument is provided
if [ -z "$1" ]; then
    echo "Error: No SSH target specified"
    echo "Usage: $0 <user@host>"
    echo "Example: $0 allen@jetson"
    exit 1
fi

SSH_TARGET="$1"
REMOTE_DIR="/home/allen/allen_ws/"

echo "================================================"
echo "Deploying to: ${SSH_TARGET}:${REMOTE_DIR}"
echo "================================================"

# Use rsync to sync files
# -a: archive mode (preserves permissions, timestamps, etc.)
# -v: verbose
# -z: compress during transfer
# --progress: show progress
# --delete: delete files on remote that don't exist locally
# --delete-excluded: delete excluded files on remote
# --mkpath: create remote directory path if it doesn't exist
# --exclude: exclude unnecessary files
rsync -avz --progress --delete --delete-excluded --mkpath \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.pytest_cache/' \
    --exclude '.mypy_cache/' \
    --exclude '.ruff_cache/' \
    --exclude '*.egg-info/' \
    --exclude 'dist/' \
    --exclude 'build/' \
    --exclude '.DS_Store' \
    --exclude '*.swp' \
    --exclude '*.swo' \
    ./ "${SSH_TARGET}:${REMOTE_DIR}"

echo "================================================"
echo "Deployment completed successfully!"
echo "================================================"
