#!/bin/bash

# Deploy script to sync files to remote Jetson machine
# Usage: ./scripts/deploy_remote.bash <username> <hostname>
# Example: ./scripts/deploy_remote.bash allen jetson

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
REMOTE_DIR="/home/${USERNAME}/.ws/vla_ws/"

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
