#!/bin/bash

# Deploy script to sync files to remote Jetson machine
# Usage: ./scripts/deploy_remote.bash <ssh_target>
# Example: ./scripts/deploy_remote.bash allen@jetson
# Example: ./scripts/deploy_remote.bash jetson  # if configured in SSH config

set -e  # Exit on error

# Check if argument is provided
if [ -z "$1" ]; then
    echo "Error: SSH target required"
    echo "Usage: $0 <ssh_target>"
    echo "Example: $0 allen@jetson"
    echo "Example: $0 jetson  # if configured in SSH config"
    exit 1
fi

SSH_TARGET="$1"

# Parse SSH target to determine remote directory
if [[ "$SSH_TARGET" == *"@"* ]]; then
    # Extract username from user@host format
    USERNAME="${SSH_TARGET%%@*}"
    REMOTE_DIR="/home/${USERNAME}/vla_ws/"
else
    # No username specified, use ~ which expands on remote
    REMOTE_DIR="~/vla_ws/"
fi

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
    --exclude 'data/' \
    --exclude 'model/' \
    --exclude '.DS_Store' \
    --exclude '*.swp' \
    --exclude '*.swo' \
    ./ "${SSH_TARGET}:${REMOTE_DIR}"

echo "================================================"
echo "Deployment completed successfully!"
echo "================================================"
