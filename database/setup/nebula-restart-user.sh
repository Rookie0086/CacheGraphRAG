#!/usr/bin/env bash

# Non-root Nebula restart helper
# Usage: bash setup/nebula-restart-user.sh /path/to/nebula-docker-compose

set -euo pipefail

TARGET_DIR=${1-}

if [ -z "$TARGET_DIR" ]; then
    echo "Usage: bash $0 /path/to/nebula-docker-compose" >&2
    exit 2
fi

if [ ! -d "$TARGET_DIR" ]; then
    echo "ERROR: Directory $TARGET_DIR does not exist." >&2
    exit 1
fi

# try to detect a docker-compose command
if command -v docker-compose >/dev/null 2>&1; then
    DC="docker-compose"
elif docker compose version >/dev/null 2>&1; then
    DC="docker compose"
else
    echo "ERROR: docker-compose or 'docker compose' is required." >&2
    exit 1
fi

if ! docker ps >/dev/null 2>&1; then
    echo "ERROR: cannot access Docker daemon. Ensure you can run 'docker ps' without sudo." >&2
    exit 1
fi

echo "Restarting Nebula in $TARGET_DIR..."
cd "$TARGET_DIR"
$DC down
$DC up -d

echo "Restart requested. Check 'docker ps' and container logs for status."