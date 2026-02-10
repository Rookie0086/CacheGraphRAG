#!/usr/bin/env bash

# Non-root Milvus standalone helper
# Usage: bash setup/milvus-install-user.sh start|stop|delete
# This script does NOT use sudo and stores data under $HOME/.local/share/milvus

set -euo pipefail

DATA_DIR="$HOME/.local/share/milvus"
VOLUMES_DIR="$DATA_DIR/volumes"
CONFIG_FILE="$DATA_DIR/embedEtcd.yaml"
CONTAINER_NAME="milvus-standalone"
IMAGE="milvusdb/milvus:v2.3.9"

ensure_docker_available() {
    if ! command -v docker >/dev/null 2>&1; then
        echo "ERROR: docker not found. Install Docker or use a host with Docker available." >&2
        exit 1
    fi

    if ! docker ps >/dev/null 2>&1; then
        echo "ERROR: cannot access Docker daemon. Check that you can run 'docker ps' without sudo." >&2
        echo "If you see a permission error, ask admin to add you to the 'docker' group or use a machine with Docker access." >&2
        exit 1
    fi
}

check_ports() {
    # check if essential ports are free (simple check)
    for p in 19530 9091 2379; do
        if ss -ltnp 2>/dev/null | grep -q ":$p "; then
            echo "WARNING: port $p appears in use; Milvus may fail to bind to this port." >&2
        fi
    done
}

run_embed() {
    mkdir -p "$VOLUMES_DIR"
    cat > "$CONFIG_FILE" <<EOF
listen-client-urls: http://0.0.0.0:2379
advertise-client-urls: http://0.0.0.0:2379
EOF

    echo "Pulling image $IMAGE (if not present)..."
    docker pull "$IMAGE"

    echo "Starting container $CONTAINER_NAME..."
    docker run -d \
        --name "$CONTAINER_NAME" \
        --security-opt seccomp:unconfined \
        -e ETCD_USE_EMBED=true \
        -e ETCD_DATA_DIR=/var/lib/milvus/etcd \
        -e ETCD_CONFIG_PATH=/milvus/configs/embedEtcd.yaml \
        -e COMMON_STORAGETYPE=local \
        -v "$VOLUMES_DIR":/var/lib/milvus \
        -v "$CONFIG_FILE":/milvus/configs/embedEtcd.yaml \
        -p 19530:19530 -p 9091:9091 -p 2379:2379 \
        --health-cmd="curl -f http://localhost:9091/healthz" \
        --health-interval=30s --health-start-period=90s --health-timeout=20s --health-retries=3 \
        "$IMAGE" milvus run standalone 1> /dev/null
}

wait_for_milvus_running() {
    echo "Waiting for Milvus to become healthy..."
    local tries=0
    while true; do
        # Prefer docker filter for health (works on modern docker)
        if docker ps --filter "name=$CONTAINER_NAME" --filter "health=healthy" --format '{{.ID}}' | grep -q .; then
            echo "Start successfully.";
            break
        fi

        # fallback: check status string
        if docker ps --format '{{.Names}} {{.Status}}' | grep -E "^$CONTAINER_NAME .*healthy" >/dev/null 2>&1; then
            echo "Start successfully.";
            break
        fi

        tries=$((tries+1))
        if [ $tries -gt 180 ]; then
            echo "ERROR: Milvus did not become healthy in time; check logs: docker logs -f $CONTAINER_NAME" >&2
            exit 1
        fi
        sleep 1
    done
}

start() {
    ensure_docker_available
    check_ports

    # already running?
    if docker ps --filter "name=$CONTAINER_NAME" --filter "status=running" --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
        echo "Milvus is already running."
        exit 0
    fi

    # container exists but stopped
    if docker ps -a --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
        echo "Starting existing container $CONTAINER_NAME..."
        docker start "$CONTAINER_NAME" >/dev/null
    else
        run_embed
    fi

    wait_for_milvus_running
}

stop() {
    if docker ps --filter "name=$CONTAINER_NAME" --filter "status=running" --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
        echo "Stopping $CONTAINER_NAME..."
        docker stop "$CONTAINER_NAME" >/dev/null
        echo "Stop successfully."
    else
        echo "Milvus is not running."
    fi
}

delete() {
    if docker ps --filter "name=$CONTAINER_NAME" --filter "status=running" --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
        echo "Please stop Milvus service before delete." >&2
        exit 1
    fi

    if docker ps -a --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
        echo "Removing container $CONTAINER_NAME..."
        docker rm "$CONTAINER_NAME" >/dev/null || { echo "ERROR: failed to remove container" >&2; exit 1; }
    fi

    echo "Removing data directory $DATA_DIR..."
    rm -rf "$DATA_DIR"
    echo "Delete successfully."
}

case ${1-} in
    start)
        start
        ;;
    stop)
        stop
        ;;
    delete)
        delete
        ;;
    *)
        echo "Usage: bash $0 start|stop|delete"
        echo "Note: this script assumes you can run 'docker' without sudo (e.g. you're in the docker group)."
        exit 2
        ;;
esac
