#!/usr/bin/env bash

# Non-root Nebula-Graph installer helper
# Usage: bash setup/nebula-install-user.sh install|start|stop|delete [working_path]
# Default working path: $HOME/.nebula-up

set -euo pipefail

WORKING_PATH_DEFAULT="$HOME/.nebula-up"
CMD=${1-}
WORKING_PATH=${2:-$WORKING_PATH_DEFAULT}
NEBULA_VERSION=${NEBULA_VERSION:-release-3.6}
STUDIO_VERSION=${STUDIO_VERSION:-3.7.0}
CONSOLE_VERSION=${CONSOLE_VERSION:-v3.5.0}

docker_compose_cmd() {
    if command -v docker-compose >/dev/null 2>&1; then
        echo "docker-compose"
    else
        # prefer "docker compose" if available
        if docker compose version >/dev/null 2>&1; then
            echo "docker compose"
        else
            return 1
        fi
    fi
}

ensure_docker_available() {
    if ! command -v docker >/dev/null 2>&1; then
        echo "ERROR: docker not found. Install Docker or use a host with Docker available." >&2
        exit 1
    fi

    if ! docker ps >/dev/null 2>&1; then
        echo "ERROR: cannot access Docker daemon. Ensure you can run 'docker ps' without sudo (in docker group)." >&2
        exit 1
    fi

    if ! docker_compose_cmd >/dev/null 2>&1; then
        echo "ERROR: docker-compose (legacy) or 'docker compose' (v2) is required. Install docker-compose or use Docker with compose v2." >&2
        exit 1
    fi
}

print_usage() {
    cat <<EOF
Usage: bash $0 install|start|stop|delete [working_path]

Examples:
  bash $0 install            # install into $HOME/.nebula-up
  bash $0 install /path/you/want
  bash $0 start  /path/you/want
  bash $0 stop   /path/you/want
  bash $0 delete /path/you/want

Note: this script does NOT use sudo. It assumes you can run Docker and docker-compose without sudo.
EOF
}

install_nebula_graph() {
    echo "Preparing Nebula working dir: $WORKING_PATH"
    mkdir -p "$WORKING_PATH"
    cd "$WORKING_PATH"

    if [ ! -d "$WORKING_PATH/nebula-docker-compose" ]; then
        echo "Cloning nebula-docker-compose (branch: $NEBULA_VERSION)"
        git clone --branch "$NEBULA_VERSION" https://github.com/vesoft-inc/nebula-docker-compose.git nebula-docker-compose
    else
        echo "nebula-docker-compose already exists, reusing it"
        cd nebula-docker-compose && git fetch --all && git checkout "$NEBULA_VERSION" || true
        cd - >/dev/null 2>&1 || true
    fi

    echo "Pulling Nebula images and starting..."
    cd "$WORKING_PATH/nebula-docker-compose"
    $(docker_compose_cmd) pull
    $(docker_compose_cmd) up -d
}

install_nebula_graph_studio() {
    echo "Installing Nebula Graph Studio (version: $STUDIO_VERSION)"
    mkdir -p "$WORKING_PATH"
    cd "$WORKING_PATH"

    studio_dir="nebula-graph-studio-$STUDIO_VERSION"
    if [ -d "$studio_dir" ]; then
        rm -rf "$studio_dir"
    fi

    tarball="nebula-graph-studio-$STUDIO_VERSION.tar.gz"
    echo "Downloading Studio tarball..."
    wget -q "https://oss-cdn.nebula-graph.com.cn/nebula-graph-studio/${STUDIO_VERSION}/$tarball" -O "$tarball" || wget -q "https://oss-cdn.nebula-graph.com.cn/nebula-graph-studio/$tarball" -O "$tarball"
    mkdir -p "$studio_dir"
    tar -zxvf "$tarball" -C "$studio_dir" >/dev/null 2>&1 || true

    cd "$studio_dir"
    # ensure network name matches nebula setup (nebula-net)
    sed -i 's/nebula-web/nebula-net/g' docker-compose.yml 2>/dev/null || true

    echo "Pulling Studio images and starting..."
    $(docker_compose_cmd) pull || true
    $(docker_compose_cmd) up -d || true
}

create_console_sh() {
    cat > "$WORKING_PATH/console.sh" <<EOF
#!/usr/bin/env bash
export DOCKER_DEFAULT_PLATFORM=linux/amd64
# Usage: console.sh [nebula-console args]
docker run --rm -ti --network nebula-net --volume $WORKING_PATH:/root vesoft/nebula-console:${CONSOLE_VERSION} -addr graphd -port 9669 -u root -p nebula "\$@"
EOF
    chmod +x "$WORKING_PATH/console.sh"
    echo "Created console helper: $WORKING_PATH/console.sh"
}

wait_for_nebula_up() {
    echo "Waiting for Nebula containers to be healthy (this may take a while)..."
    local max_attempts=60
    local attempt=1
    while [ $attempt -le $max_attempts ]; do
        healthy_count=$(docker ps --filter health=healthy --filter "name=nebula" | grep -v CONTAINER | wc -l | tr -d ' ')
        expected=9
        if [ "$healthy_count" = "$expected" ]; then
            echo "All Nebula containers appear healthy."
            return 0
        fi
        echo "Attempt $attempt/$max_attempts: healthy=$healthy_count, waiting..."
        attempt=$((attempt+1))
        sleep 5
    done
    echo "Warning: Not all Nebula containers reached healthy state; check 'docker ps' and logs." >&2
}

start() {
    ensure_docker_available
    if [ -d "$WORKING_PATH/nebula-docker-compose" ]; then
        cd "$WORKING_PATH/nebula-docker-compose"
        echo "Starting Nebula Graph containers..."
        $(docker_compose_cmd) up -d
    else
        echo "Nebula docker-compose not found in $WORKING_PATH. Run 'install' first." >&2
        exit 1
    fi
}

stop() {
    ensure_docker_available
    if [ -d "$WORKING_PATH/nebula-docker-compose" ]; then
        cd "$WORKING_PATH/nebula-docker-compose"
        echo "Stopping Nebula Graph containers..."
        $(docker_compose_cmd) down
    else
        echo "Nebula docker-compose not found in $WORKING_PATH." >&2
    fi

    if [ -d "$WORKING_PATH/nebula-graph-studio-$STUDIO_VERSION" ]; then
        cd "$WORKING_PATH/nebula-graph-studio-$STUDIO_VERSION"
        echo "Stopping Nebula Graph Studio..."
        $(docker_compose_cmd) down || true
    fi
}

delete() {
    stop
    echo "Removing working directory: $WORKING_PATH"
    rm -rf "$WORKING_PATH"
    echo "Delete successfully."
}

case "$CMD" in
    install)
        ensure_docker_available
        install_nebula_graph
        install_nebula_graph_studio
        create_console_sh
        wait_for_nebula_up
        ;;
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
        print_usage
        exit 2
        ;;
esac
