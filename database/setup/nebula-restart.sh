restart_nebula_service() {
    if [ -z "$1" ]; then
        echo "Please provide a destination path."
        return 1  # Exit the function if no path is provided
    fi

    local docker_path=$1

    echo "cd $(pwd)"
    echo "restart $2..."
    cd $docker_path
    docker-compose down
    docker-compose up -d
}

# restart_nebula_service ~/.nebula-up/nebula-docker-compose nebula_graph
restart_nebula_service /home/shuyurui/dataset/.nebula-up/nebula-graph-studio-3.7.0 nebula_graph_studio
