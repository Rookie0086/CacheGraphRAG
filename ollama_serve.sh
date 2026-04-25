echo "🚀 Starting Ollama..."

# ====== 可配置 ======
MODEL_SIZE=${1:-4B}
LOG_FILE=~/ollama.log
OLLAMA_PATH=~/.local/bin/ollama

# ====== 杀掉旧进程 ======
pkill ollama 2>/dev/null

# ====== 环境变量 ======
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# ====== 并行控制 ======
if [ "$MODEL_SIZE" = "4B" ]; then
    export OLLAMA_NUM_PARALLEL=4
    echo "🧠 Model: 4B → parallel=4"
elif [ "$MODEL_SIZE" = "7B" ]; then
    export OLLAMA_NUM_PARALLEL=2
    echo "🧠 Model: 7B → parallel=2"
else
    export OLLAMA_NUM_PARALLEL=2
    echo "⚠️ Unknown model size → parallel=2"
fi

# ====== 启动 ======
nohup $OLLAMA_PATH serve > $LOG_FILE 2>&1 &

sleep 2

echo "✅ Ollama started"
echo "📄 Log: tail -f $LOG_FILE"
echo "📊 GPU: watch -n 1 nvidia-smi"