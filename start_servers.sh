#!/bin/bash

# 1. 안전장치: 스크립트 종료 (Ctrl + c) 시 실행 중인 Python Process들도 같이 Kill
clean_up() {
    echo ""
    echo "Stopping all servers..."

    kill $(jobs -p)>/dev/null

    exit 1
}
trap clean_up SIGINT SIGTERM EXIT

wait_for_server() {
    PORT=$1
    NAME=$2
    echo "[INFO] Waiting for $NAME ($PORT) to load model..."

    while ! python -c "import urllib.request; urllib.request.urlopen('http://localhost:$PORT/v1/models')" > /dev/null 2>&1; do
        sleep 2
        echo -n "."
    done
    echo ""
    echo "[INFO] $NAME is READY"
}

echo "[INFO] Starting Servers..."

# 2. Agent Server (GPU 0번, agent_server.log)
echo "[INFO] Starting Agent Server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python server.py --role agent --port 8000 > logs/server_logs/agent_server.log 2>&1 &
AGENT_PID=$!

wait_for_server 8000 "Agent"

# 3. Generator Server (GPU 1번, generator_server.log)
echo "[INFO] Starting Generator Server on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python server.py --role generator --port 8001 > logs/server_logs/generator_server.log 2>&1 &
GEN_PID=$!

wait_for_server 8001 "Generator"

echo "[INFO] All Systems READY"
echo "      - Agent PID: $AGENT_PID"
echo "      - Generator PID: $GEN_PID"
echo "[INFO] Logs are being saved to 'logs/server_logs/' directory"
echo "[INFO] Press Ctrl+C to stop both servers"

wait
