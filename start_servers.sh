#!/bin/bash

cleanup() {
  echo ""
  echo "[INFO] Stopping All Servers..."

  # shellcheck disable=SC2046
  kill $(jobs -p) 2>/dev/null

  exit 1
}

trap cleanup SIGINT SIGTERM EXIT

wait_for_server() {
  PORT=$1
  NAME=$2

  echo "[INFO] Waiting for $NAME ($PORT) to load model..."

  while ! python -c "import urllib.request; urllib.request.urlopen('http://localhost:$PORT/v1/models')" > /dev/null 2>&1; do
    sleep 5
    echo -n "."
  done

  echo ""
  echo "[INFO] $NAME is READY"
}

echo "[INFO] Starting Servers Sequentially..."

# 1. Agent Server 실행 (Port 8000)
echo "Starting Agent Server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python server.py --role agent --port 8000 > logs/server_logs/agent_server.log 2>&1 &
AGENT_PID=$!

wait_for_server 8000 "Agent"

echo "----------------------------------------------------"
# 2. Generator Server 실행 (Port 8001)
echo "Starting Agent Server on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python server.py --role generator --port 8001 > logs/server_logs/generator_server.log 2>&1 &
GEN_PID=$!

wait_for_server 8001 "Generator"

echo "----------------------------------------------------"
echo "All Systems READY"
echo "Logs: tail -f logs/server_logs/agent_server.log logs/server_logs/generator_server.log"

wait
