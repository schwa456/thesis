import os
import time

os.environ["TZ"] = "Asia/Seoul"
if hasattr(time, "tzset"):
    time.tzset()

import yaml
import uvicorn
import logging
import argparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

logger = logging.getLogger("Agent-Server")
logging.basicConfig(level=logging.INFO)

# === Config File Path ===
CONFIG_PATH = "./config/server_config.yaml"
app = FastAPI(title="vLLM Model Server")

engine = None
role_name = ""

# === 1. Load the config ===
def load_config(path):
    logger.info(f"[SERVER] Loading configuration from {path}...")
    with open (path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
    
# === 2. Lifespan ===
@app.on_event("startup")
async def startup_event():
    global engine
    pass

# === 3. Request Schema ===
class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 2048
    temperature: float = 0.1
    stop: list[str] = []

# === 4. API Endpoint ===   
@app.post("/generate")
async def generate(req: GenerationRequest):
    if not engine:
        raise HTTPException(status_code=503, detail="Model Not Loaded")
    
    request_id = f"req={hash(req.prompt)}"

    sampling_params = SamplingParams(
        temperature = req.temperature,
        max_tokens = req.max_tokens,
        stop = req.stop + ["<|im_end|>"]
    )

    results_generator = engine.generate(req.prompt, sampling_params, request_id)

    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    return {"result": final_output.outputs[0].text}

@app.get("/v1/models")
async def show_models():
    current_model = getattr(app.state, "model_id", 'unknown-model')
    return {
        "object": "list",
        "data": [
            {
                "id": current_model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "vllm"
            }
        ]
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", type=str, required=True, choices=["agent", "generator"], help="Model role")
    parser.add_argument("--port", type=int, required=True, help="Server Port")
    parser.add_argument("--gpu-util", type=float, default=0.8, help="GPU Memory Utilization")
    args = parser.parse_args()

    role_name = args.role
    config = load_config(CONFIG_PATH)

    if args.role == "agent":
        agent_config = config.get('agent', {})
        model_id = agent_config.get('model_id', 'Qwen/Qwen3-Coder-30B-A3B-Instruct')
        max_len = 16384
    elif args.role == "generator":
        gen_config = config.get('generator', {})
        model_id = config.get('model_id', "XGenerationLab/XiYanSQL-QwenCoder-32B-2504")
        max_len = 8192

    app.state.model_id = model_id

    logger.info(f"[INFO] Loading Agent Model: {model_id} ...")
    logger.info(f"[INFO] Model: {model_id}")
    logger.info(f"[INFO] Port: {args.port}")
    logger.info(f"[INFO] GPU Util: {args.gpu_util}")

    
    if args.role == "agent":    
        engine_args = AsyncEngineArgs(
            model=model_id, 
            trust_remote_code=True,
            max_model_len=max_len,
            gpu_memory_utilization=args.gpu_util,
            quantization="fp8",
            dtype="auto",
            disable_log_stats=False
        )
    
    elif args.role == "generator":
        engine_args = AsyncEngineArgs(
            model=model_id,
            max_model_len=max_len,
            gpu_memory_utilization=args.gpu_util,
            quantization="bitsandbytes",
            load_format="bitsandbytes",
            dtype="auto",
            disable_log_stats=False
        )

    engine = AsyncLLMEngine.from_engine_args(engine_args)

    uvicorn.run(app, host="0.0.0.0", port=args.port)
