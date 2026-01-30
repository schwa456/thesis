import yaml
import uvicorn
import logging
import argparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sympy.physics.units import temperature
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vLLM-Server")

# === Config File Path ===
CONFIG_PATH = "./config/exp_config.yaml"
app = FastAPI(title="vLLM-Server")

engine = None
role_name = ""

# === 1. Configuration Loading ===
def load_config(path):
    logger.info(f"[SERVER] Loading configuration from {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# === 2. Startup Event (Model Loading) ===
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
        raise HTTPException(status_code=503, detail="Model is not loaded")

    request_id = f"{role_name}-{hash(req.prompt)}"

    sampling_params = SamplingParams(
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        stop=req.stop + ["<|im_end|>"]
    )

    results_generator = engine.generate(req.prompt, sampling_params, request_id)

    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    return {"result": final_output.outputs[0].text}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM Server")
    parser.add_argument("--role", type=str, required=True, choices=['agent', 'generator'], help="Model role")
    parser.add_argument("--port", type=int, required=True, help="Server Port")
    parser.add_argument("--gpu-util", type=float, default=0.8, help="GPU Memory Utilization (0.0 ~ 1.0)")
    args = parser.parse_args()

    role_name = args.role
    config = load_config(CONFIG_PATH)

    if args.role == "agent":
        model_id = config.get('agent_model_id', 'Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8')
        max_len = 32768

        # Initializing vLLM engine
        engine_args = AsyncEngineArgs(
            model=model_id,
            trust_remote_code=True,
            max_model_len=max_len,
            gpu_memory_utilization=args.gpu_util,
            quantization="fp8",
            dtype="auto",
            disable_log_stats=False
        )

        engine = AsyncLLMEngine.from_engine_args(engine_args)

    elif args.role == "generator":
        model_id = config.get('generator_model_id', 'XGenerationLab/XiYanSQL-QwenCoder-32B-2504')
        max_len = 2048

        # Initializing vLLM engine
        engine_args = AsyncEngineArgs(
            model=model_id,
            trust_remote_code=True,
            max_model_len=max_len,
            gpu_memory_utilization=args.gpu_util,
            quantization="bitsandbytes",
            load_format="bitsandbytes",
            dtype="auto",
            disable_log_stats=False
        )

        engine = AsyncLLMEngine.from_engine_args(engine_args)

    else:
        model_id = None
        max_len = None

    logger.info(f"[INFO] Starting Server for [{args.role.upper()}]")
    logger.info(f"[INFO] Model: {model_id}")
    logger.info(f"[INFO] Port: {args.port}")
    logger.info(f"[INFO] GPU Util: {args.gpu_util}")

    uvicorn.run(app, host="0.0.0.0", port=args.port)
