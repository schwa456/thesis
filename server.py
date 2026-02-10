import os
import time
import uuid
import traceback

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
from typing import List, Optional, Union

logger = logging.getLogger("Agent-Server")
logging.basicConfig(level=logging.INFO)

# === Config File Path ===
CONFIG_PATH = "./config/server_config.yaml"
app = FastAPI(title="vLLM Model Server")

engine = None
engine_args = None
role_name = ""

# === 1. Load the config ===
def load_config(path):
    logger.info(f"[SERVER] Loading configuration from {path}...")
    with open (path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
    
# === 2. Lifespan ===
@app.on_event("startup")
async def startup_event():
    global engine, engine_args
    logger.info("[STARTUP] Initializing vLLM Engine within Event Loop...")

    if engine_args is None:
        logger.error("[ERROR] Engine Args not set.")
        raise RuntimeError("Engine Args must be set before startup.")

    try:
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info("[STARTUP] vLLM Engine Initialized Successfully.")
    except Exception as e:
        logger.error(f"[STARTUP] Failed to initialize engine: {e}")
        traceback.print_exc()
        raise e
    
    logger.info("[STARTUP] Warming up vLLM Engine...")
    try:
        dummy_req = SamplingParams(max_tokens=1)
        async for _ in engine.generate("Hello", dummy_req, str(uuid.uuid4())):
            pass
        logger.info("[STARTUP] Warm-up Completed.")
    except Exception as e:
        logger.warning(f"[STARTUP] Warm-up failed (Non-Critical): {e}")

# === 3. Request Schema ===
class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 2048
    temperature: float = 0.1
    stop: list[str] = []

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.1
    stop: Optional[Union[str, List[str]]] = []
    stream: Optional[bool] = False

# === 4. API Endpoint ===   
@app.post("/generate")
async def generate(req: GenerationRequest):
    if not engine:
        raise HTTPException(status_code=503, detail="Model Not Loaded")
    
    request_id = str(uuid.uuid4())

    sampling_params = SamplingParams(
        temperature = req.temperature,
        max_tokens = req.max_tokens,
        stop = req.stop + ["<|im_end|>"]
    )

    results_generator = engine.generate(req.prompt, sampling_params, request_id)

    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    if final_output is None:
        logger.warning(f"Model generated Empty Output for Req: {request_id}")
        return {"result": ""}

    return {"result": final_output.outputs[0].text}

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """ [New] OpenAI Client 호환용 엔드포인트 (SemanticGraphBuilder용) """
    if not engine:
        raise HTTPException(status_code=503, detail="Model Not Loaded yet")

    request_id = str(uuid.uuid4())

    # 1. ChatML Prompt Construction (Qwen Style)
    # <|im_start|>system...<|im_end|><|im_start|>user...<|im_end|><|im_start|>assistant
    prompt = ""
    for msg in req.messages:
        prompt += f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"

    # 2. Stop words 처리
    stop_words = ["<|im_end|>"]
    if req.stop:
        if isinstance(req.stop, str):
            stop_words.append(req.stop)
        else:
            stop_words.extend(req.stop)

    try:
        sampling_params = SamplingParams(
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            stop=stop_words
        )

        results_generator = engine.generate(prompt, sampling_params, request_id)

        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        
        output_text = ""
        if final_output:
            output_text = final_output.outputs[0].text

        # 3. OpenAI-compatible Response Construction
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": output_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(final_output.prompt_token_ids) if final_output else 0,
                "completion_tokens": len(final_output.outputs[0].token_ids) if final_output else 0,
                "total_tokens": (len(final_output.prompt_token_ids) + len(final_output.outputs[0].token_ids)) if final_output else 0
            }
        }

    except Exception as e:
        logger.error(f"Chat Completion Error [Req: {request_id}]: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

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
        model_id = agent_config.get('model_id', 'Qwen/Qwen3-8B-AWQ')
        max_len = 16384
    elif args.role == "generator":
        gen_config = config.get('generator', {})
        model_id = config.get('model_id', "XGenerationLab/XiYanSQL-QwenCoder-3B-2504")
        max_len = 8192

    app.state.model_id = model_id

    logger.info(f"[INFO] Loading Agent Model: {model_id} ...")
    logger.info(f"[INFO] Model: {model_id}")
    logger.info(f"[INFO] Port: {args.port}")
    logger.info(f"[INFO] GPU Util: {args.gpu_util}")

    quant_method = None
    load_fmt = "auto"

    model_upper = model_id.upper()
    if "AWQ" in model_upper:
        quant_method = "awq"
    elif "FP8" in model_upper:
        quant_method = "fp8"
    elif "GPTQ" in model_upper:
        quant_method = "gptq"

    if "BITSANDBYTES" in model_upper:
        quant_method = "bitsandbytes"
        load_fmt="bitsandbytes"

    if args.role == "agent":    
        engine_args = AsyncEngineArgs(
            model=model_id, 
            trust_remote_code=True,
            max_model_len=max_len,
            gpu_memory_utilization=args.gpu_util,
            quantization=quant_method,
            load_format=load_fmt,
            dtype="auto",
            disable_log_stats=False
        )
    
    elif args.role == "generator":
        engine_args = AsyncEngineArgs(
            model=model_id,
            max_model_len=max_len,
            gpu_memory_utilization=args.gpu_util,
            quantization=quant_method,
            load_format=load_fmt,
            enforce_eager=True,
            dtype="auto",
            disable_log_stats=False
        )

    uvicorn.run(app, host="0.0.0.0", port=args.port)
