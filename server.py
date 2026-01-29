import yaml
import uvicorn
import logging
import argparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import AsyncLLMEngine, AsynvEngineArgs, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vLLM-Server")

# === Config File Path ===
CONFIG_PATH = "./config/exp_config.yaml"
app = FastAPI(title="vLLM-Server")

engine = None
role_name = ""

# === 1. Configuration Loading ===
def load_config(path):
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