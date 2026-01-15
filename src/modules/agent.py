import json
import re
import torch
from transformers import pipeline, AutoTokenizer, BitsAndBytesConfig
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def filter_schema(self, question, candidates):
        pass

class LlamaRejectionAgent(BaseAgent):
    def __init__(self, model_id, quantization=True):
        print(f"Loading Agent: {model_id}...")

        if quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

        model_kwargs = {"load_in_4bit": True} if quantization else {}

        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs=model_kwargs,
            device_map="auto",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    def filter_schema(self, question, candidates):
        candidates_str = "\n".join([f"- {t}" for t in candidates])
    
        system_prompt = """You are a highly intelligent Database Architect.
    Your Goal: Select the minimal set of tables required to answer the user's SQL question from the candidates.

    RULES:
    1. **Bridge Tables**: If a table is needed to join two relevant tables (even if not explicitly mentioned in the question), YOU MUST KEEP IT.
    2. **Output Format**: Return ONLY a valid JSON list of strings. No explanations outside the JSON.
    Example: ["table_a", "table_b"]"""

        user_prompt = f"""Question: "{question}"
    Candidate Tables:
    {candidates_str}

    Which tables are truly necessary? Return JSON only."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        outputs = self.pipe(
            prompt,
            max_new_tokens=256,
            do_sample=False, 
            temperature=0.1,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        return outputs[0]["generated_text"][len(prompt):]

    def _parse_json(self, text):
        try:
            match = re.search(r'\[.*?\]', text, re.DOTALL)
            if match:
                return json.loads(match.group(0).replace("'", '"'))
        except:
            pass
        return []