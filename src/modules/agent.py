import json
import os
import re
import ast
import requests
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    @abstractmethod
    def filter_schema(self, question, candidates):
        pass

class FilteringAgent(BaseAgent):
    def __init__(self, server_url: str = "http://localhost:8000", system_prompt_path: str = None, user_prompt_path: str = None):
        self.server_url = server_url
        self.endpoint = f"{self.server_url}/generate"
        self.system_prompt_path = system_prompt_path
        self.user_prompt_path = user_prompt_path
        logger.debug(f"Agent Initialized via Server: {self.endpoint}")

    def _load_system_prompt(self, path):
        if not path or not os.path.exists(path):
            logger.warning(f"[WARNING] Prompt File Not Found at {path}. Using default.")
            return """
            [Default Prompt]
            You are an expert Data Analyst & Database Architect.
            Your task is to select the **relevant columns** from the 'Candidate Schema' to answer the user's question.

            Provide the final list in a valid JSON block: ```json ["Table.Column", ...] ```
            """
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"[ERROR] Failed to load prompt: {e}")
            raise e
    
    def _load_user_prompt(self, path):
        if not path or not os.path.exists(path):
            logger.warning(f"[WARNING] Prompt File Not Found at {path}. Using default.")
            return """
            [Default Prompt]
            User Question: {question}

            Candidate Schema: {candidates_str}

            Task: Which columns are truly necessary to construct the whole SQL query? Return JSON only.
            """
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"[ERROR] Failed to load prompt: {e}")
            raise e
    
    def filter_schema(self, question, retrieved_graph, evidence=None):
        schema_lines = []

        table_nodes = [n for n, attr in retrieved_graph.nodes(data=True) if attr.get('type') == 'table']
        table_nodes.sort()

        for table_node in table_nodes:
            table_name = retrieved_graph.nodes[table_node].get('name')

            cols = []
            neighbors = retrieved_graph.neighbors(table_node)

            for n in neighbors:
                neighbor_attr = retrieved_graph.nodes[n]
                if neighbor_attr.get('type') == 'column':
                    col_name = neighbor_attr.get('name')
                    col_type = neighbor_attr.get('dtype', '')

                    col_desc = f"{col_name}"
                    if 'values' in neighbor_attr:
                        cold_desc += f" (sample: {neighbor_attr['values']})"
                    
                    cols.append(col_desc)
            
            if cols:
                schema_lines.append(f"- {table_name}: [{', '.join(cols)}]")
            else:
                schema_lines.append(f"- {table_name}: []")
        
        candidates_str = "\n".join(schema_lines)
    
        system_prompt = self._load_system_prompt(self.system_prompt_path)
        
        try:
            user_prompt = self._load_user_prompt(self.user_prompt_path).format(
                question=question,
                candidates_str=candidates_str,
                evidence=evidence
            )
        except KeyError as e:
            logger.error(f"[ERROR] Prompt Formatting Error: Missing Key: {e}")
            return ""

        full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

        try:
            payload = {
                "prompt": full_prompt,
                "max_tokens": 200,
                "temperature": 0.1
            }

            logger.debug("Sending Request to Agent Server...")
            response = requests.post(self.endpoint, json=payload, timeout=600)

            if response.status_code == 200:
                result_text = response.json().get('result', '')
                if "<|im_start|>assistant" in result_text:
                    result_text = result_text.split("<|im_start|>assistant")[-1]
                
                result_text = result_text.replace("<|im_end|>", "").strip()

                logger.debug(f"Agent Output: {result_text}")
                return self._parse_llm_output(result_text)
            else:
                logger.error(f"[ERROR] {response.status_code} - {response.text}")
                return []
        
        except Exception as e:
            logger.error(f"[ERROR] Agent Filtering Connection Failed: {e}")
            return []

    def _parse_json(self, text):
        try:
            match = re.search(r'\[.*?\]', text, re.DOTALL)
            if match:
                return json.loads(match.group(0).replace("'", '"'))
        except:
            pass
        return []

    def _parse_llm_output(self, output_text):
        try:
            # 1. Regex로 JSON 리스트 블록 추출
            match = re.search(r'\[(.*?)\]', output_text, re.DOTALL)
            
            parsed_result = []
            if match:
                json_str = match.group(0)
                try:
                    parsed_result = json.loads(json_str)
                except json.JSONDecodeError:
                    parsed_result = ast.literal_eval(json_str)

            else:
                logger.warning(f"[Failed] Failed to find JSON block in output: {output_text[:100]}...")
                return []
        
            if isinstance(parsed_result, list):
                clean_list = [item for item in parsed_result if isinstance(item, str)]
                return clean_list
        
            return []
        
        except Exception as e:
            logger.error(f"[ERROR] Parsing Failed: {e}")
            return []

    def generate(self, prompt: str) -> str:
        if "<|im_start|>" not in prompt:
            full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            full_prompt = prompt
        
        payload = {
            "prompt": full_prompt,
            "max_tokens": 2048,
            "temperature": 0.0
        }

        try:
            response = requests.post(self.endpoint, json=payload, timeout=600)

            if response.status_code == 200:
                result_text = response.json().get('result', '')
                if "<|im_start|>assistant" in result_text:
                    result_text = result_text.split("<|im_start|>assistant")[-1]
                
                result_text = result_text.replace("<|im_end|>", "").strip()
                return result_text
            else:
                logger.error(f"[ERROR] Generate Request Failed: {response.status_code} - {response.text}")
                return ""
        
        except Exception as e:
            logger.error(f"[ERROR] Agent Connection Failed during generate(): {e}")
            return ""