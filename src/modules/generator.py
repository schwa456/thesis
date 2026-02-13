import os
import re
import requests
import logging
from openai import AzureOpenAI
from abc import abstractmethod, ABC

logger = logging.getLogger(__name__)

class BaseGenerator(ABC):
    @abstractmethod
    def generate_query(self, question, schema_info):
        pass

class XiYanGenerator(BaseGenerator):
    def __init__(self, server_url: str = "http://localhost:8001", prompt_path: str = None):
        self.server_url = server_url
        self.endpoint = f"{self.server_url}/generate"
        self.prompt_template = self._load_prompt(prompt_path)
        logger.info(f"Generator Client Initialized: {self.endpoint}")
    
    def _load_prompt(self, path):
        """ 프롬프트 파일을 읽어오는 헬퍼 함수"""
        if not path or not os.path.exists(path):
            logger.warning(f"[WARNING] Prompt File Not Found at {path}. Using default.")
            return """
            [Default Prompt]
            Schema: {schema_info}
            Question: {question}
            Evidence: {evidence}
            Generate SQL:
            """
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"[ERROR] Failed to load prompt: {e}")
            raise e

    def generate_query(self, question, schema_info, evidence=None):

        evidence_str = ""
        if evidence:
            evidence_str = f"{evidence}"
        
        try:
            prompt = self.prompt_template.format(
                dialect='sqlite',
                schema_info=schema_info,
                question=question,
                evidence=evidence_str
            )
        except KeyError as e:
            logger.error(f"[ERROR] Prompt Formatting Error: Missing Key: {e}")
            return ""

        chat_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n```sql\n"

        logger.debug(f"Prompt: \n{prompt}")

        try:
            payload = {
                "prompt": chat_prompt,
                "max_tokens": 2048,
                "temperature": 0.1,
                "stop": ["```", "<|im_end|>", ";\n", "```sql"], 
                "repetition_penalty": 1.1,
                "echo": False
            }

            logger.debug("Sending SQL Generation Request...")
            response = requests.post(self.endpoint, json=payload, timeout=600)

            if response.status_code == 200:
                result_text = response.json().get('result', '')
                logger.debug(f"[DEBUG] Generator Raw Output: \n{result_text}")

                return self._process_response(result_text)
            else:
                logger.error(f"SQL Generation Failed: {response.status_code} - {response.text}")
                return ""
        
        except requests.exceptions.ConnectionError:
            logger.critical("Failed to connect to SQL Server. Is 'server.py'  running?")
            return ""
        
        except Exception as e:
            logger.error(f"[ERROR] Error during SQL Generation: {e}")
            return ""
    
    def _process_response(self, response_text: str) -> str:
        query = self._extract_sql_query(response_text)
        
        query = query.strip().rstrip(';')
        
        return query

    def _extract_sql_query(self, text: str) -> str:
        if not text:
            return ""
            
        # 1. 마크다운 블록 추출 (```sql ... ```)
        code_block_pattern = r"```(?:sql)?\s*(.*?)```"
        match = re.search(code_block_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            sql = match.group(1).strip()
            if ";" in sql:
                sql = sql.split(";")[0] + ";"
            return sql
        
        # 2. SELECT 문 추출 (SELECT ... ;)
        select_pattern = r"(SELECT\s[\s\S]+?;)"
        match = re.search(select_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # 3. SELECT 문 추출 (SELECT ...) - 세미콜론 없을 때
        # 이미 ```sql 로 유도했으므로, 텍스트 전체가 SQL일 가능성이 높음
        if "SELECT" in text.upper():
            # 첫 번째 세미콜론까지만 자름
            if ";" in text:
                return text.split(";")[0].strip() + ";"
            return text.strip()
        
        # 최후의 수단: 그냥 원본 반환
        return text.strip()