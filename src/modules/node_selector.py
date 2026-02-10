import logging
import torch
import json
import ast
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Any

logger = logging.getLogger(__name__)

class BaseNodeSelector(ABC):
    @abstractmethod
    def select_seed(self, scores, candidates):
        pass

class FixedTopKSelector(BaseNodeSelector):
    """ 단순 Top-k """
    def __init__(self, k=3):
        self.k = k
    
    def select_seed(self, scores, candidates):
        top_k_indices = torch.topk(scores, k=min(self.k, len(candidates))).indices
        return [candidates[i] for i in top_k_indices]
    
class AdaptiveSelector(BaseNodeSelector):
    """ 임계값 기반 적응형 선택기 """
    def __init__(self, alpha=0.8, min_k=2, max_k=5):
        self.alpha = alpha
        self.min_k = min_k
        self.max_k = max_k

    def select_seed(self, scores, candidates):
        if not candidates: return []
        
        # 점수 내림차순 정렬
        sorted_indices = torch.argsort(scores, descending=True)
        top_score = scores[sorted_indices[0]].item()
        
        seeds = []
        for idx in sorted_indices:
            score = scores[idx].item()
            # 조건: (Top1 점수의 alpha% 이상) OR (최소 개수 미달 시)
            if (score >= top_score * self.alpha) or (len(seeds) < self.min_k):
                seeds.append(candidates[idx])
            else:
                break
            
            if len(seeds) >= self.max_k:
                break
                
        return seeds
    
class AgentNodeSelector(BaseNodeSelector):
    """ LLM Agent 기반 Seed 선택기, Uncertainty를 PCST의 Prize로 활용 """
    def __init__(self, agent_model):
        self.agent = agent_model

    def select_seed(self, scores, candidates, **kwargs) -> Dict[str, float]:
        question = kwargs.get('question')
        if not question:
            raise ValueError("AgentNodeSelector requires 'question' in kwargs.")
        
        prompt = self._construct_prompt(question, candidates)

        logger.debug(f"[AgentSelector] Prompt Preview: {prompt[:500]}...")

        response = self.agent.generate(prompt)

        logger.debug(f"\n[DEBUG] >>> Agent Raw Output:\n{response}\n------------------------------------------------")

        try:
            parsed_result = self._parse_json_response(response)
            logger.debug(f"[DEBUG] >>> Parsed JSON: {parsed_result}")
        except Exception as e:
            logger.error(f"[ERROR] JSON Parsing Failed: {e}")
            return {}
        
        if not parsed_result.get('is_answerable',True):
            logger.info("[AgentSelector] Agent decided the question is UNANSWERABLE (is_answerable=False).")
            return {}
        
        weighted_seeds = parsed_result.get('selected_tables', {})

        final_seeds = {}
        for k, v in weighted_seeds.items():
            try:
                # 점수 정규화
                final_seeds[k] = min(max(float(v), 0.0), 1.0)
            except:
                continue
        
        if not final_seeds:
            logger.warning("[AgentSelector] Parsed JSON has no valid 'selected_tables'.")

        return final_seeds
    
    def _construct_prompt(self, question, candidates):
        return f"""
        You are a database expert focusing on selecting relevant tables for a SQL query.
        Given the user's question and the list of available tables, select the tables that are **strictly necessary** to answer the question.

        Current Task:
        1. Analyze the question: "{question}"
        2. Review available tables: {candidates}
        3. Select relevant tables and assign a **confidence score (0.0 to 1.0)** for each based on how sure you are.
        4. If the question cannot be answered with the given tables, set "is_answerable" to false.

        Output Format (JSON Only):
        {{
            "is_answerable": true,
            "reasoning": "Brief explanation here...",
            "selected_tables": {{
                "table_name_A": 0.95,
                "table_name_B": 0.7
            }}
        }}
        """
    
    def _parse_json_response(self, response: str) -> dict:
            """
            LLM 응답에서 JSON을 추출하고 파싱합니다.
            싱글 쿼트('), 소문자 불리언(true/false) 등이 섞여 있어도 처리할 수 있도록 강화되었습니다.
            """
            text_to_parse = response.strip()

            # 1. 마크다운 코드 블록 제거
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
            if json_match:
                text_to_parse = json_match.group(1)
            else:
                # 코드 블록 없으면 중괄호 탐색
                json_match = re.search(r"(\{.*\})", response, re.DOTALL)
                if json_match:
                    text_to_parse = json_match.group(1)

            # 2. 표준 JSON 파싱 시도 (가장 빠르고 정확함)
            try:
                return json.loads(text_to_parse)
            except json.JSONDecodeError:
                pass # 실패 시 다음 단계로

            # 3. Python Literal Eval 시도 (싱글 쿼트 처리)
            # JSON의 true/false/null을 Python의 True/False/None으로 치환해야 ast가 인식함
            py_compatible_text = (
                text_to_parse
                .replace("true", "True")
                .replace("false", "False")
                .replace("null", "None")
            )
            
            try:
                return ast.literal_eval(py_compatible_text)
            except Exception:
                pass # 이것도 실패하면 에러 발생

            raise ValueError(f"Failed to parse JSON content: {text_to_parse[:50]}...")