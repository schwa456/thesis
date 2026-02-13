import torch
from sentence_transformers import CrossEncoder
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class SchemaReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """
        Cross-Encoder 기반 Reranker 초기화
        """
        logger.debug(f"[INFO] Loading Reranker Model: {model_name}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=device)

    def compute_scores(self, question: str, candidates: List[str]) -> List[float]:
        """
        질문과 후보군(Table/Column 텍스트) 쌍의 적합도 점수 계산
        Args:
            question: 질문 텍스트
            candidates: 스키마 노드 텍스트 리스트 (Table name, Column info...)
        Returns:
            scores: 0~1 사이의 정규화된 점수 리스트 (Sigmoid 적용 권장)
        """

        if not candidates:
            return []
        
        pairs = [[question, cand] for cand in candidates]

        scores = self.model.predict(pairs)

        scores_tensor = torch.tensor(scores)
        probs = torch.sigmoid(scores_tensor).tolist()

        return probs