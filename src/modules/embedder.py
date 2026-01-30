import torch
from sentence_transformers import SentenceTransformer, util
import logging

logger = logging.getLogger(__name__)

class SchemaEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug(f"Loading Embedder: {model_name} on {self.device}...")
        
        self.model = SentenceTransformer(model_name, device=self.device)
        
        self.cached_embeddings = None
        self.cached_node_ids = None

    def index_schema_nodes(self, node_texts, node_ids):
        """
        [전처리 단계] 
        그래프 노드들의 텍스트 설명(node_texts)을 받아 임베딩 후 저장합니다.
        한 번만 실행하면 됩니다.
        """
        if not node_texts:
            logger.warning("No schema texts provided for indexing.")
            return

        logger.debug(f"Indexing {len(node_texts)} schema nodes...")
        
        # 텍스트 -> 벡터 변환 (한 번만 수행)
        self.cached_embeddings = self.model.encode(
            node_texts, 
            convert_to_tensor=True, 
            device=self.device,
            show_progress_bar=False
        )
        self.cached_node_ids = node_ids # 점수 계산 후 ID 매핑을 위해 저장
        
        logger.debug("Schema indexing complete.")

    def get_similarity_scores(self, question):
        """
        [실행 단계]
        질문 하나를 받아, 미리 캐싱된 스키마 노드들과의 유사도를 계산합니다.
        """
        if self.cached_embeddings is None:
            raise ValueError("Schema embeddings not indexed! Call index_schema_nodes() first.")

        # 1. 질문 임베딩
        q_emb = self.model.encode(question, convert_to_tensor=True, device=self.device, show_progress_bar=False)
        
        # 2. 유사도 계산 (질문 vs 미리 저장된 스키마 벡터들)
        scores = util.cos_sim(q_emb, self.cached_embeddings)[0]
        
        return scores
