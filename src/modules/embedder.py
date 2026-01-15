import torch
from sentence_transformers import SentenceTransformer, util

class SchemaEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Embedder: {model_name} on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.cache = {}

    def get_similarity_scores(self, question, table_list):
        """
        질문과 Table List 간의 코사인 유사도 반환
        """

        if not table_list:
            return torch.tensor([], decvice=self.device)

        q_emb = self.model.encode(question, convert_to_tensor=True, device=self.device)
        t_embs = self.model.encode(table_list, convert_to_tensor=True, device=self.device)

        scores = util.cos_sim(q_emb, t_embs)[0]
        return scores