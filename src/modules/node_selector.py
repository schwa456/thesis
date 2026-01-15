import torch
from abc import ABC, abstractmethod

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