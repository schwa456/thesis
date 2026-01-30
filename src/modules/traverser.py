import networkx as nx
import itertools
from abc import ABC, abstractmethod
from typing import List, Set, Any

class BaseGraphTraverser(ABC):
    @abstractmethod
    def traverse(self, graph: nx.Graph, seeds: List[str]) -> Set[str]:
        """
        Args:
            graph: 전체 지식 그래프(NetworkX)
            seeds: 출발점이 될 노드 ID 리스트

        Returns:
            Set[str]: 최종 서브 그래프에 포함될 노드 ID 집합
        """
        pass

class ShortestPathTraverser(BaseGraphTraverser):
    """
    선택된 Seed들 사이의 최단 경로를 찾아 연결하는 방식.
    (Steiner Tree의 근사 해법)
    """
    def __init__(self, max_hops=3):
        self.max_hops = max_hops

    def traverse(self, graph, seeds):
        result_nodes = set(seeds)

        if len(seeds) > 1:
            for src, tgt in itertools.combinations(seeds, 2):
                try:
                    if nx.has_path(graph, src, tgt):
                        path = nx.shortest_path(graph, src, tgt)
                        if len(path) <= self.max_hops + 1:
                            result_nodes.update(path)
                except Exception:
                    continue
        
        return result_nodes

class NeighborTraverser(BaseGraphTraverser):
    """
    (실험용) 단순히 Seed 노드의 n-hop 이웃들을 모두 가져오는 방식
    """
    def __init__(self, hops: int = 1):
        self.hops = hops
    
    def traverse(self, graph, seeds):
        result_nodes = set(seeds)
        current_boundary = set(seeds)

        for _ in range(self.hops):
            next_boundary = set()
            for node in current_boundary:
                if node in graph:
                    neighbors = graph.neighbors(node)
                    next_boundary.update(neighbors)
            
            result_nodes.append(next_boundary)
            current_boundary = next_boundary
        
        return result_nodes
