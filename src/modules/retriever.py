import networkx as nx
import itertools
from abc import ABC, abstractmethod

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve_subgraph(self, G, seeds):
        pass

class ShortestPathRetriever(BaseRetriever):
    """ Seed Node 간의 최단 경로를 찾아 합집합 반환 """
    def __init__(self, max_hops=3):
        self.max_hops = max_hops

    def retrieve_subgraph(self, G, seeds):
        subgraph_nodes = set(seeds)

        if len(seeds) > 1:
            for src, tgt in itertools.combinations(seeds, 2):
                try:
                    if nx.has_path(G, src, tgt):
                        path = nx.shortest_path(G, src, tgt)
                        if len(path) <= self.max_hops + 1:
                            subgraph_nodes.update(path)
                except:
                    continue
        
        return list(subgraph_nodes)

class PCSTRetriever(BaseRetriever):
    """ PCST 기반 알고리즘 """
    def retrieve_subgraph(self, G, seeds):
        #TODO: pcst logic 구현 필요
        pass
    