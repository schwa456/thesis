from abc import ABC, abstractmethod
import pcst_fast
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Set

from src.modules.node_selector import *
from src.modules.traverser import *
from src.modules.embedder import *

import logging

logger = logging.getLogger(__name__)

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, full_graph, question):
        pass

class GraphSchemaRetriever(BaseRetriever):
    def __init__(self,
                 embedder: SchemaEmbedder,
                 node_selector: BaseNodeSelector,
                 graph_traverser: BaseGraphTraverser):
        

        self.embedder = embedder
        self.selector = node_selector
        self.traverser = graph_traverser

        # Caching
        self.node_ids = []
    
    def index_graph(self, full_graph):
        """그래프 -> 텍스트 변환 -> Embedder에게 인덱싱 위임"""
        self.node_ids = []
        texts = []

        logger.debug("Converting Graph Nodes to Text...")
        for node, attr in full_graph.nodes(data=True):
            node_type = attr.get('type')

            if node_type == 'table':
                text = f"Table: {attr.get('name')}"
            elif node_type == 'column':
                col_name = attr.get('name')
                parent = attr.get('table')
                dtype = attr.get('dtype', '')
                text = f"Column: {col_name} in Table: {parent} ({dtype})"
                if 'values' in attr: 
                    text += f" (values: {attr['values']})"
            else:
                continue
                
            self.node_ids.append(node)
            texts.append(text)
            
        self.embedder.index_schema_nodes(texts, self.node_ids)
    
    def retrieve(self, full_graph, question) -> List[str]:
        """
        [Pipeline]: Score -> Select -> Refine -> Traverse -> Finalize
        """

        # 1. Scoring
        scores = self.embedder.get_similarity_scores(question)

        # 2. Selection
        seeds = self.selector.select_seed(scores, self.node_ids)
        logger.debug(f"SEED: {seeds}")

        # 3. Refinement (Graph Expansion)
        refined_seeds = set(seeds)
        for node in seeds:
            if node in full_graph:
                attr = full_graph.nodes[node]
                if attr.get('type') == 'column':
                    parent = attr.get('table')
                    if parent: refined_seeds.add(parent)
        
        fk_expanded_nodes = set(refined_seeds)

        for node in list(refined_seeds):
            if node in full_graph:
                for neighbor in full_graph.neighbors(node):
                    edge_data = full_graph.get_edge_data(node, neighbor)

                    if edge_data and edge_data.get('relation') == 'foreign_key':
                        fk_expanded_nodes.add(neighbor)

                        neighbor_attr = full_graph.nodes[neighbor]
                        if neighbor+attr.get('type') == 'column':
                            target_table= neighbor_attr.get('table')
                            if target_table:
                                fk_expanded_nodes.add(target_table)


        seed_list = list(fk_expanded_nodes)

        # 4, Traversal
        relevant_nodes = self.traverser.traverse(full_graph, seed_list)
        logger.debug(f"Relevant Nodes: {relevant_nodes}")

        # 5. Finalize (Mandatory PK Inclusion)
        final_nodes = set(relevant_nodes)
        for node in list(final_nodes):
            if node in full_graph and full_graph.nodes[node].get('type') == 'table':
                for n in full_graph.neighbors(node):
                    if full_graph.nodes[n].get('is_pk'):
                        final_nodes.add(n)
        
        return list(final_nodes)

class PCSTRetriever(BaseRetriever):
    def __init__(self, embedder, cost_e = 0.5):
        """
        Args:
            embedder: SchemaEmbedder Instance
            cost_e (float): 일반 edge connection cost (기본값 0.5)
        """

        self.embedder = embedder
        self.cost_e = cost_e

        self.node_to_idx = {}       # {node_id: int index} 
        

    def index_graph(self, G):
        """ Graph의 모든 Node를 Text로 변환 후 Embedding """
        logger.debug("Indexing Graph for PCST with Semantic Edges ...")

        nodes = list(G.nodes)
        node_texts = []
        node_ids = []

        self.node_to_idx = {n: i for i, n in enumerate(nodes)}

        for n in nodes:
            attr = G.nodes[n]
            if attr.get('type') == 'table':
                desc = attr.get('description') or ""
                text = f"Table {attr['name']}: {desc}"
            else:
                table_name = attr.get('table', '')
                desc = attr.get('description') or ""
                text = f"Column {attr['name']} in {table_name}: {desc}"
            
            edge_contexts = []
            for neighbor in G.neighbors(n):
                edge_data = G.get_edge_data(n, neighbor)

                if edge_data and 'textual_lable' in edge_data:
                    label = edge_data['textual_label']
                    edge_contexts.append(f"(Rel with {neighbor}: {label})")
            
            if edge_contexts:
                text += " | Context: " + ", ".join(edge_contexts)

            node_texts.append(text)
            node_ids.append(n)
        
        self.embedder.index_schema_nodes(node_texts, node_ids)

        logger.debug(f"Indexed {len(nodes)} nodes for PCST.")
    
    def retrieve(self, G, question, top_k=20):
        """
        PCST 알고리즘 활용 질문과 관련성 높으면서 연결된 Subgraph 반환
        """
        logger.debug(f"Running PCST Retrieval for: {question}")

        # 1. Prize Calculation
        scores_tensor = self.embedder.get_similarity_scores(question)
        prizes = scores_tensor.cpu().numpy()
        prizes = np.maximum(prizes, 0.0)
        

        # 2. Graph를 PCST 입력 형식으로 변환
        edges_list = []
        costs_list = []

        if not self.node_to_idx:
            self.index_graph(G)

        for u, v, data in G.edges(data=True):
            if u not in self.node_to_idx or v not in self.node_to_idx:
                continue

            relation = data.get('relation')

            if relation == 'table_foreign_key':
                current_cost = 0.05   # Table 간 직접 연결 장려
            elif relation == 'foreign_key':
                current_cost = 0.1  # 일반 FK
            elif relation in['contains', 'table_column']:
                current_cost = 0.1  # 구조적 정보
            else:
                current_cost = self.cost_e # 그 외
        

            u_idx = self.node_to_idx[u]
            v_idx = self.node_to_idx[v]

            edges_list.append([u_idx, v_idx])
            costs_list.append(current_cost)

        if not edges_list:
            logger.warning(f"No Edges Found in Graph. Falling back to Top-K.")
            edges = np.zeros((0, 2), dtype=np.int64)
            costs = np.zeros((0,), dtype=np.float64)
        
        else:
            edges = np.array(edges_list, dtype=np.int64)
            costs = np.array(costs_list, dtype=np.float64)


        # 4. PCST Fast
        # root=-1 (root 노드 고정 X), num_clusters=1 (1개의 연결된 Tree 반환)

        selected_indices = set()

        if edges.shape[0] > 0:
            try:
                vertices, _ = pcst_fast.pcst_fast(edges, prizes, costs, -1, 1, 'gw', 0)
                selected_indices.update(vertices)
            except Exception as e:
                logger.error(f"PCST Algorithm Failed: {e}. Faling back to top-k similarity.")
        
        top_k_indices = np.argsort(prizes)[-top_k:]
        selected_indices.update(top_k_indices)

        final_nodes = {self.embedder.cached_node_ids[i] for i in selected_indices}

        expanded_nodes = set(final_nodes)
        for node in final_nodes:
            if node in G:
                for neighbor in G.neighbors(node):
                    expanded_nodes.add(neighbor)
        
        # 5. Node ID List return
        logger.debug(f"PCST Selected {len(expanded_nodes)} nodes (connects disconnected components).")

        return expanded_nodes
