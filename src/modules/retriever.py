from abc import ABC, abstractmethod
import pcst_fast
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Set

from src.modules.node_selector import *
from src.modules.traverser import *
from src.modules.embedder import *
from src.modules.reranker import *

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

        self.node_ids = nodes
        self.node_to_idx = {n: i for i, n in enumerate(nodes)}
        self.idx_to_node = {i: n for i, n in enumerate(nodes)}

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
        
        self.embedder.index_schema_nodes(node_texts, self.node_ids)

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
                current_cost = 1.0   # Table 간 직접 연결 장려
            elif relation == 'foreign_key':
                current_cost = 2.0  # 일반 FK
            elif relation in['contains', 'table_column']:
                current_cost = 0.5  # 구조적 정보
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

class HybridPCSTRetriever(PCSTRetriever):
    """ Agent Seed Selection -> PCST Expansion """
    def __init__(self,
                 embedder: SchemaEmbedder,
                 node_selector: BaseNodeSelector,
                 cost_e: float=0.5,
                 agent_weight: float=10.0
                 ):
        
        super().__init__(embedder, cost_e)

        self.selector = node_selector
        self.agent_weight = agent_weight

        if not hasattr(self, 'node_ids'):
            self.node_ids = []
        if not hasattr(self, 'node_to_idx'):
            self.node_to_idx = {}
    
    def retrieve(self, G, question: str, top_k: int = 20) -> List[str]:
        logger.debug(f"[DEBUG] Retrieval Processing: {question}")

        # 0. Check Graph Indexing
        if not self.node_to_idx or not self.node_ids:
            logger.debug("Graph index not found. Building index now...")
            self.index_graph(G)
        
        # 1. Calculate Base Prizes (Embedding Score)
        scores_tensor = self.embedder.get_similarity_scores(question)
        if scores_tensor is not None:
            base_prizes = scores_tensor.cpu().numpy()
            base_prizes = np.maximum(base_prizes, 0.0)
        else:
            base_prizes = np.zeros(len(self.node_ids))

        # 2. Agent Seed Selection

        if not self.node_ids:
            logger.error("[ERROR] self.node_ids is empty even after indexing.")
            return []

        try:
            seed_result = self.selector.select_seed(scores_tensor, self.node_ids, question=question)
        except Exception as e:
            logger.warning(f"[WARNING] Agent Selection Failed: {e}. Fallback to Embedding only.")
            seed_result = {}
        
        # 2-1. Detect Unanswerable
        if not seed_result:
            logger.warning(f"[WARNING] Agent Found NO relevant tables. Stopping retrieval.")
            return []
        
        # 3. Hybrid Prize Scoring
        final_prizes = base_prizes.copy()

        seed_items = seed_result.items() if isinstance(seed_result, dict) else {k: 1.0 for k in seed_result}.items()

        logger.debug(f"[DEBUG] Hybrid Retrieval Seeds: {list(seed_items)}")

        for node_name, confidence in seed_items:
            if node_name in self.node_to_idx:
                idx = self.node_to_idx[node_name]
                boost_score = confidence * self.agent_weight
                final_prizes[idx] = max(final_prizes[idx], boost_score)
        
        # 4. Ready for PCST Input
        edges, costs, final_prizes = self._prepare_pcst_input_hybrid(G, final_prizes, question=question)

        if len(edges) == 0:
            return list(seed_result.keys())
        
        # 5. Run PCST Algorithm
        try:
            vertices, _ = pcst_fast.pcst_fast(edges, final_prizes, costs, -1, 1, 'gw', 0)
        except Exception as e:
            logger.error(f"[ERROR] PCST Failed: {e}. Returning Seeds Only.")
            return list(seed_result.keys())
        
        # 6. Mapping Result Nodes
        selected_nodes = {self.idx_to_node[i] for i in vertices}

        for s, _ in seed_items:
            selected_nodes.add(s)

        # 7. Mandatory Neighbor Node Expansion
        final_nodes = self._expand_neighbors(G, selected_nodes)

        logger.debug(f"[DEBUG] Hybrid Retrieval Final Nodes: {len(final_nodes)} (Graph Size Reduced)")
        return list(final_nodes)
    
    def _prepare_pcst_input_hybrid(self, G, prizes, question=None):
        # --- [A] Keyword Boosting ---
        # 질문 토큰화 (간단한 공백 기준 + 소문자화)
        question_tokens = set(question.lower().split()) if question else set()
        
        # 문장부호 제거 등 정규화가 필요할 수 있으나 여기선 약식으로 처리
        # prizes 배열은 numpy array 상태
        
        for i, n in enumerate(self.node_ids): # self.node_ids 순서와 prizes 순서 일치
            node_attr = G.nodes[n]
            node_name = node_attr.get('name', '').lower()
            original_name = node_attr.get('original_name', '').lower()
            
            # 질문의 단어가 컬럼/테이블 이름에 포함되어 있으면 보너스
            # (예: 'age' in 'student_age')
            # 글자 수 2개 이하인 단어(is, a, of 등)는 제외
            if any(token in node_name for token in question_tokens if len(token) > 2) or \
               any(token in original_name for token in question_tokens if len(token) > 2):
                
                prizes[i] += 0.5  # [Optimized] Keyword Match Bonus

        edges_list = []
        costs_list = []

        for u, v, data in G.edges(data=True):
            if u not in self.node_to_idx or v not in self.node_to_idx:
                continue

            u_idx = self.node_to_idx[u]
            v_idx = self.node_to_idx[v]

            relation = data.get('relation')

            if relation == 'table_foreign_key':
                cost = 0.2
            elif relation == 'foreign_key':
                cost = 2.0
            elif relation in ['contains', 'table_column', 'primary_key']:
                cost = 0.1
            else:
                cost = self.cost_e
            
            edges_list.append([u_idx, v_idx])
            costs_list.append(cost)

        if not edges_list:
            return np.zeros((0, 2), dtype=np.int64), np.zeros((0, ), dtype=np.float64), prizes
        
        return np.array(edges_list, dtype=np.int64), np.array(costs_list, dtype=np.float64), prizes
    
    def _expand_neighbors(self, G, nodes_set):
        expanded = set(nodes_set)
        for node in nodes_set:
            if node in G and G.nodes[node].get('type') == 'table':
                for neighbor in G.neighbors(node):
                    if G.nodes[neighbor].get('is_pk', False):
                        expanded.add(neighbor)
        return expanded
    
class RerankedPCSTRetriever(PCSTRetriever):
    """
    [NEW] Embedding Retrieval (Top-K) -> Reranking -> PCST Expansion
    - Score Scaling & Safety Net 적용으로 '빈 껍데기' 반환 방지
    """
    def __init__(self,
                 embedder: SchemaEmbedder,
                 reranker: SchemaReranker,
                 cost_e: float=0.2):
        
        super().__init__(embedder, cost_e)
        self.reranker = reranker
    
    def retrieve(self, G, question: str, top_k_candidates: int = 500) -> List[str]:
        # [수정 1] top_k_candidates 100 -> 500 (MiniLM의 놓침 방지)
        logger.debug(f"[RerankedPCST] Processing: {question}")

        if not self.node_to_idx:
            self.index_graph(G)
        
        # --- Phase 1: High-Recall Retrieval ---
        emb_scores = self.embedder.get_similarity_scores(question).cpu().numpy()
        top_indices = np.argsort(emb_scores)[-top_k_candidates:]
        
        candidate_texts = []
        candidate_indices = []
        
        for idx in top_indices:
            node_id = self.node_ids[idx]
            attr = G.nodes[node_id]
            
            if attr.get('type') == 'table':
                # 테이블 설명 강화
                text = f"Table {attr.get('name')}: {attr.get('description', '')}"
            else:
                # [핵심 수정] 컬럼 값(Values) 정보 복구!
                col_name = attr.get('name')
                table_name = attr.get('table', '')
                desc = attr.get('description', '')
                
                text = f"Column {col_name} in Table {table_name}: {desc}"
                
                # ★ Reranker에게 결정적 힌트 제공 (예: Values: ['Continuation School', ...])
                if 'values' in attr and attr['values']:
                    # 리스트인 경우 문자열로 변환
                    val_str = str(attr['values']) if isinstance(attr['values'], list) else str(attr['values'])
                    text += f" (Values: {val_str})"
                
                # (선택) 데이터 타입 힌트 추가
                if 'dtype' in attr:
                    text += f" [Type: {attr['dtype']}]"

            candidate_texts.append(text)
            candidate_indices.append(idx)
            
        # --- Phase 2: Precision Reranking ---
        rerank_scores = self.reranker.compute_scores(question, candidate_texts)
        
        # --- Phase 3: Score Scaling (핵심!) ---
        # Reranker 점수가 0.001 처럼 낮을 수 있으므로, 
        # 상위권 점수가 PCST Cost(0.05)를 충분히 넘도록 스케일링합니다.
        
        final_prizes = np.zeros(len(self.node_ids))
        max_score = max(rerank_scores) if rerank_scores else 0.0
        
        # Scaling Factor: 1등 점수가 무조건 1.0이 되도록 배율 조정 (최소 배율 1.0)
        scale_factor = 1.0 / max_score if max_score > 0 else 1.0
        
        # 너무 과도한 증폭 방지 (예: max가 0.0001일 때 10000배 튀는 것 방지)
        # 하지만 Cost를 뚫으려면 과감해야 함. 여기선 1등을 1.0으로 맞춤.
        
        top_reranked_indices = [] # Safety Net용

        for local_idx, raw_score in enumerate(rerank_scores):
            global_idx = candidate_indices[local_idx]
            
            # [수정 2] 점수 증폭 (Cost 0.05를 이기기 위함)
            # 1등은 1.0, 나머지는 비율대로.
            boosted_score = raw_score * scale_factor 
            final_prizes[global_idx] = boosted_score
            
        # Safety Net을 위해 정렬
        # (raw_score 기준 내림차순 정렬된 로컬 인덱스)
        sorted_local_indices = np.argsort(rerank_scores)[::-1]
        
        # --- Phase 4: PCST Input ---
        edges, costs, final_prizes = self._prepare_pcst_input_reranked(G, final_prizes)
        
        selected_nodes = set()
        
        # --- Phase 5: Run PCST ---
        if len(edges) > 0:
            try:
                # root=-1, num_clusters=1
                vertices, _ = pcst_fast.pcst_fast(edges, final_prizes, costs, -1, 1, 'gw', 0)
                for i in vertices:
                    selected_nodes.add(self.idx_to_node[i])
            except Exception as e:
                logger.error(f"PCST Failed: {e}")

        # --- Phase 6: Safety Net (Top-K 강제 포함) ---
        # [수정 3] PCST가 다 잘라버렸을 경우를 대비해, Reranker 상위 5개는 무조건 살림
        force_k = 5
        for i in range(min(force_k, len(sorted_local_indices))):
            local_idx = sorted_local_indices[i]
            global_idx = candidate_indices[local_idx]
            node_id = self.node_ids[global_idx]
            selected_nodes.add(node_id)
            
        # --- Phase 7: Expansion ---
        final_nodes = self._expand_neighbors(G, selected_nodes)
        
        logger.debug(f"[INFO] Final Nodes: {len(final_nodes)}")
        return list(final_nodes)

    def _prepare_pcst_input_reranked(self, G, prizes):
        # (기존과 동일하지만 Cost 확인)
        edges_list = []
        costs_list = []
        
        for u, v, data in G.edges(data=True):
            if u not in self.node_to_idx or v not in self.node_to_idx: continue
            
            u_idx = self.node_to_idx[u]
            v_idx = self.node_to_idx[v]
            relation = data.get('relation', 'generic')
            
            if relation == 'table_foreign_key': cost = 0.2
            elif relation == 'foreign_key': cost = 2.0
            
            # [중요] Reranker가 1.0(Top-1)을 주면 0.05 비용은 아주 쉽게 통과함
            elif relation in ['contains', 'table_column', 'primary_key']:
                cost = 0.05  
            else: cost = 0.2
            
            edges_list.append([u_idx, v_idx])
            costs_list.append(cost)
            
        return np.array(edges_list, dtype=np.int64), np.array(costs_list, dtype=np.float64), prizes
    
    def _expand_neighbors(self, G, nodes_set):
        # (기존과 동일)
        expanded = set(nodes_set)
        for node in nodes_set:
            if node in G and G.nodes[node].get('type') == 'table':
                for neighbor in G.neighbors(node):
                    if G.nodes[neighbor].get('is_pk', False):
                        expanded.add(neighbor)
        return expanded
    
class FastHybridRetriever(PCSTRetriever):
    """
    [Game Changer] Reranker + Agent Hybrid
    1. Reranker가 Noise를 제거하고 Top-N(30~50) 후보만 추출
    2. Agent(LLM)가 압축된 후보 중에서 '논리적 추론'을 통해 최종 선택
    -> 속도: 80초 -> 5초 (입력 토큰 대폭 감소)
    -> 정확도: Reranker의 정밀함 + Agent의 추론 능력 결합
    """
    def __init__(self,
                 embedder: SchemaEmbedder,
                 reranker: SchemaReranker,
                 node_selector: BaseNodeSelector,
                 agent_weight: float=10.0,
                 cost_e: float=0.2):
        
        super().__init__(embedder, cost_e)
        self.reranker = reranker
        self.selector = node_selector
        self.agent_weight = agent_weight
    
    def retrieve(self, G, question: str, top_k_rerank: int = 30) -> List[str]:
        logger.debug(f"[FastHybrid] Processing: {question}")
        
        if not self.node_to_idx:
            self.index_graph(G)

        # -------------------------------------------------------
        # Phase 1: Reranking (Candidate Filtering)
        # -------------------------------------------------------
        # 임베딩으로 500개 가져오기
        emb_scores = self.embedder.get_similarity_scores(question).cpu().numpy()
        top_indices = np.argsort(emb_scores)[-500:]
        
        candidate_texts = []
        candidate_indices = []
        
        for idx in top_indices:
            node_id = self.node_ids[idx]
            attr = G.nodes[node_id]
            # 텍스트 생성 (Values 포함 버전 권장)
            if attr.get('type') == 'table':
                text = f"Table {attr.get('name')}: {attr.get('description', '')}"
            else:
                text = f"Column {attr.get('name')} in {attr.get('table','')}: {attr.get('description', '')}"
                if 'values' in attr and attr['values']:
                    val_str = str(attr['values'])
                    text += f" (Values: {val_str})"
            
            candidate_texts.append(text)
            candidate_indices.append(idx)
            
        # Reranker 점수 계산
        rerank_scores = self.reranker.compute_scores(question, candidate_texts)
        
        # [핵심] Reranker 점수 상위 30개만 추출하여 Agent에게 전달
        # 이렇게 하면 Agent가 읽어야 할 양이 확 줄어듦
        zipped = sorted(zip(candidate_indices, rerank_scores), key=lambda x: x[1], reverse=True)
        top_n_indices = [idx for idx, score in zipped[:top_k_rerank]]
        
        # -------------------------------------------------------
        # Phase 2: Agent Reasoning (Logical Selection)
        # -------------------------------------------------------
        # Agent에게 보여줄 "Top-30 Node List" 준비
        filtered_node_ids = [self.node_ids[i] for i in top_n_indices]
        
        # Reranker 점수를 힌트로 줄 수도 있음 (여기선 생략하고 Agent 판단 존중)
        try:
            # Agent는 이제 30개만 보면 되므로 매우 빠름!
            seed_result = self.selector.select_seed(None, filtered_node_ids, question=question)
        except Exception as e:
            logger.warning(f"Agent Failed: {e}")
            seed_result = {}

        # -------------------------------------------------------
        # Phase 3: Hybrid Scoring & PCST
        # -------------------------------------------------------
        final_prizes = np.zeros(len(self.node_ids))
        
        # 1. Reranker Score를 Base Prize로 깔아줌 (Recall 보정)
        for i, (global_idx, score) in enumerate(zip(candidate_indices, rerank_scores)):
             final_prizes[global_idx] = score

        # 2. Agent Selection을 강력하게 반영 (Precision/Logic 보정)
        # Agent가 "District Name"을 찍었다면 점수 폭등
        for node_name, conf in seed_result.items():
            if node_name in self.node_to_idx:
                idx = self.node_to_idx[node_name]
                boost = conf * self.agent_weight
                # Reranker가 낮게 줬어도(0.1), Agent가 찍으면(10.0) 살아남음
                final_prizes[idx] = max(final_prizes[idx], boost)

        # 3. Run PCST
        edges, costs, final_prizes = self._prepare_pcst_input_reranked(G, final_prizes)
        
        # Safety Net (Agent 선택은 무조건 포함)
        force_nodes = set(seed_result.keys())
        
        if len(edges) > 0:
            try:
                vertices, _ = pcst_fast.pcst_fast(edges, final_prizes, costs, -1, 1, 'gw', 0)
                for i in vertices:
                    force_nodes.add(self.idx_to_node[i])
            except: pass
            
        final_nodes = self._expand_neighbors(G, force_nodes)
        
        logger.debug(f"[FastHybrid] Final Nodes: {len(final_nodes)}")
        return list(final_nodes)

    def _prepare_pcst_input_reranked(self, G, prizes):
        return super()._prepare_pcst_input_reranked(G, prizes) if hasattr(super(), '_prepare_pcst_input_reranked') else self._default_pcst_prep(G, prizes)

    def _default_pcst_prep(self, G, prizes):
        # 만약 상속 문제가 있다면 여기 구현 (이전 코드 참조)
        edges_list = []
        costs_list = []
        for u, v, data in G.edges(data=True):
             if u not in self.node_to_idx or v not in self.node_to_idx: continue
             relation = data.get('relation')
             if relation == 'table_foreign_key': c = 0.2
             elif relation == 'foreign_key': c = 2.0
             elif relation in ['contains', 'table_column', 'primary_key']: c = 0.05
             else: c = 0.2
             edges_list.append([self.node_to_idx[u], self.node_to_idx[v]])
             costs_list.append(c)
        return np.array(edges_list, dtype=np.int64), np.array(costs_list, dtype=np.float64), prizes