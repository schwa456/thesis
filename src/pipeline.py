from src.modules.embedder import *
from src.modules.graph_builder import *
from src.modules.node_selector import *
from src.modules.retriever import *
from src.modules.agent import *
from src.modules.traverser import *
from src.utils.graph_to_mschema import *

import logging

logger = logging.getLogger(__name__)

class Pipeline:
    def __init__(self, config):
        self.config = config
        self.graph_cache = {}
        self.current_db_id = None

        # 1. Agent
        agent_config = config.get('agent')
        if agent_config['usage']:
            server_url = agent_config.get('url')
            system_prompt_path = agent_config.get('system_prompt_path')
            user_prompt_path = agent_config.get('user_prompt_path')
            self.agent = FilteringAgent(server_url, system_prompt_path, user_prompt_path)
        else:
            self.agent = None

        logger.info(f"Agent Model ID: {config['agent']['model_id']}")

        # 2. Embedder
        embedder_model_id = config['embedder']['model_id']
        self.embedder = SchemaEmbedder(model_name=embedder_model_id)

        logger.info(f"Embedder Model: {embedder_model_id}")

        # 3. Graph Builder
        graph_type = config.get('graph_type', 'simple')

        if graph_type == 'simple':
            self.graph_builder = SimpleFKGraphBuilder(mode=config['data']['mode'])
        elif graph_type == 'semantic':
            self.graph_builder = SemanticGraphBuilder(self.config)
        elif graph_type == 'shortcut':
            self.graph_builder = ShortcutGraphBuilder()
        elif graph_type == 'none':
            self.graph_builder = SimpleFKGraphBuilder(mode=config['data']['mode'])

        logger.info(f"Graph Builder Type: {graph_type}")

        # 4. Node Selector
        selector_type = config.get('selector_type', 'adaptive')

        if selector_type == 'adaptive':
            self.node_selector = AdaptiveSelector(
                alpha=config.get('alpha', 0.8),
                min_k=config.get('min_k', 2),
                max_k=config.get('max_k', 5)
                )
        elif selector_type == 'fixed':
            self.node_selector = FixedTopKSelector(k=config.get('top_k', 3))
        elif selector_type == 'agent':
            self.node_selector = AgentNodeSelector(self.agent)
        elif selector_type == 'none':
            self.node_selector = None

        logger.info(f"Node Selector Type: {config['selector_type']}")
        
        # 4. Traverser
        traverser_type = config.get('traverser_type', 'shortest')

        if traverser_type == 'shortest_path':
            self.traverser = ShortestPathTraverser(max_hops=config.get('max_hops', 2))
        elif traverser_type == 'neighbor':
            self.traverser = NeighborTraverser(hops=config.get('hops', 1))
        elif traverser_type == 'none':
            self.traverser = None
        
        logger.info(f"Graph Traverser Type: {config['traverser_type']}")

        # 5. Retriever
        retriever_type = config.get('retriever_type', 'pcst')

        if retriever_type == 'graph_schema':
            self.retriever = GraphSchemaRetriever(
                embedder=self.embedder,
                node_selector=self.node_selector,
                graph_traverser=self.traverser
            )
        elif retriever_type == 'pcst':
            self.retriever = PCSTRetriever(
                embedder=self.embedder
            )
        elif retriever_type == 'hybrid':
            if not isinstance(self.node_selector, AgentNodeSelector):
                logger.warning(f"[WARNING] Hybrid Retriever Suggests using 'agent' selector, but current is {selector_type}")
            self.retriever = HybridPCSTRetriever(
                embedder=self.embedder,
                node_selector=self.node_selector,
                cost_e = config.get('pcst_cost', 0.5),
                agent_weight=config.get('agent_weight', 10.0)
            )
        elif retriever_type == 'none':
            self.retriever = None

        logger.info(f"Retriever Type: {config['retriever_type']}")


    def _load_graph_context(self, db_id, db_engine, bird_meta):
        """
        1. 요청된 db_id에 맞는 그래프를 로드합니다. (메모리 캐시 -> 파일 캐시 -> 빌드 순)
        2. DB가 변경된 경우 Retriever에게 새로운 그래프를 인덱싱하도록 지시합니다.
        """

        if db_id in self.graph_cache:
            G = self.graph_cache[db_id]
        else:
            logger.debug(f"[Context Switch] Building Graph for DB: {db_id}")

            if isinstance(self.graph_builder, SemanticGraphBuilder):
                cache_dir = "./data/cache_graphs"
                os.makedirs(cache_dir, exist_ok=True)
                self.graph_builder.cache_path = os.path.join(cache_dir, f"{db_id}_semantic.pkl")
            
            G, _ = self.graph_builder.build_graph(db_engine, db_id, bird_meta)
            self.graph_cache[db_id] = G
        
        if self.current_db_id != db_id:
            logger.debug(f"Indexing Graph for Retriever (DB: {db_id})")
            self.retriever.index_graph(G)
            self.current_db_id = db_id
        
        return G
    
    def _boost_recall_neighbor_graph(self, G, current_nodes, max_neighbors=5):
        """
        [Revised Strategy] Full Column Retention
        Agent가 선택한 테이블이라면, 룰베이스로 필터링하지 않고 '모든 컬럼'을 Generator에게 전달합니다.
        """
        # 1. Seed Table 식별
        seed_tables = set()

        for node in current_nodes:
            if node not in G: continue
            if G.nodes[node].get('type') == 'table':
                seed_tables.add(node)
            elif G.nodes[node].get('type') == 'column':
                parent = G.nodes[node].get('table')
                if parent: seed_tables.add(parent)
        
        # 2. 이웃 테이블 탐색 (Safe Neighbors via FK)
        #    - Seed 테이블과 FK로 직접 연결된 테이블만 제한적으로 가져옵니다.
        neighbor_candidates = set()
        for table in list(seed_tables):
            if table not in G: continue
            for col in G.neighbors(table): 
                for connected_col in G.neighbors(col):
                    edge_data = G.get_edge_data(col, connected_col)
                    if edge_data and edge_data.get('relation') == 'foreign_key':
                        target_table = G.nodes[connected_col].get('table')
                        if target_table and target_table not in seed_tables:
                            neighbor_candidates.add(target_table)

        # 이웃 개수 제한 (Hub Node 방지)
        final_neighbors = list(neighbor_candidates)
        if len(final_neighbors) > max_neighbors:
            final_neighbors = sorted(final_neighbors, key=lambda x: len(x))[:max_neighbors]
        
        # 최종 대상 테이블 확정
        all_target_tables = seed_tables.union(set(final_neighbors))
        
        # 3. Full Column Expansion (No more hard-coding)
        final_set = set(current_nodes) 

        for table in all_target_tables:
            if table not in G: continue
            
            # 테이블 노드 추가
            final_set.add(table)

            # [핵심 변경] 해당 테이블의 '모든' 컬럼을 추가
            # LLM이 직접 보고 판단하도록 정보 손실을 막음
            for col_node in G.neighbors(table):
                # 엣지 타입이 'contains'인 것만 컬럼임 (혹시 모를 다른 연결 배제)
                edge_data = G.get_edge_data(table, col_node)
                
                # 보통 G.neighbors는 방향성 없이 가져올 수 있으므로 타입 체크
                if G.nodes[col_node].get('type') == 'column':
                    final_set.add(col_node)

        logger.debug(f"[Context Expansion] Selected {len(all_target_tables)} tables. Retaining ALL columns ({len(final_set)} nodes) for Generator.")
        return final_set

    def _filter_by_agent(self, G, question, retrieved_nodes_list, evidence):
        if not retrieved_nodes_list:
            logger.warning("No nodes retrieved.")
            final_items = []
        
        else:
            retrieved_subgraph = G.subgraph(retrieved_nodes_list)
            try:
                final_items = self.agent.filter_schema(question, retrieved_subgraph, evidence)
            except Exception as e:
                logger.error(f"Agent Logic Failed: {e}")
                final_items = []
        
        if final_items is None or final_items is Ellipsis:
            logger.warning("Agent returned None/Ellipses. Fallback to Retrieval results.")
            final_items = []

        if not final_items:
            logger.debug("Agent selected nothing. Using all retrieved nodes.")
            final_items = retrieved_nodes_list
        
        return final_items
    
    def _formatting_filtered_result(self, G, final_items, retrieved_nodes_list, db_engine, db_id, bird_meta=None):
        graph_node_lookup = {n.lower(): n for n in G.nodes}
        
        final_nodes_set = set()

        # 4-1. Agent 선택 항목 Mapping
        for item in final_items:
            if not isinstance(item, str):
                logger.warning(f"Ignored invalid item type: {type(item)} - {item}")
                continue

            if item in G:
                final_nodes_set.add(item)
                continue

            item_lower = item.lower()
            if item_lower in graph_node_lookup:
                real_node_name = graph_node_lookup[item_lower]
                final_nodes_set.add(real_node_name)
                logger.debug(f"Matches (Case-Insensitive): {item} -> {real_node_name}")
                continue
            
            elif "." in item:
                table_part = item.split(".")[0]
                table_lower = table_part.lower()

                if table_part in G:
                    final_nodes_set.add(table_part)
                elif table_lower in graph_node_lookup:
                    real_table_name = graph_node_lookup[table_lower]
                    final_nodes_set.add(real_table_name)
                    logger.debug(f"Table Fallback: {item} -> {real_table_name}")
                else:
                    logger.warning(f"Ignored Hallucinated Item: {item}")
        
        if not final_nodes_set: 
            logger.warning(f"[WARNING] Agent selected invalid nodes. Falling back to retrieval results.")
            final_nodes_set = set(retrieved_nodes_list)

        # 4-2. Bridge Logic: 선택된 Table 사이의 FK 탐색
        current_tables = {n for n in final_nodes_set if G.nodes[n].get('type') == 'table'}

        for node in list(final_nodes_set):
            if G.nodes[node].get('type') == 'column':
                parent = G.nodes[node].get('table')
                if parent: current_tables.add(parent)

        for table in current_tables:
            for col_node in G.neighbors(table):
                for neighbor_col in G.neighbors(col_node):
                    edge_data = G.get_edge_data(col_node, neighbor_col)

                    if edge_data and edge_data.get('relation') == 'foreign_key':
                        target_table = G.nodes[neighbor_col].get('table')

                        if target_table in current_tables:
                            final_nodes_set.add(col_node)
                            final_nodes_set.add(neighbor_col)
                            final_nodes_set.add(table)
                            final_nodes_set.add(target_table)

        # 4-3. Node Expansion: 선택된 테이블의 모든 컬럼 활용
        expanded_nodes = self._boost_recall_neighbor_graph(G, final_nodes_set)

        logger.debug(f"Final Node Count: {len(final_nodes_set)} -> {len(expanded_nodes)}")
        
        # 최종 Subgraph 생성
        if not expanded_nodes:
            final_focused_graph = nx.Graph()
        else:
            final_focused_graph = G.subgraph(list(expanded_nodes)).copy()
        
        # M-Schema 문자열 반환
        schema_string = graph_to_mschema(final_focused_graph, db_engine, db_id=db_id, bird_meta=bird_meta)

        return schema_string, expanded_nodes

    def run(self, question, db_id, db_engine, evidence=None, bird_meta=None, status_callback=None):
        # 1. Graph Build
        if status_callback: status_callback("Graph Check")
        if self.retriever:
            G = self._load_graph_context(db_id, db_engine, bird_meta)
        else:
            G, tables = self.graph_builder.build_graph(db_engine, db_id)
            schema_string = graph_to_mschema(G, db_engine, db_id)

        # Path A: Hybrid Retrieval (Agent Seed -> PCST)
        if isinstance(self.retriever, HybridPCSTRetriever):
            if status_callback: status_callback("Hybrid Retrieval")

            retrieved_nodes_list = self.retriever.retrieve(G, question)

            expanded_nodes = self._boost_recall_neighbor_graph(G, retrieved_nodes_list)

            schema_string = nodes_to_mschema(expanded_nodes, G, db_engine, db_id, bird_meta)
            return {
                "question": question,
                "retrieved_nodes": retrieved_nodes_list,    # 1차 검색 결과
                "selected_items": expanded_nodes,              # Agent가 Filtering 한 것
                "final_schema_str": schema_string           # 최종 SQL 생성용 String
            }

        else:
            # Path B: Retrieval -> Agent Filtering

            # 2. Schema Retrieval
            if status_callback: status_callback("Retrieving")
            if self.retriever:
                retrieved_nodes_list = self.retriever.retrieve(G, question)
            else: 
                retrieved_nodes_list = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'column']

            logger.debug(f"Retrieved Nodes ({len(retrieved_nodes_list)}): {retrieved_nodes_list}")
        
            # 3. Agent Filtering
            if status_callback: status_callback("Agent Filtering")
            if self.agent:
                final_items = self._filter_by_agent(G, question, retrieved_nodes_list, evidence)
            else: pass

            # 4. Result Formatting
            # Agent가 선택한 아이템을 다시 Graph로 확장
            if status_callback: status_callback("Formatting")
            if self.agent and self.config['agent']['formatting']:
                schema_string, expanded_nodes = self._formatting_filtered_result(G, final_items, retrieved_nodes_list, db_engine, db_id, bird_meta)

                if status_callback: status_callback("Done")
                logger.debug(f"Filtered Schema:\n{schema_string}")
            elif self.agent and not self.config['agent']['formatting']:
                expanded_nodes = final_items
                schema_string = nodes_to_mschema(expanded_nodes, G, db_engine, db_id, bird_meta)  
            else:
                expanded_nodes = retrieved_nodes_list
                schema_string = nodes_to_mschema(retrieved_nodes_list, G, db_engine, db_id, bird_meta)

            return {
                "question": question,
                "retrieved_nodes": retrieved_nodes_list,    # 1차 검색 결과
                "selected_items": expanded_nodes,              # Agent가 Filtering 한 것
                "final_schema_str": schema_string           # 최종 SQL 생성용 String
            }
