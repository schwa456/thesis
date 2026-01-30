from src.modules.embedder import *
from src.modules.graph_builder import *
from src.modules.node_selector import *
from src.modules.retriever import *
from src.modules.agent import *
from src.modules.traverser import *
from src.utils.graph_to_mschema import graph_to_mschema

import logging

logger = logging.getLogger(__name__)

class Pipeline:
    def __init__(self, config):
        self.config = config
        self.graph_cache = {}
        self.current_db_id = None
        agent_server_url = config.get('agent_url', 'http://localhost:8000')

        # 1. Embedder
        self.embedder = SchemaEmbedder(model_name=config['embedder_model_id'])

        logger.info(f"Embedder Model: {config['embedder_model_id']}")

        # 2. Graph Builder
        if config['graph_type'] == 'simple':
            self.graph_builder = SimpleFKGraphBuilder()
        elif config['graph_type'] == 'semantic':
            self.graph_builder = SemanticGraphBuilder()
        elif config['graph_type'] == 'shortcut':
            self.graph_builder = ShortcutGraphBuilder()

        logger.info(f"Graph Builder Type: {config['graph_type']}")

        # 3. Node Selector
        if config['selector_type'] == 'adaptive':
            self.node_selector = AdaptiveSelector(
                alpha=config.get('alpha', 0.8),
                min_k=config.get('min_k', 2),
                max_k=config.get('max_k', 5)
                )
        elif config['selector_type'] == 'fixed':
            self.node_selector = FixedTopKSelector(k=config.get('top_k', 3))

        logger.info(f"Node Selector Type: {config['selector_type']}")
        
        # 4. Traverser
        if config['traverser_type'] == 'shortest_path':
            self.traverser = ShortestPathTraverser(max_hops=config.get('max_hops', 2))
        elif config['traverser_type'] == 'neighbor':
            self.traverser = NeighborTraverser(hops=config.get('hops', 1))
        
        logger.info(f"Graph Traverser Type: {config['traverser_type']}")

        # 5. Retriever
        if config['retriever_type'] == 'graph_schema':
            self.retriever = GraphSchemaRetriever(
                embedder=self.embedder,
                node_selector=self.node_selector,
                graph_traverser=self.traverser
            )
        elif config['retriever_type'] == 'pcst':
            self.retriever = PCSTRetriever(
                embedder=self.embedder
            )

        logger.info(f"Retriever Type: {config['retriever_type']}")

        # 6. Agent
        self.agent = FilteringAgent(agent_server_url)

        logger.info(f"Agent Model ID: {config['agent_model_id']}")
    
    def _load_graph_context(self, db_id, db_engine):
        """
        1. 요청된 db_id에 맞는 그래프를 로드합니다. (메모리 캐시 -> 파일 캐시 -> 빌드 순)
        2. DB가 변경된 경우 Retriever에게 새로운 그래프를 인덱싱하도록 지시합니다.
        """

        if db_id in self.graph_cache:
            G = self.graph_cache[db_id]
        else:
            logger.info(f"[Context Switch] Building Graph for DB: {db_id}")

            if isinstance(self.graph_builder, SemanticGraphBuilder):
                cache_dir = "./data/cache_graphs"
                os.makedirs(cache_dir, exist_ok=True)
                self.graph_builder.cache_path = os.path.join(cache_dir, f"{db_id}_semantic.pkl")
            
            G, _ = self.graph_builder.build_graph(db_engine)
            self.graph_cache[db_id] = G
        
        if self.current_db_id != db_id:
            logger.info(f"Indexing Graph for Retriever (DB: {db_id})")
            self.retriever.index_graph(G)
            self.current_db_id = db_id
        
        return G
    
    def _boost_recall_neighbor_graph(self, G, current_nodes, max_neighbors=5):
        # 1. Seed Table 식별
        seed_tables = set()
        agent_selected_columns = set() # Agent가 명시적으로 고른 건 절대 버리지 않음

        for node in current_nodes:
            if G.nodes[node].get('type') == 'table':
                seed_tables.add(node)
            elif G.nodes[node].get('type') == 'column':
                agent_selected_columns.add(node)
                parent = G.nodes[node].get('table')
                if parent: seed_tables.add(parent)
        
        logger.debug(f"Seed Tables: {len(seed_tables)} tables")

        # 2. 이웃 테이블 탐색 (Safe Neighbors)
        neighbor_candidates = set()
        
        for table in list(seed_tables): # [중요] list()로 복사해서 순회
            if table not in G: continue
            
            # FK로 연결된 테이블 찾기
            for col in G.neighbors(table): 
                for connected_col in G.neighbors(col):
                    edge_data = G.get_edge_data(col, connected_col)
                    if edge_data and edge_data.get('relation') == 'foreign_key':
                        target_table = G.nodes[connected_col].get('table')
                        if target_table and target_table not in seed_tables:
                            neighbor_candidates.add(target_table)

        # 이웃 개수 제한 (옵션: 너무 많으면 Hub Node일 가능성 있음)
        final_neighbors = list(neighbor_candidates)
        if len(final_neighbors) > max_neighbors:
             # 간단한 휴리스틱: 이름이 짧은 순(코드성 테이블일 확률 높음) or 가나다순
            final_neighbors = sorted(final_neighbors, key=lambda x: len(x))[:max_neighbors]
            logger.warning(f"Capped neighbors at {max_neighbors}")
        
        all_target_tables = seed_tables.union(set(final_neighbors))
        
        # 3. Smart Column Filtering (쓰레기 거르기)
        final_set = set()
        
        # (1) Agent가 선택한 건 무조건 포함 (Base)
        final_set.update(current_nodes) 

        for table in all_target_tables:
            if table not in G: continue
            
            # 테이블 노드 자체는 포함
            final_set.add(table)

            # 컬럼 선별 작업
            for col_node in G.neighbors(table):
                # 이미 선택된 건 패스
                if col_node in final_set: continue
                
                col_name = col_node.split(".")[-1].upper() # Table.Column -> COLUMN
                col_data = G.nodes[col_node]

                # Rule 1: Structural (PK, FK)
                if col_data.get('is_primary') or col_data.get('is_foreign'):
                    final_set.add(col_node)
                    continue
                
                # Rule 2: Validity (DLT_FLG)
                if "DLT_FLG" in col_name:
                    final_set.add(col_node)
                    continue

                # Rule 3: Analysis Keywords (Name, Date, Value)
                # 이 키워드들이 포함된 컬럼만 가져옴
                keywords = ["_NM", "_NAME", "NAME",             # 명칭
                            "_DT", "_DATE", "YMD", "_MON",      # 날짜
                            "_DATA", "_AMT", "_CNT", "PRICE",   # 수치
                            "_SEQ"]                             # PK ID
                
                if any(k in col_name for k in keywords):
                    final_set.add(col_node)
                    continue
                
                # 나머지는 버림 (DESC, RMK, REG_ID, UPD_DT, ADDR 등)

        logger.debug(f"Smart Expansion: {len(current_nodes)} -> {len(final_set)} nodes (Target Tables: {len(all_target_tables)})")
        return final_set


    def run(self, question, db_id, db_engine, status_callback=None):
        # 1. Graph Build
        if status_callback: status_callback("Graph Check")
        G = self._load_graph_context(db_id, db_engine)

        initial_schema_str = graph_to_mschema(G, db_engine)
        with open('schema_str.txt', 'w', encoding='utf-8-sig') as f:
            f.write(initial_schema_str)

        # 2. Schema Retrieval
        if status_callback: status_callback("Retrieving")
        retrieved_nodes_list = self.retriever.retrieve(G, question)

        logger.debug(f"Retrieved Nodes ({len(retrieved_nodes_list)}): {retrieved_nodes_list}")
    
        # 3. Agent Filtering
        if status_callback: status_callback("Agent Filtering")
        if not retrieved_nodes_list:
            logger.warning("No nodes retrieved.")
            final_items = []
        
        else:
            retrieved_subgraph = G.subgraph(retrieved_nodes_list)
            try:
                final_items = self.agent.filter_schema(question, retrieved_subgraph)
            except Exception as e:
                logger.error(f"Agent Logic Failed: {e}")
                final_items = []
        
        if final_items is None or final_items is Ellipsis:
            logger.warning("Agent returned None/Ellipses. Fallback to Retrieval results.")
            final_items = []

        if not final_items:
            logger.info("Agent selected nothing. Using all retrieved nodes.")
            final_items = retrieved_nodes_list

        # 4. Result Formatting
        # Agent가 선택한 아이템을 다시 Graph로 확장
        if status_callback: status_callback("Formatting")

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
            if status_callback: status_callback("Agent Failed -> Using Retrieval")  
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
        schema_string = graph_to_mschema(final_focused_graph, db_engine)

        if status_callback: status_callback("Done")
        logger.debug(f"Filtered Schema:\n{schema_string}")

        return {
            "question": question,
            "retrieved_nodes": retrieved_nodes_list,    # 1차 검색 결과
            "selected_items": expanded_nodes,              # Agent가 Filtering 한 것
            "final_schema_str": schema_string           # 최종 SQL 생성용 String
        }
