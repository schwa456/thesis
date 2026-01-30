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
        agent_server_url = config.get('agent_url', 'http://localhost:8000")

        # 1. Embedder
        self.embedder = SchemaEmbedder(model_name=config['embedder_model_id'])

        # 2. Graph Builder
        if config['graph_type'] == 'simple':
            self.graph_builder = SimpleFKGraphBuilder()
        elif config['graph_type'] == 'semantic':
            self.graph_builder = SemanticGraphBuilder()
        elif config['graph_type'] == 'shortcut':
            self.graph_builder = ShortcutGraphBuilder()

        logger.info(f"[INFO] Graph Builder Type: {config['graph_type']}")

        # 3. Node Selector
        if config['selector_type'] == 'adaptive':
            self.node_selector = AdaptiveSelector(
                alpha=config.get('alpha', 0.8),
                min_k=config.get('min_k', 2),
                max_k=config.get('max_k', 5)
                )
        elif config['selector_type'] == 'fixed':
            self.node_selector = FixedTopKSelector(k=config.get('top_k', 3))

        # 4. Traverser
        if config['traverser_type'] == 'shortest_path':
            self.traverser = ShortestPathTraverser(max_hops=config.get('max_hops', 2))
        elif config['traverser_type'] == 'neighbor':
            self.traverser = NeighborTraverser(hops=config.get('hops', 1))

        logger.info(f"[INFO] Graph Traverser Type: {config['traverser_type']}")
        
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

    def run(self, db_id, question, tables_metadata):
        # 1. Graph Build
        G, all_tables = self.graph_builder.build_graph(db_id, tables_metadata)

        # 2. Vector Search
        scores = self.embedder.get_similarity_scores(question, all_tables)

        # 3. Initial Node Selection
        seeds = self.node_selector.select_seed(scores, all_tables)

        # 4. Graph Expansion
        expanded_nodes = self.retriever.retrieve_subgraph(G, seeds)

        # 5. Agent Filtering
        final_schema = self.agent.filter_schema(question, expanded_nodes)

        return {
            "initial_seeds": seeds,
            "expanded_nodes": expanded_nodes,
            "final_schema": final_schema
        }
        
        
