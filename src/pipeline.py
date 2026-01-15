from src.modules.embedder import *
from src.modules.graph_builder import *
from src.modules.node_selector import *
from src.modules.retriever import *
from src.modules.agent import *

class Pipeline:
    def __init__(self, config):
        self.config = config

        # 1. Embedder
        self.embedder = SchemaEmbedder(model_name=config['embedder_model_id'])

        # 2. Graph Builder
        if config['graph_type'] == 'simple':
            self.graph_builder = SimpleFKGraphBuilder()
        elif config['graph_type'] == 'semantic':
            self.graph_builder = SemanticGraphBuilder()

        # 3. Node Selector
        if config['selector_type'] == 'adaptive':
            self.node_selector = AdaptiveSelector(alpha=config.get('alpha', 0.8))
        elif config['selector_type'] == 'fixed':
            self.node_selector = FixedTopKSelector(k=config.get('top_k', 3))
        
        # 4. Retriever
        if config['retriever_type'] == 'shortest_path':
            self.retriever = ShortestPathRetriever()
        elif config['retriever_type'] == 'pcst':
            self.retriever = PCSTRetriever()

        # 5. Agent
        self.agent = LlamaRejectionAgent(model_id=config['agent_model_id'])

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
        
        
