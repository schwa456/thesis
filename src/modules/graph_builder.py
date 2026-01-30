import os
import difflib
import logging
import pickle
import openai
import networkx as nx
from tqdm import tqdm
from abc import ABC, abstractmethod
from sqlalchemy import inspect, create_engine
from openai import AzureOpenAI


logger = logging.getLogger(__name__)

class BaseGraphBuilder(ABC):
    @abstractmethod
    def build_graph(self, db_engine):
        pass

class SimpleFKGraphBuilder(BaseGraphBuilder):
    """ 
    [Node]: Table, Column 
    [Edge]: Table-Column(contains), Column-Column(FK)
    PK/FK 구조와 comment를 포함하는 상세 그래프 빌더
    """
    def __init__(self):
        self.manual_pks = {

        }

        self.manual_fks = [

        ]
    def build_graph(self, db_engine):
        logger.info("Building Knowledge Graph...")

        G = nx.Graph()
        inspector = inspect(db_engine)
        tables = inspector.get_table_names()

        # 1. Node 생성 (Table & Column)
        for table_name in tables:
            # 1-1. Table Node
            try:
                table_comment = inspector.get_table_comment(table_name).get('text')
            except:
                table_comment = None
            
            G.add_node(table_name, type='table', name=table_name, description=table_comment)

            # 1-2. Column Node
            try:
                columns = inspector.get_columns(table_name)
                pk_constraint = inspector.get_pk_constraint(table_name)
                real_pks = pk_constraint.get('constrained_columns', [])

                manual_pk_col = self.manual_pks.get(table_name)

                inferred_pk = None
                if not manual_pk_col and not real_pks:
                    inferred_pk = self._infer_primary_key(table_name, [c['name'] for c in columns])

                for col in columns:
                    col_name = col['name']
                    col_node_id = f"{table_name}.{col_name}"
                    col_type = str(col['type'])
                    col_comment = col.get('comment')

                    is_pk=False

                    if manual_pk_col:
                        if col_name == manual_pk_col:
                            is_pk = True
                    elif col_name in real_pks:
                        is_pk = True
                    elif inferred_pk and col_name == inferred_pk:
                        is_pk = True
                    

                    G.add_node(
                        col_node_id,
                        type='column',
                        table=table_name,
                        name=col_name,
                        dtype=col_type,
                        is_pk=is_pk,
                        description=col_comment
                    )

                    # 1-3. Edge: Table -> Column (contains)
                    G.add_edge(table_name, col_node_id, relation='containts')

            except Exception as e:
                logger.error(f"[ERROR] Processing columns for table {table_name}: {e}")
        
        # 2. Edge 생성 (FK)
        self._build_relationships(G, tables, inspector)
        
        
        logger.debug(f"Graph Built: {len(G.nodes)} nodes, {len(G.edges)} edges")

        logger.debug(f"Initial Schema: {G}")

        return G, tables

    def _infer_primary_key(self, table_name, column_names):
        candidates = [c for c in column_names if c.endswith("_SEQ") or c.endswith("_ID") or c.upper() == "ID"]

        if not candidates:
            return None
        
        if len(candidates) == 1:
            return candidates[0]
        
        best_match = difflib.get_close_matches(table_name.upper(), candidates, n=1, cutoff=0.0)

        if best_match:
            logger.debug(f"[Smart PK] Inferred PK for {table_name}: {best_match[0]}")
            return best_match[0]

        return candidates[0]
        
    def _build_relationships(self, G, tables, inspector):
        logger.debug(f"Applying {len(self.manual_fks)} Manual FK rules...")

        # 1. DB 정보 기반
        for table_name in tables:
            try:
                fks = inspector.get_foreign_keys(table_name)
                for fk in fks:
                    for loc, rem in zip(fk['constrained_columns'], fk['referred_columns']):
                        self._add_edge(G, table_name, loc, fk['referred_table'], rem)
            except: pass
    
        # 2. Manual Mapping 기반
        for src_tbl, src_col, tgt_tbl, tgt_col in self.manual_fks:
            # 테이블 존재 여부 확인
            if src_tbl not in tables:
                logger.warning(f"⚠️ [FK Fail] Source Table '{src_tbl}' not found in DB tables: {tables}")
                continue
            
            if tgt_tbl not in tables:
                logger.warning(f"⚠️ [FK Fail] Target Table '{tgt_tbl}' not found in DB tables.")
                continue

            self._add_edge_debug(G, src_tbl, src_col, tgt_tbl, tgt_col)
    
    def _add_edge(self, G, t1, c1, t2, c2, is_virtual=False):
        u, v = f"{t1}.{c1}", f"{t2}.{c2}"
        if G.has_node(u) and G.has_node(v):
            G.add_edge(u, v, relation='foreign_key')

    def _add_edge_debug(self, G, t1, c1, t2, c2):
            u = f"{t1}.{c1}"
            v = f"{t2}.{c2}"
            
            # 노드가 실제로 그래프에 있는지 확인
            u_exists = G.has_node(u)
            v_exists = G.has_node(v)

            if u_exists and v_exists:
                G.add_edge(u, v, relation='foreign_key')
                logger.debug(f"[FK Liked] {u} <-> {v}")
            else:
                # 실패 원인 상세 로그
                if not u_exists:
                    # 대소문자 문제인지 힌트 제공
                    logger.warning(f"🚫 [FK Fail] Node '{u}' NOT FOUND. (Is the column name correct?)")
                    # 해당 테이블의 실제 컬럼들 보여주기
                    real_cols = [n for n in G.neighbors(t1)]
                    logger.warning(f"   -> Available nodes in {t1}: {real_cols}")
                    
                if not v_exists:
                    logger.warning(f"🚫 [FK Fail] Node '{v}' NOT FOUND.")
                    real_cols = [n for n in G.neighbors(t2)]
                    logger.warning(f"   -> Available nodes in {t2}: {real_cols}")

class ShortcutGraphBuilder(SimpleFKGraphBuilder):
    def build_graph(self, db_engine):
        G, tables = super().build_graph(db_engine)

        table_shortcuts = set()

        for u, v, data in list(G.edges(data=True)):
            if data.get('relation') == 'foreign_key':
                u_attr = G.nodes[u]
                v_attr = G.nodes[v]

                t1 = u_attr.get('table') or u.split('.')[0]
                t2 = v_attr.get('table') or v.split('.')[0]

                if t1 and t2 and t1 != t2:
                    edge_pair = tuple(sorted([t1, t2]))
                    table_shortcuts.add(edge_pair)
        
        added_count = 0
        for t1, t2 in table_shortcuts:
            G.add_edge(
                t1, t2,
                relation='table_foreign_key',
                weight=1.0,
                type='shortcut'
            )
            added_count += 1
        
        logger.info(f"Added {added_count} Shortcut Edges between Tables.")

        return G, tables

class SemanticGraphBuilder(ShortcutGraphBuilder):
    def __init__(self, config, cache_path="./data/cache_graphs"):
        """
        Args:
            cache_path: 생성된 그래프를 저장/로드할 경로
        """
        super().__init__()
        self.cache_path = cache_path
        self.config = config

    def get_db_engine_for_db(self, db_id):
        mode = self.config['data']['mode']

        if mode == 'bird':
            db_root = self.config['data']['bird']['db_root_path']
            db_path = os.path.join(db_root, db_id, f"{db_id}.sqlite")
            return create_engine(f"sqlite:///{db_path}")

        elif mode == "lg55":
            return create_engine(self.config['data']['lg55']['db_uri'])

        return None

    def build_graph(self, db_engine):
        if os.path.exists(self.cache_path):
            logger.debug(f"Loading Cached Semantic Graph from {self.cache_path}...")
            with open(self.cache_path, 'rb') as f:
                saved_data = pickle.load(f)
                self.G = saved_data['G']
                return self.G, saved_data['tables']

        G, tables = super().build_graph(db_engine)


        logger.info("Generating Semantic Description for Edges (This may take a while)...")
        self._enrich_edges_with_llm(G)

        
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'wb') as f:
            pickle.dump({'G': G, 'tables': tables}, f)
        logger.info(f"Graph cached to {self.cache_path}")

        return G, tables
    
    def _enrich_edges_with_llm(self, G):
        """
        Graph의 모든 FK Edge를 순회하며 Semantic Desciption 추가
        """
        
        edges_to_process = []

        # FK 관계인 Edge만 추출
        for u, v, data in G.edges(data=True):
            if data.get('relation') in ['foreign_key', 'table_foreign_key']:
                edges_to_process.append((u, v, data))
        
        logger.info(f"Processing {len(edges_to_process)} edges for semantic labeling...")
        
        for u, v, data in tqdm(edges_to_process):
            if 'textual_label' in data:
                continue
                
            u_node = G.nodes[u]
            v_node = G.nodes[v]

            prompt = self._create_edge_prompt(u_node, v_node, data)

            try:
                description = self._call_llm(prompt)

                G[u][v]['textual_label'] = description
            
            except Exception as e:
                logger.error(f"Failed to generate label for {u} -> {v}: {e}")
            
    
    def _create_edge_prompt(self, node_a, node_b, edge_data):
        return f"""
        Analyze the database relationship between these two schema elements:
        
        1. Source: {node_a['name']} (Type: {node_a['type']}, Desc: {node_a.get('description', 'N/A')})
        2. Target: {node_b['name']} (Type: {node_b['type']}, Desc: {node_b.get('description', 'N/A')})
        3. Relation Type: {edge_data.get('relation')}
        
        Task: Write a VERY SHORT (5-10 words) semantic description of this relationship.
        Example: "Users place orders" or "Order items belong to an order".
        Output ONLY the description.
        """
    
    def _call_llm(self, prompt):
        gpt_config = self.config.get('gpt_config', {})

        model_deployment_name = gpt_config.get('model_deployment_name', '')
        end_point = gpt_config.get('end_point', '')
        api_key = gpt_config.get('api_key', '')
        api_version = gpt_config.get('api_version', '')

        client = AzureOpenAI(
            azure_endpoint = end_point,
            api_key=api_key,
            api_version=api_version
        )

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        completion = client.chat.completions.create(
            model=model_deployment_name,
            messages=messages,
            temperature=0.0,
        )

        return completion.choices[0].message.content.strip()
