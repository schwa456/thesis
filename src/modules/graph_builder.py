import os
import difflib
import logging
import pickle
import asyncio
import networkx as nx
from tqdm import tqdm
from abc import ABC, abstractmethod
from sqlalchemy import inspect, create_engine
from openai import AsyncOpenAI, RateLimitError, APIError
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

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
    def __init__(self, mode):
        self.mode = mode
        self.manual_pks = {}
        self.manual_fks = []

        if mode == 'lg55':
            self.manual_pks = {

            }

            self.manual_fks = [

            ]

    def build_graph(self, db_engine, db_id=None, bird_meta=None):
        logger.debug(f"Building Knowledge Graph for {db_id}...")

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
                    G.add_edge(table_name, col_node_id, relation='contains')

            except Exception as e:
                logger.error(f"[ERROR] Processing columns for table {table_name}: {e}")
        
        # 2. Edge 생성 (FK)
        self._build_relationships(G, tables, inspector)

        # 3. BIRD Metadata 활용
        if self.mode == 'bird' and bird_meta:
            self._apply_bird_metadata(G, bird_meta)
        
        
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
    
    def _apply_bird_metadata(self, G, bird_meta):
        """
        BIRD dev_tables.json 정보를 그래프에 적용 (안전한 버전)
        """
        logger.info("Applying BIRD Metadata...")
        
        tbl_names_list = bird_meta.get('table_names', [])
        col_names_list = bird_meta.get('column_names', []) 
        orig_names_list = bird_meta.get('column_names_original', [])
        bird_fks = bird_meta.get('foreign_keys', [])
        bird_pks = bird_meta.get('primary_keys', [])

        # [Safety Check] 테이블 이름 매핑 (Metadata Name -> Graph Node Name)
        # 메타데이터의 이름이 실제 DB와 다를 경우(공백, 대소문자 등)를 대비해 매핑 테이블 생성
        meta_to_graph_table = {}
        
        # 그래프에 있는 실제 테이블 노드 목록
        graph_tables = {n: n for n in G.nodes if G.nodes[n].get('type') == 'table'}
        # 대소문자 무시 매핑용
        graph_tables_lower = {n.lower(): n for n in graph_tables}

        for meta_name in tbl_names_list:
            # 1. 정확히 일치하는 경우
            if meta_name in graph_tables:
                meta_to_graph_table[meta_name] = meta_name
            # 2. 대소문자만 다른 경우
            elif meta_name.lower() in graph_tables_lower:
                meta_to_graph_table[meta_name] = graph_tables_lower[meta_name.lower()]
            # 3. 매칭 실패 (frpm vs free meals 처럼 아예 다른 경우) -> 스킵해야 함
            else:
                pass 
                # logger.warning(f"Mismatch: Meta table '{meta_name}' not found in Graph tables.")

        # -------------------------------------------------------
        # 1. Description (Original Name) 업데이트 & PK 보정
        # -------------------------------------------------------
        for col_idx, (tbl_idx, col_name) in enumerate(col_names_list):
            if tbl_idx == -1: continue
            
            try:
                meta_tbl_name = tbl_names_list[tbl_idx]
                
                # [Fix] 실제 그래프에 있는 테이블 이름으로 변환
                real_tbl_name = meta_to_graph_table.get(meta_tbl_name)
                
                # 테이블을 못 찾았으면 컬럼 처리도 불가능하므로 스킵
                if not real_tbl_name:
                    continue

                # 노드 ID 구성 (GraphBuilder가 만든 방식대로: table.column)
                # 주의: col_name도 메타데이터와 실제 DB가 다를 수 있음 (대소문자 등)
                target_node_id = f"{real_tbl_name}.{col_name}"
                
                # 그래프에서 해당 노드 찾기 (없으면 이웃 노드 뒤져서 찾기)
                final_node_id = None
                if G.has_node(target_node_id):
                    final_node_id = target_node_id
                else:
                    # 대소문자/공백 차이로 못 찾을 경우, 해당 테이블의 컬럼들을 뒤져서 매칭
                    col_lower = col_name.lower().replace(" ", "")
                    for n in G.neighbors(real_tbl_name):
                        # n은 'table.column' 형태
                        n_col_part = n.split('.', 1)[1]
                        if n_col_part.lower().replace(" ", "") == col_lower:
                            final_node_id = n
                            break
                
                if not final_node_id:
                    continue

                # (1) Description 업데이트
                orig_name = orig_names_list[col_idx][1]
                if orig_name:
                    # 기존 description에 덧붙이거나 덮어쓰기
                    curr_desc = G.nodes[final_node_id].get('description')
                    if curr_desc:
                        G.nodes[final_node_id]['description'] = f"{curr_desc}, {orig_name}"
                    else:
                        G.nodes[final_node_id]['description'] = orig_name

                # (2) PK 보정
                if col_idx in bird_pks:
                    G.nodes[final_node_id]['is_pk'] = True

            except (IndexError, KeyError):
                continue

        # -------------------------------------------------------
        # 2. Foreign Keys 연결
        # -------------------------------------------------------
        for src_idx, tgt_idx in bird_fks:
            try:
                # Source
                src_tbl_idx, src_col_name = col_names_list[src_idx]
                src_meta_tbl = tbl_names_list[src_tbl_idx]
                src_real_tbl = meta_to_graph_table.get(src_meta_tbl)

                # Target
                tgt_tbl_idx, tgt_col_name = col_names_list[tgt_idx]
                tgt_meta_tbl = tbl_names_list[tgt_tbl_idx]
                tgt_real_tbl = meta_to_graph_table.get(tgt_meta_tbl)

                if src_real_tbl and tgt_real_tbl:
                    self._add_edge_debug(G, src_real_tbl, src_col_name, tgt_real_tbl, tgt_col_name)
                
            except IndexError:
                continue
    
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
                logger.debug(f"[FK Linked] {u} <-> {v}")
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

    def _find_real_node_id(self, G, table_name, target_node_id):
        """ 그래프에 존재하는 실제 노드 ID를 대소문자 무시하고 찾음 """
        if G.has_node(target_node_id):
            return target_node_id
        
        # 없으면 대소문자 무시 검색
        target_lower = target_node_id.lower()
        if G.has_node(table_name): # 테이블 노드가 있다면 그 이웃(컬럼)을 검색
            for n in G.neighbors(table_name):
                if n.lower() == target_lower:
                    return n
        return None

class ShortcutGraphBuilder(SimpleFKGraphBuilder):
    def build_graph(self, db_engine, db_id=None, bird_meta=None):
        G, tables = super().build_graph(db_engine, db_id=db_id, bird_meta=bird_meta)

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
    def __init__(self, config, cache_dir="./data/cache_graphs"):
        """
        Args:
            cache_path: 생성된 그래프를 저장/로드할 경로
        """
        super().__init__(mode=config['data']['mode'])
        self.cache_dir = cache_dir
        self.config = config

        llm_config = self.config.get('agent', {})

        raw_url = llm_config.get('url', 'http://localhost:8000/v1')
        if not raw_url.endswith('/v1'):
            raw_url = f"{raw_url.rstrip('/')}/v1"
        
        self.base_url = raw_url
        self.api_key = llm_config.get('api_key', 'EMPTY')
        self.model_name = llm_config.get('model_id', 'Qwen/Qwen3-8B-AWQ')

        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=120.0
        )

        self.sem = asyncio.Semaphore(20)

        logger.info(f"[INFO] Graph Builder Initialized with {self.model_name}")

    def get_db_engine_for_db(self, db_id):
        mode = self.config['data']['mode']

        if mode == 'bird':
            db_root = self.config['data']['bird']['db_root_path']
            db_path = os.path.join(db_root, db_id, f"{db_id}.sqlite")
            return create_engine(f"sqlite:///{db_path}")

        return None

    def build_graph(self, db_engine, db_id, bird_meta=None):
        if bird_meta:
            self.cache_path = os.path.join(self.cache_dir, f"{db_id}.pkl")
        else:
            self.cache_path = os.path.join(self.cache_dir, "lg55_db_semantic.pkl")
            
        if os.path.exists(self.cache_path):
            logger.debug(f"Loading Cached Semantic Graph from {self.cache_path}...")
            with open(self.cache_path, 'rb') as f:
                saved_data = pickle.load(f)
                self.G = saved_data['G']
                return self.G, saved_data['tables']

        G, tables = super().build_graph(db_engine, db_id=db_id, bird_meta=bird_meta)


        logger.info("Generating Semantic Description for Edges (This may take a while)...")
        asyncio.run(self._enrich_edges_with_llm(G))

        self._sanitize_graph_for_pickle(G)

        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'wb') as f:
            pickle.dump({'G': G, 'tables': tables}, f)
        logger.info(f"Graph cached to {self.cache_path}")

        return G, tables
    
    def _sanitize_graph_for_pickle(self, G):
        cleaned = 0
        for u, v, data in G.edges(data=True):
            for k, val in data.items():
                if asyncio.iscoroutine(val) or asyncio.isfuture(val):
                    logger.warning(f"[WARNING] Pickle Guard: Coroutine found in {u}-{v} key '{k}'. Converting to string.")
                    G[u][v][k] = str(val)
                    cleaned += 1
        
        if cleaned > 0:
            logger.warning(f"[WARNING] Pickle Guard. Cleaned {cleaned} coroutine objects from graph.")

    async def _enrich_edges_with_llm(self, G):
        """
        Graph의 모든 FK Edge를 순회하며 Semantic Desciption 추가
        """
        
        edges_to_process = []

        # FK 관계인 Edge만 추출
        for u, v, data in G.edges(data=True):
            if data.get('relation') in ['foreign_key', 'table_foreign_key']:
                if 'textual_label' not in data:
                    edges_to_process.append((u, v, data))
        
        logger.info(f"Processing {len(edges_to_process)} edges for semantic labeling...")
        
        tasks = [self._process_single_edge(G, u, v, data) for u, v, data in edges_to_process]

        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            await f

    
    async def _process_single_edge(self, G, u, v, data):
        async with self.sem:
            u_node = G.nodes[u]
            v_node = G.nodes[v]
            prompt = self._create_edge_prompt(u_node, v_node, data)

            try:
                description = await self._call_llm(prompt)
                if asyncio.iscoroutine(description):
                    description = await description

                G[u][v]['textual_label'] = description
            
            except Exception as e:
                logger.error(f"Failed to generate label for {u} -> {v}: {str(e)}")
                G[u][v]['textual_label'] = "related to"
            
    
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
    
    @retry(
            wait=wait_random_exponential(min=1, max=20),
            stop=stop_after_attempt(5),
            retry=retry_if_exception_type((RateLimitError, APIError, ConnectionError))
    )
    async def _call_llm(self, prompt):
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        completion = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=50,
        )

        return completion.choices[0].message.content.strip()
