import os
import sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(curr_dir, 'm_schema')

if lib_path not in sys.path:
    sys.path.append(lib_path)

from m_schema import MSchema

import networkx as nx
from sqlalchemy import text


def _parse_bird_comments(bird_meta):
    """
    BIRD 메타데이터에서 (table, col) -> original_name 매핑을 생성합니다.
    """
    if not bird_meta:
        return {}

    comment_map = {}
    
    # BIRD json 구조:
    # table_names: ["student", "course", ...] (물리적 이름)
    # column_names: [[0, "id"], [0, "name"], [1, "id"] ...] (물리적 이름)
    # column_names_original: [[0, "id"], [0, "student name"], ...] (논리적 이름)
    
    phy_tables = bird_meta.get('table_names', [])
    phy_cols = bird_meta.get('column_names', [])
    orig_cols = bird_meta.get('column_names_original', [])

    for idx, (tbl_idx, phy_col_name) in enumerate(phy_cols):
        if tbl_idx == -1: continue # '*' 컬럼 등은 건너뜀
        
        try:
            table_name = phy_tables[tbl_idx]
            orig_col_name = orig_cols[idx][1] # 원래 이름 (예: "Student Name")
            
            # 검색하기 쉽게 소문자로 키 생성
            key = (table_name.lower(), phy_col_name.lower())
            comment_map[key] = orig_col_name
        except IndexError:
            continue
            
    return comment_map

def graph_to_mschema(focused_graph, db_engine=None, db_id="database", bird_meta=None):
    # 1. MSchema 객체 초기화
    mschema = MSchema(db_id=db_id)

    # [NEW] BIRD 메타데이터 파싱 (매핑 테이블 생성)
    bird_comments = _parse_bird_comments(bird_meta)
    is_bird = (bird_meta is not None)

    # 2. Table / Column 정보 입력
    table_nodes = [n for n, attr in focused_graph.nodes(data=True) if attr.get('type') == 'table']
    table_nodes.sort()

    for table_name in table_nodes:
        # 2-1. 테이블 추가
        # (선택) 테이블에도 코멘트가 있다면 여기서 추가 가능
        mschema.add_table(table_name)

        neighbors = focused_graph.neighbors(table_name)
        cols = []
        for n in neighbors:
            node_attr = focused_graph.nodes[n]
            if node_attr.get('type') == 'column':
                cols.append((n, node_attr))
        
        cols.sort(key=lambda x: (not x[1].get('is_pk', False), x[1].get('name')))

        for col_id, col_attr in cols:
            col_name = col_attr.get('name')
            col_type = str(col_attr.get('dtype', 'TEXT')).upper()
            is_pk = col_attr.get('is_pk', False)
            
            # --- [핵심 수정: 코멘트 로직 강화] ---
            # 1순위: BIRD dev_tables.json의 Original Name
            # 2순위: 그래프에 있는 description
            # 3순위: 이름에서 언더바 제거
            
            comment = ""
            
            # (1) BIRD 메타데이터 확인
            bird_key = (table_name.lower(), col_name.lower())
            if bird_key in bird_comments:
                comment = bird_comments[bird_key]
            
            # (2) 없으면 기존 로직 (그래프 속성 or 언더바 제거)
            if not comment:
                comment = col_attr.get('description', '')
            if not comment:
                comment = col_name.replace("_", " ")
            # -----------------------------------

            # Example 값 가져오기
            examples = []
            if db_engine:
                examples = _get_column_examples(db_engine, table_name, col_name, is_bird=is_bird)
            
            if examples is None:
                examples = []

            # MSchema에 필드 추가
            mschema.add_field(
                table_name=table_name,
                field_name=col_name,
                field_type=col_type,
                primary_key=is_pk,
                comment=comment,     # <--- 여기에 BIRD의 친절한 설명이 들어갑니다
                examples=examples
            )

    # 3. Foreign Key 작성 (기존 동일)
    # (BIRD의 FK 정보도 bird_meta['foreign_keys']에 있지만, 
    #  보통 Graph 구축 단계에서 이미 FK가 연결되어 있으므로 여기선 그래프 정보를 우선합니다.)
    
    added_fks = set()
    for u, v, data in focused_graph.edges(data=True):
        if data.get('relation') == 'foreign_key':
            u_attr = focused_graph.nodes[u]
            v_attr = focused_graph.nodes[v]

            if u_attr.get('type') == 'column' and v_attr.get('type') == 'column':
                t1, c1 = u_attr.get('table'), u_attr.get('name')
                t2, c2 = v_attr.get('table'), v_attr.get('name')

                if t1 and t2 and t1 != t2:
                    pair_key = tuple(sorted([f"{t1}.{c1}", f"{t2}.{c2}"]))
                    if pair_key not in added_fks:
                        mschema.add_foreign_key(t1, c1, None, t2, c2)
                        added_fks.add(pair_key)

    return mschema.to_mschema()
    

def _get_column_examples(db_engine, table, col, is_bird=False):
    """DB에서 실제 값 3개를 가져오는 헬퍼 함수"""
    try:
        with db_engine.connect() as conn:
            should_fetch_all = False
            
            col = f'`{col}`'
            table = f'`{table}`'

            # 1. 컬럼명이 _NM으로 끝나는지 확인
            if not is_bird and col.endswith("_NM"):
                count_query = text(f"SELECT count(DISTINCT {col}) FROM {table} WHERE {col} IS NOT NULL")
                count_val = conn.execute(count_query).scalar()

                if count_val is not None and count_val <= 150:
                    should_fetch_all = True
            
            # 2. 쿼리 구성
            base_query = f"SELECT DISTINCT {col} FROM {table} WHERE {col} IS NOT NULL"
            
            # 조건에 따라 LIMIT 추가 여부 결정
            if not should_fetch_all:
                base_query += " LIMIT 3"

            # 실제 데이터 조회
            query = text(base_query)
            result = conn.execute(query).fetchall()
            
            values = []
            for row in result:
                values.append(row[0])   

            return values
        
    except Exception:
        return []
    
def nodes_to_mschema(retrieved_nodes, full_graph, db_engine, db_id, bird_meta=None):
    final_nodes = set(retrieved_nodes)

    for node in retrieved_nodes:
        if '.' in node:
            table_name = node.split('.')[0]

            if full_graph.has_node(table_name):
                final_nodes.add(table_name)
    
    focused_graph = full_graph.subgraph(final_nodes).copy()

    schema_string = graph_to_mschema(
        focused_graph=focused_graph,
        db_engine=db_engine,
        db_id=db_id,
        bird_meta=bird_meta
    )

    return schema_string
