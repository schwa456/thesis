import networkx as nx
from sqlalchemy import text

def graph_to_mschema(focused_graph, db_engine=None, full_graph=None, db_id="database"):
    """
    Focused Graph를 입력받아 표준 M-Schema 포맷(이미지 우측)으로 변환합니다.
    
    Format Example:
    【DB_ID】 database
    【Schema】
    # Table: users
    [
      (id:INTEGER, Primary Key, user id, Examples: [1, 2]),
      (name:TEXT, user name, Examples: ['Alice', 'Bob'])
    ]
    【Foreign keys】
    orders.user_id = users.id
    """
    
    lines = []

    # 1. Header
    lines.append(f"【DB_ID】 {db_id}")
    lines.append(f"【Schema】")

    # 2. Table / Column 정보
    table_nodes = [n for n, attr in focused_graph.nodes(data=True) if attr.get('type') == 'table']
    table_nodes.sort()

    for table_name in table_nodes:
        lines.append(f"# Table: {table_name}")
        lines.append("[")

        col_defs = []

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
            
            # (1) 이름:타입
            parts = [f"{col_name}:{col_type}"]

            # (2) PK 여부
            if col_attr.get('is_pk'):
                parts.append("Primary Key")

            # (3) Description
            if 'description' in col_attr and col_attr['description']:
                parts.append(col_attr['description'])
            else:
                parts.append(col_name.replace("_", " "))

            # (4) Example
            if db_engine:
                examples = _get_column_examples(db_engine, table_name, col_name)
                if examples:
                    parts.append(f"Examples: {examples}")
            
            col_defs.append(f"  ({', '.join(parts)})")

        lines.append(",\n".join(col_defs))
        lines.append("]")

    # 3. FK 작성
    fk_statement = set()

    for u, v, data in focused_graph.edges(data=True):
        if data.get('relation') == 'foreign_key':

            # u, v가 각각 Column Node인지 확인
            if focused_graph.nodes[u].get('type') == 'column' and focused_graph.nodes[v].get('type') == 'column':
                
                u_attr = focused_graph.nodes[u]
                v_attr = focused_graph.nodes[v]

                t1, c1 = u_attr.get('table'), u_attr.get('name')
                t2, c2 = v_attr.get('table'), v_attr.get('name')

                if t1 and t2 and t1 != t2:
                    pair = sorted([f"{t1}.{c1}", f"{t2}.{c2}"])
                    fk_statement.add(f"{pair[0]} = {pair[1]}")
    
    if fk_statement:
        lines.append("【Foreign keys】")
        for fk in sorted(list(fk_statement)):
            lines.append(fk)
    
    return "\n".join(lines)
    

def _get_column_examples(db_engine, table, col):
    """DB에서 실제 값 3개를 가져오는 헬퍼 함수"""
    try:
        with db_engine.connect() as conn:
            should_fetch_all = False

          #TODO: 컬럼명 확인 로직 수정 필요
            # 1. 컬럼명이 _id으로 끝나는지 확인
            if col.endswith("_id"):
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
                val = row[0]
                if isinstance(val, str):
                    values.append(f"'{val}'") # 문자열은 따옴표
                else:
                    values.append(str(val))
            
            if values:
                return f"[{', '.join(values)}]"
            return None
    except Exception:
        return None
