import os
import json
import re
import ast
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pcst_fast import pcst_fast
from modules.embedder import SchemaEmbedder

# =========================================================
# [설정] 최적화할 파라미터 후보군 (Grid)
# =========================================================
GRID_PARAMS = {
    "agent_weight": [5.0, 10.0, 20.0],
    "cost_contains": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],          # 내부 컬럼 연결 비용 (낮을수록 Recall 증가)
    "cost_table_fk": [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
    "cost_fk": [0.5, 1.0, 1.5, 2.0],                 # 외부 테이블 연결 비용
    "pcst_cost": [0.5, 1.0, 1.5, 2.0]                # 기본 비용
}

LOG_FILE_PATH = "../logs/exp_logs/2026-02-11/bird_log_main_framework.log"
CACHE_DIR = "../data/cache_graphs"
DEV_JSON_PATH = "../data/BIRD_dev/dev.json"
SAMPLE_SIZE = 50 

# =========================================================

def parse_agent_logs(log_path):
    """로그에서 질문별 Agent 선택 결과 추출"""
    print(f"Parsing Agent logs from {log_path}...")
    agent_choices = {} # {question_text: {node_name: confidence}}
    
    current_q = None
    
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        # 질문 추출
        q_match = re.search(r"Question: (.*)", line)
        if q_match:
            current_q = q_match.group(1).strip()
            continue
            
        # JSON 파싱 (Agent 선택)
        if "[DEBUG] >>> Parsed JSON:" in line:
            if current_q:
                try:
                    json_str = line.split("Parsed JSON: ")[1].strip()
                    data = ast.literal_eval(json_str)
                    selected = data.get("selected_items", {})
                    # {node: score} 형태로 저장
                    agent_choices[current_q] = selected
                except:
                    pass
    
    print(f" -> Found Agent records for {len(agent_choices)} queries.")
    return agent_choices

def generate_node_texts(G, node_ids):
    texts = []
    for n in node_ids:
        attr = G.nodes[n]
        node_type = attr.get('type')
        if node_type == 'table':
            text = f"Table {attr.get('name')}: {attr.get('description', '')}"
        elif node_type == 'column':
            text = f"Column {attr.get('name')} in Table {attr.get('table')}: {attr.get('description', '')}"
        else:
            text = str(n)
        texts.append(text)
    return texts

def load_resources():
    print("1. Loading Embedder...")
    embedder = SchemaEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("2. Loading Graphs...")
    graphs = {}
    node_maps = {}
    for f in os.listdir(CACHE_DIR):
        if f.endswith(".pkl"):
            db_id = f.replace(".pkl", "")
            with open(os.path.join(CACHE_DIR, f), 'rb') as file:
                data = pickle.load(file)
                G = data['G'] if isinstance(data, dict) and 'G' in data else data
                graphs[db_id] = G
                node_maps[db_id] = {n: i for i, n in enumerate(G.nodes)}

    print("3. Loading Ground Truth & Logs...")
    with open(DEV_JSON_PATH, 'r') as f:
        data = json.load(f)
    
    # 로그 파싱
    agent_history = parse_agent_logs(LOG_FILE_PATH)
    
    # 로그에 기록이 있는 질문만 샘플링 (Replay를 위해)
    samples = []
    for item in data:
        q_text = item['question'].strip()
        if q_text in agent_history:
            samples.append({
                "question": q_text,
                "db_id": item['db_id'],
                "agent_selection": agent_history[q_text] # {node: conf}
            })
    
    # 너무 많으면 50개만
    if len(samples) > 50:
        samples = samples[:50]
        
    return embedder, graphs, node_maps, samples

def run_pcst_simulation(G, node_map, prizes, params):
    edges = []
    costs = []
    node_list = list(G.nodes)
    
    for u, v, data in G.edges(data=True):
        if u not in node_map or v not in node_map: continue
        
        relation = data.get('relation', 'generic')
        if relation in ['contains', 'table_column', 'primary_key']:
            c = params['cost_contains']
        elif relation == 'foreign_key':
            c = params['cost_fk']
        elif relation == 'table_foreign_key':
            c = params['cost_table_fk']
        else:
            c = params['pcst_cost']
        
        costs.append(c)
        edges.append([node_map[u], node_map[v]])
        
    if not edges: return 0

    vertices, _ = pcst_fast(
        np.array(edges).astype(np.int64), 
        np.array(prizes).astype(np.float64), 
        np.array(costs).astype(np.float64), 
        -1, 1, "strong", 0
    )
    return len(vertices)

def main():
    embedder, graphs, node_maps, samples = load_resources()
    
    if not samples:
        print("[Error] No matching queries found in logs. Check log file path.")
        return

    print(f"\nStarting Grid Search with Agent Replay ({len(samples)} samples)...")
    
    # Embedder 계산 및 Agent Prize 적용
    print("Pre-calculating Combined Prizes...")
    query_data_list = []
    current_indexed_db = None
    
    for s in tqdm(samples):
        db_id = s['db_id']
        if db_id not in graphs: continue
        G = graphs[db_id]
        
        # 1. Embedder Score
        if current_indexed_db != db_id:
            node_ids = list(G.nodes)
            texts = generate_node_texts(G, node_ids)
            embedder.index_schema_nodes(texts, node_ids)
            current_indexed_db = db_id
            
        emb_scores = embedder.get_similarity_scores(s['question']).cpu().numpy()
        emb_scores[emb_scores < 0.1] = 0.0 # Noise Filter
        
        query_data_list.append({
            "db_id": db_id,
            "emb_scores": emb_scores,
            "agent_selection": s['agent_selection'], # {node_name: conf}
            "node_list": list(G.nodes)
        })

    # Grid Search
    import itertools
    keys, values = zip(*GRID_PARAMS.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    results = []
    print(f"Testing {len(combinations)} combinations...")
    
    for params in tqdm(combinations):
        total_nodes = 0
        valid_count = 0
        
        for q_data in query_data_list:
            db_id = q_data['db_id']
            # 복사해서 사용 (원본 보존)
            final_prizes = q_data['emb_scores'].copy()
            
            # 2. Agent Weight 적용 (Replay Logic)
            # Prize = Embedding + (Agent_Conf * Agent_Weight)
            agent_sel = q_data['agent_selection']
            node_list = q_data['node_list']
            
            for idx, node_name in enumerate(node_list):
                if node_name in agent_sel:
                    conf = agent_sel[node_name]
                    # Agent가 선택한 노드는 Weight만큼 Prize를 뻥튀기
                    boost = conf * params['agent_weight']
                    
                    # 방식: Max(기존점수, Boost) 또는 합산
                    # 보통은 확실히 선택되게 하기 위해 Max나 큰 값 할당
                    if boost > final_prizes[idx]:
                        final_prizes[idx] = boost

            size = run_pcst_simulation(graphs[db_id], node_maps[db_id], final_prizes, params)
            total_nodes += size
            valid_count += 1
            
        avg = total_nodes / valid_count if valid_count else 0
        
        res = params.copy()
        res['avg_nodes'] = avg
        results.append(res)

    # 분석
    df = pd.DataFrame(results)
    
    # 20~30개 사이를 가장 선호 (Agent가 끼면 확실한 것들이 추가되므로 조금 늘어남)
    df['score'] = abs(df['avg_nodes'] - 25) 
    df_sorted = df.sort_values('score')

    print("\n" + "="*60)
    print("TOP 5 Settings (Considering Agent Weight)")
    print("="*60)
    print(df_sorted.head(5))
    
    best = df_sorted.iloc[0]
    print(f"\n[Winner]")
    print(f"Agent Weight: {best['agent_weight']}")
    print(f"Contains: {best['cost_contains']}, Table FK: {best['cost_table_fk']}")
    print(f"Expected Size: {best['avg_nodes']:.1f}")

if __name__ == "__main__":
    main()