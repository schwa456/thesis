import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sqlglot
from sqlglot import exp
from modules.embedder import SchemaEmbedder

# ==========================================
# [설정]
# ==========================================
CACHE_DIR = "../data/cache_graphs"
DEV_JSON_PATH = "../data/BIRD_dev/dev.json"
SAMPLE_SIZE = 100  # 100개 샘플로 통계 확인
OUTPUT_IMG_PATH = "bge-large_model_embedding_distribution.png"
# ==========================================

def extract_schema_from_sql(sql):
    """SQL에서 정답 컬럼/테이블 추출 (sqlglot 사용)"""
    schema = set()
    if not sql: return schema
    try:
        parsed = sqlglot.parse_one(sql, read="mysql")
    except:
        return schema

    tables = {}
    for table in parsed.find_all(exp.Table):
        tables[table.alias_or_name] = table.name
        schema.add(table.name.lower()) # 테이블 추가

    unique_tables = list(set(tables.values()))
    for col in parsed.find_all(exp.Column):
        col_name = col.name
        tbl_ref = col.table
        final_table = tables.get(tbl_ref, tbl_ref)
        if not final_table and len(unique_tables) == 1:
            final_table = unique_tables[0]
        
        if final_table and col_name != "*":
            schema.add(f"{final_table}.{col_name}".lower())
    return schema

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

def main():
    print("1. Loading Embedder...")
    # 사용 중인 모델명 확인 (main_framework.yaml 참조)
    embedder = SchemaEmbedder(model_name="BAAI/bge-large-en-v1.5")

    print("2. Loading Graphs...")
    graphs = {}
    for f in os.listdir(CACHE_DIR):
        if f.endswith(".pkl"):
            db_id = f.replace(".pkl", "")
            with open(os.path.join(CACHE_DIR, f), 'rb') as file:
                data = pickle.load(file)
                graphs[db_id] = data['G'] if isinstance(data, dict) and 'G' in data else data

    print("3. Loading Data & Calculating Scores...")
    with open(DEV_JSON_PATH, 'r') as f:
        data = json.load(f)

    gt_scores = []
    noise_scores = []
    
    current_indexed_db = None

    for item in tqdm(data[:SAMPLE_SIZE]):
        db_id = item['db_id']
        if db_id not in graphs: continue
        
        G = graphs[db_id]
        
        # 1. Embedder Indexing
        if current_indexed_db != db_id:
            node_ids = list(G.nodes)
            texts = generate_node_texts(G, node_ids)
            embedder.index_schema_nodes(texts, node_ids)
            current_indexed_db = db_id
            
        # 2. GT Set 추출
        gt_set = extract_schema_from_sql(item['SQL'])
        if not gt_set: continue
        
        # 3. Score 계산
        scores = embedder.get_similarity_scores(item['question']).cpu().numpy()
        node_list = list(G.nodes)
        
        # 4. GT vs Noise 분류
        for idx, node_name in enumerate(node_list):
            score = scores[idx]
            # 노드 이름 정규화 (소문자)
            node_norm = node_name.lower()
            
            # GT 집합에 포함되면 정답, 아니면 노이즈
            if node_norm in gt_set:
                gt_scores.append(score)
            else:
                noise_scores.append(score)

    # --- 결과 분석 ---
    gt_scores = np.array(gt_scores)
    noise_scores = np.array(noise_scores)
    
    print("\n" + "="*50)
    print("      EMBEDDING SCORE DISTRIBUTION      ")
    print("="*50)
    print(f"Total GT Items   : {len(gt_scores)}")
    print(f"Total Noise Items: {len(noise_scores)}")
    print("-" * 50)
    print(f"GT    Mean: {np.mean(gt_scores):.4f} | Median: {np.median(gt_scores):.4f} | Min: {np.min(gt_scores):.4f}")
    print(f"Noise Mean: {np.mean(noise_scores):.4f} | Median: {np.median(noise_scores):.4f} | Max: {np.max(noise_scores):.4f}")
    print("="*50)
    
    # Gap 분석
    gap = np.mean(gt_scores) - np.mean(noise_scores)
    print(f"Separation Gap (Mean): {gap:.4f}")
    if gap < 0.1:
        print("⚠️ WARNING: Poor separation! Embedder is struggling.")
    else:
        print("✅ Separation is decent.")

    # --- 시각화 (이미지 저장) ---
    plt.figure(figsize=(10, 6))
    sns.kdeplot(gt_scores, fill=True, label='Ground Truth (GT)', color='green')
    sns.kdeplot(noise_scores, fill=True, label='Noise', color='red')
    plt.title('Embedding Score Distribution: GT vs Noise')
    plt.xlabel('Similarity Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_IMG_PATH)
    print(f"\nDistribution plot saved to: {OUTPUT_IMG_PATH}")

if __name__ == "__main__":
    main()