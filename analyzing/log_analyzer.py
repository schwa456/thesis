import os
import json
import re
import ast
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import sqlglot
from sqlglot import exp

# =============================================================================
# [설정] 환경에 맞게 경로를 수정하세요
# =============================================================================
LOG_FILE_PATH = "../logs/exp_logs/2026-02-13/bird_log_main_framework.log"      # 분석할 로그 파일
DEV_JSON_PATH = "../data/BIRD_dev/dev.json"            # 정답 데이터셋 (SQL 포함)
CACHE_GRAPH_DIR = "../data/cache_graphs"               # .pkl 파일 저장 경로 (Raw Node 계산용)
OUTPUT_CSV_PATH = "result_analysis.csv"  # 결과 저장 경로
# =============================================================================

def extract_schema_from_sql(sql):
    """SQL을 파싱하여 정답 스키마(Table, Column) 집합 추출"""
    schema = set()
    if not sql: return schema
    try:
        parsed = sqlglot.parse_one(sql, read="mysql")
    except:
        return schema

    tables = {}
    for table in parsed.find_all(exp.Table):
        real_name = table.name
        alias = table.alias_or_name
        tables[alias] = real_name
        schema.add(real_name.lower())
    
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

def get_raw_graph_sizes(cache_dir):
    """캐시된 그래프(.pkl)에서 DB별 전체 노드 수 로드"""
    db_sizes = {}
    if not os.path.exists(cache_dir): return {}
    
    for filename in os.listdir(cache_dir):
        if filename.endswith(".pkl"):
            db_id = filename.replace(".pkl", "")
            try:
                with open(os.path.join(cache_dir, filename), 'rb') as f:
                    data = pickle.load(f)
                G = data['G'] if isinstance(data, dict) and 'G' in data else data
                db_sizes[db_id] = len(G.nodes)
            except: pass
    return db_sizes

def load_ground_truth(json_path):
    """dev.json에서 메타데이터(질문ID, DB_ID, GT Schema) 로드"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    gt_map = {}
    for idx, item in enumerate(data):
        q_norm = item['question'].strip()
        gt_map[q_norm] = {
            "question_id": item.get("question_id", idx),
            "db_id": item["db_id"],
            "gt_schema_set": extract_schema_from_sql(item['SQL']),
            "difficulty": item["difficulty"]
        }
    return gt_map

def parse_schema_from_prompt(prompt_lines):
    """로그에 기록된 프롬프트에서 예측된 스키마 추출"""
    pred_schema = set()
    current_table = None
    
    for line in prompt_lines:
        tbl_match = re.search(r"# Table:\s*(\w+)", line)
        if tbl_match:
            current_table = tbl_match.group(1).lower()
            pred_schema.add(current_table)
            continue
        
        if line.strip().startswith('(') and current_table:
            content = line.strip().strip('(),')
            parts = [p.strip() for p in content.split(',')]
            col_name = parts[1] if len(parts) >= 2 else parts[0].split(':')[0]
            if col_name:
                pred_schema.add(f"{current_table}.{col_name.lower()}")
    return pred_schema

def analyze_logs(log_path, gt_map, raw_sizes):
    results = []
    
    time_pat = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
    q_pat = re.compile(r"Question: (.*)")
    
    entry = {}
    current_prompt = []
    is_prompt = False
    
    # [핵심 수정] 타임스탬프가 없는 라인을 위해 마지막 유효 시간 기억
    last_valid_time = None

    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        # 1. 타임스탬프 추출
        tm_match = time_pat.match(line)
        if tm_match:
            curr_time = datetime.strptime(tm_match.group(1), "%Y-%m-%d %H:%M:%S")
            last_valid_time = curr_time # 유효 시간 갱신
        else:
            curr_time = last_valid_time # 타임스탬프 없으면 직전 시간 사용

        # 2. 질문 시작
        q_match = q_pat.search(line)
        if q_match:
            if entry.get("question"):
                finalize_entry(entry, gt_map, raw_sizes, results)
            
            entry = {
                "question": q_match.group(1).strip(),
                "t_start": curr_time, "t_end": curr_time,
                "t_agent_start": None, "t_agent_end": None,
                "t_gen_start": None, "t_gen_end": None,
                "agent_selected": None, "agent_reasoning": None,
                "pcst_nodes": 0, "final_nodes": 0,
                "generated_sql": None, "status": "UNKNOWN", "error": None,
                "pred_schema_set": set()
            }
            current_prompt = []
            is_prompt = False
            continue
        
        if not entry.get("question"): continue
        if curr_time: entry["t_end"] = curr_time

        # 3. Agent 파싱
        if "[AgentSelector] Prompt Preview" in line:
            entry["t_agent_start"] = curr_time
        
        if "[DEBUG] >>> Parsed JSON:" in line:
            entry["t_agent_end"] = curr_time
            json_str = line.split("Parsed JSON: ")[1].strip()
            try:
                data = ast.literal_eval(json_str)
                entry["agent_selected"] = str(data.get("selected_items", {}))
                entry["agent_reasoning"] = data.get("reasoning", "")
            except:
                entry["agent_selected"] = "Error"

        # 4. PCST
        if "[Integrity Check] PCST Output" in line:
            nums = re.findall(r"\d+", line)
            if len(nums) >= 2:
                entry["pcst_nodes"] = int(nums[-2])
                entry["final_nodes"] = int(nums[-1])

        # 5. Prompt Parsing
        if "Prompt:" in line:
            is_prompt = True
            continue
        if "Sending SQL Generation Request" in line:
            is_prompt = False
            entry["pred_schema_set"] = parse_schema_from_prompt(current_prompt)
            entry["t_gen_start"] = curr_time
            continue
        if is_prompt:
            current_prompt.append(line)

        # 6. SQL 및 결과 파싱 (수정됨)
        # 타임스탬프 없는 SELECT 라인이나 main.py 로그 라인 모두 대응
        clean_line = line.strip()
        # main.py에서 찍는 로그: "... - SELECT ..." 형태 감지
        if " - SELECT " in line or clean_line.startswith("SELECT") or clean_line.startswith("WITH"):
            # 로그 메시지 부분만 추출 시도 (타임스탬프 있는 경우)
            if " - " in line:
                parts = line.split(" - ")
                # 마지막 부분이 메시지일 확률 높음
                potential_sql = parts[-1].strip()
                if potential_sql.startswith("SELECT") or potential_sql.startswith("WITH"):
                    entry["generated_sql"] = potential_sql
            else:
                entry["generated_sql"] = clean_line
            
            # [핵심] SQL 생성 종료 시간 기록 (last_valid_time 덕분에 None 아님)
            entry["t_gen_end"] = curr_time

        if "[SUCCESS]" in line: entry["status"] = "SUCCESS"
        if "[FAIL]" in line: entry["status"] = "FAIL"
        if "[ERROR]" in line: 
            entry["status"] = "ERROR"
            entry["error"] = line.split("[ERROR]")[1].strip()

    if entry.get("question"):
        finalize_entry(entry, gt_map, raw_sizes, results)

    return pd.DataFrame(results)

def finalize_entry(entry, gt_map, raw_sizes, results):
    q = entry["question"]
    gt_info = gt_map.get(q)
    
    row = {
        "question_id": -1, "db_id": "UNKNOWN", "difficulty": "UNKNOWN",
        "Question": q, "Status": entry["status"],
        "Agent_Selected_Items": entry["agent_selected"],
        "Agent_Reasoning": entry["agent_reasoning"],
        "RAW_NODES": 0,
        "PCST_NODES": entry["pcst_nodes"],
        "Final_Nodes": entry["final_nodes"],
        "Generated_SQL": entry["generated_sql"],
        "Error_Detail": entry["error"],
    }

    gt_set = set()
    if gt_info:
        row["question_id"] = gt_info["question_id"]
        row["db_id"] = gt_info["db_id"]
        row["difficulty"] = gt_info["difficulty"]
        gt_set = gt_info["gt_schema_set"]
        row["RAW_NODES"] = raw_sizes.get(gt_info["db_id"], 0)

    pred_set = entry["pred_schema_set"]
    tp = len(gt_set & pred_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    row.update({
        "Schema_Recall": round(recall, 4),
        "Schema_Precision": round(precision, 4),
        "Schema_F1": round(f1, 4),
        "Pred_Count": len(pred_set),
        "GT_Count": len(gt_set),
        "Missing_col": list(gt_set - pred_set),
        "Extra_col": list(pred_set - gt_set)
    })

    def calc_sec(start, end):
        # start나 end가 None이면 0.0 반환
        return (end - start).total_seconds() if (start and end) else 0.0

    row["Agent_Time_sec"] = calc_sec(entry["t_agent_start"], entry["t_agent_end"])
    row["Generator_Time_sec"] = calc_sec(entry["t_gen_start"], entry["t_gen_end"])
    row["Total_Time_sec"] = calc_sec(entry["t_start"], entry["t_end"])

    results.append(row)

def main():
    print("1. Loading Data Sources...")
    gt_map = load_ground_truth(DEV_JSON_PATH)
    raw_sizes = get_raw_graph_sizes(CACHE_GRAPH_DIR)
    
    print("2. Parsing Logs & Analyzing...")
    df = analyze_logs(LOG_FILE_PATH, gt_map, raw_sizes)
    
    cols = [
        "question_id", "db_id", "difficulty", "Question", "Status", 
        "Agent_Selected_Items", "Agent_Reasoning", 
        "RAW_NODES", "PCST_NODES", "Final_Nodes", 
        "Schema_Recall", "Schema_Precision", "Schema_F1", 
        "Pred_Count", "GT_Count", "Missing_col", "Extra_col", 
        "Generated_SQL", "Error_Detail", 
        "Agent_Time_sec", "Generator_Time_sec", "Total_Time_sec"
    ]
    final_cols = [c for c in cols if c in df.columns]
    df = df[final_cols]

    print(f"3. Saving to {OUTPUT_CSV_PATH}...")
    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*50)
    print(f"✅ Analysis Complete! (Processed {len(df)} queries)")
    print(f"Avg Schema Recall: {df['Schema_Recall'].mean():.4f}")
    print(f"Avg Schema Precision: {df['Schema_Precision'].mean():.4f}")
    print(f"Avg Schema F1: {df['Schema_F1'].mean():.4f}")
    print(f"Avg Reasoning Time: {df['Agent_Time_sec'].mean():.2f}s")
    print(f"Avg Generator Time: {df['Generator_Time_sec'].mean():.2f}s")
    print(f"EX: {len(df[df['Status'] == 'SUCCESS']) / len(df):.4f}")
    print("="*50)

if __name__ == "__main__":
    main()