import json
import yaml
import argparse
from tqdm import tqdm

from src.pipeline import Pipeline
from src.utils.evaluation import *

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data(data_path, table_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    with open(table_path, 'r') as f:
        tables = json.load(f)
        tables_map = {t['db_id']: t for t in tables}
    
    return data, tables_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/exp_config.yaml', help='Path to config file')
    args = parser.parse_args()

    # 1. Config 로드
    print(f"Loading Configuration from {args.config}...")
    config = load_config(args.config)

    # 2. Pipeline 초기화
    print("🚀 Initializing Pipeline...")
    pipeline = Pipeline(config)

    # 3. Data 로드
    data, tables_map = load_data(config['data_path'], config['table_path'])

    print("Start Processing...")
    for i, sample in enumerate(tqdm(data)):
        db_id = sample['db_id']
        question = sample['question']

        gold_sql = sample.get('SQL', '')
        
        if db_id not in tables_map: continue

        meta = tables_map[db_id]

        # 파이프라인 실행
        result = pipeline.run(db_id, question, meta)

        gold_tables = extract_gold_tables(gold_sql)

        print(f"\n[Sample {i+1}]")
        print(f"Q: {question}")
        print(f"Seeds: {result['initial_seeds']}")
        print(f"Expanded: {result['expanded_nodes']}")
        print(f"Final Agent Selection: {result['final_schema']}")
        print(f"Gold Tables: {gold_tables}")
        print("-" * 50)

if __name__ == "__main__":
    main()
       