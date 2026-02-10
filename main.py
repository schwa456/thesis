import os
import json
import yaml
import logging
import argparse
import requests
import traceback
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sqlalchemy import create_engine, inspect, text

from src.pipeline import Pipeline
from src.utils.evaluation import *
from src.modules.generator import *
from src.utils.graph_to_mschema import *
from src.utils.logger import *
from src.utils.data_loader import *

SERVER_URL = "http://localhost:8000"

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_db_engine(config, db_id):
    mode = config['data']['mode']

    if mode == 'lg55':
        db_config = config['data']['lg55']
        conn_str = db_config.get('db_uri')
        return create_engine(conn_str)
    
    elif mode == 'bird':
        db_root = config['data']['bird']['db_root_path']
        db_path = os.path.join(db_root, db_id, f"{db_id}.sqlite")

        if not os.path.exists(db_path):
            logger.error(f"[ERROR] DB File Not Found: {db_path}")
            return None

        return create_engine(f"sqlite:///{db_path}")
    
    return None

def execute_sql(engine, sql_query):
    clean_sql = sql_query.replace("```sql", "").replace("```", "").strip()

    try:
        with engine.connect() as connection:
            result = connection.execute(text(clean_sql))

            rows = result.fetchall()
            columns = result.keys()

            if not rows:
                return None
            
            df = pd.read_sql(text(clean_sql), connection)
            return df
    
    except Exception as e:
        logger.error(f"[ERROR] SQL Execution Failed: {str(e)}", exc_info=True)
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/main_framework.yaml', help='Path to config file')
    args = parser.parse_args()

    # 1. Config 로드
    config = load_config(args.config)
    loader = DataLoader(config)

    config_path = args.config
    exp_name = Path(config_path).stem
    
    mode = config['data']['mode']
    output_csv = f"./output/{mode}_exp_result_{exp_name}.csv"

    setup_logger(mode, exp_name)

    logger = logging.getLogger(__name__)

    # 2. Pipeline 초기화
    logger.info("[INFO] Initializing Pipeline...")
    pipeline = Pipeline(config)
    logger.info("[INFO] Initializing Pipeline Completed.")

    # 3. Data 로드
    logger.info("[INFO] Loading Dataset...")
    dataset = loader.load_data()
    logger.info(f"[INFO] Loaded {len(dataset)} items from [{config['data']['mode']}] dataset.")

    # 4. SQL Generator 초기화
    gen_config = config.get('generator', {})
    server_url = gen_config.get('url')
    prompt_path = gen_config.get('prompt_path')
    
    generator = XiYanGenerator(
        server_url=server_url,
        prompt_path=prompt_path
    )
    
    # 5. Evaluation
    evaluator = SchemaEvaluator()

    logger.info("Start Processing...")
    
    evaluation_logs = []

    pbar = tqdm(dataset)

    for idx, item in enumerate(pbar):
        question = item['question']
        db_id = item['db_id']
        gt_schema = item['gt_schema']
        gold_sql = item.get("sql", "")
        bird_meta = item.get('meta_schema', '')
        logger.debug(f"Question: {question}")

        pbar.set_description(f"[{db_id}] QID: {idx}")
        
        log_entry = {
            "id": idx,
            "db_id": db_id,
            "question": question,
            "selected_schema": "",
            "gt_schema": gt_schema,
            "gold_sql": item.get("sql", ""),
            "generated_sql": "",    
            "execution_success": False,
            "row_count": 0,
            "error_msg": "",
        }
        
        db_engine = get_db_engine(config, db_id)
        if not db_engine:
            log_entry['error_msg'] = 'DB Connection Failed'
            evaluation_logs.append(log_entry)
            continue

        evidence = item.get('evidence', '')

        try:
            # 파이프라인 실행
            result = pipeline.run(
                question=question,  
                db_id=db_id,
                db_engine=db_engine,
                evidence=evidence,
                bird_meta=bird_meta,
                status_callback=None
            )

            pred_schema = result.get('selected_items', [])
            final_schema_str = result.get('final_schema_str', '')
            log_entry['selected_schema'] = pred_schema

            # 6. SQL Query Generation
            generated_sql = generator.generate_query(question=question, schema_info=final_schema_str, evidence=evidence)

            if generated_sql:
                generated_sql = generated_sql.replace("```sql", "").replace("```", "").strip()

            log_entry['generated_sql'] = generated_sql
            logger.debug(generated_sql)
            
            # Evaluation
            eval_metrics = evaluator.evaluate_single(
                gt_schema=gt_schema,
                pred_schema=pred_schema,
                gt_sql=gold_sql,
                pred_sql=generated_sql,
                db_engine=db_engine
            )

            log_entry.update(eval_metrics)

            if eval_metrics['schema_recall'] < 1.0:
                logger.debug(f"[Low Recall] Missing: {eval_metrics['missing_cols']}")

            if eval_metrics['ex'] == 1.0:
                logger.debug(f"[SUCCESS] Execution Match! (Time: {eval_metrics['pred_exec_time']}s)")
            else:
                logger.debug(f"[FAIL] Execution Mismatch. Err: {eval_metrics.get('exec_error')}")

        except Exception as e:
            logger.error(f"[ERROR] Failed on item {idx}: {e}")
            logger.error(traceback.format_exc())
            log_entry['error_msg'] = str(e)

        evaluation_logs.append(log_entry)
    
    final_df = pd.DataFrame(evaluation_logs)

    if config['evaluation_log']:
        final_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        logger.info(f"[INFO] Evaluation Completed! Saved to {output_csv}")
    else:
        logger.info(f"[INFO] Evaluation Completed! Not saved as configuration.")

    if not final_df.empty:
        logger.info("========== Final Result ==========")
        logger.info(f"Avg. Schema Recall:    {final_df['schema_recall'].mean():.4f}")
        logger.info(f"Avg. Schema Precision: {final_df['schema_precision'].mean():.4f}")
        logger.info(f"Avg. Schema F1:        {final_df['schema_f1'].mean():.4f}")
        logger.info(f"Avg. Schema Jaccard:   {final_df['schema_jaccard'].mean():.4f}")
        logger.info("-" * 30)
        logger.info(f"Execution Accuracy (EX): {final_df['ex'].mean():.4f}")
        logger.info(f"Exact Matching (EM):     {final_df['em'].mean():.4f}")
        logger.info(f"Relative VES (R-VES):    {final_df['r_ves'].mean():.4f}")
        logger.info(f"Soft-F1 (Execution):     {final_df['exec_soft_f1'].mean():.4f}")
        logger.info("==================================")

    pbar.close()

if __name__ == "__main__":
    main()
