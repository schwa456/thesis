import os
import json
import yaml
import logging
import argparse
import traceback
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sqlalchemy import create_engine, text

# === Custom Modules ===
from src.pipeline import Pipeline
from src.utils.evaluation import *
from src.modules.generator import *
from src.utils.graph_to_mschema import *
from src.utils.logger import *
from src.utils.data_loader import DataLoader

# Logger 설정
setup_logger()
logger = logging.getLogger(__name__)

SERVER_URL = "http://localhost:8000"

# === Helper Functions ===

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
    if not engine or not sql_query: return None
    clean_sql = sql_query.replace("```sql", "").replace("```", "").strip()
    if clean_sql.endswith(";"): clean_sql = clean_sql[:-1]

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
        logger.warning(f"[ERROR] SQL Execution Failed: {str(e)}", exc_info=True)
        return None

# === Main Execution ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/exp_config.yaml')
    args = parser.parse_args()

    # 1. Config 로드
    logger.info(f"[INFO] Loading Configuration from {args.config}...")
    config = load_config(args.config)
    loader = DataLoader(config)
    logger.info(f"[INFO] Configuration Loading Completed.")

    # 2. Pipeline 초기화
    logger.info("[INFO] Initializaing Pipeline...")
    pipeline = Pipeline(config)
    logger.info([INFO] Initializaing Pipeline Completed.")

    # 3. Data 로드
    logger.info("[INFO] Loading Dataset...")
    dataset = loader.load_data()
    logger.info("[INFO] Loaded {len(dataset)} items from [{config['data']['mode']}] dataset.")
    
    # 4. SQL Generator 초기화
    generator = XiYanGenerator(config.get('server_url', 'http://localhost:8001'))

    # 5. Evaluator
    evaluator = SchemaEvaluator()

    logger.info("[INFO] Start Processing...")
    
    # 2. Load Data
    dataset = loader.load_data()
    logger.info(f"Loaded {len(dataset)} items.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = f"./logs/eval_results/eval_result_{timestamp}.csv"
    evaluation_logs = []

    pbar = tqdm(dataset)

    for idx, item in enumerate(pbar):
        question = item['question']
        db_id = item['db_id']
        evidence = item.get('evidence', '')
        gt_schema = item['gt_schema']
        logger.debug(f"Question: {question}")

        pbar.set_description(f"[{db_id}] QID:{idx}")

        log_entry = {
            "id": idx, 
            "db_id": db_id, 
            "question": question,
            "selected_schema": "",
            "gt_schema": gt_schema,
            "gold_sql": "",
            "generated_sql": "",
            "execution_success": False,
            "row_count": 0, 
            "error_msg": "",
        }

        # DB Engine 생성
        db_engine = get_db_engine(config, db_id)
        if not db_engine:
            log_entry['error_msg'] = "DB Connection Failed"
            evaluation_logs.append(log_entry)
            continue

        try:
            # 파이프라인 실행
            result = pipeline.run(
                question=question,
                db_id=db_id,
                db_engine=db_engine,
                status_callback=None
            )

            final_schema_str = result.get("final_schema_str", "")
            
            eval_metrics = evaluator.evaluate_single(gt_schema_list, result['selected_items'])

            # 메트릭 기록
            log_entry.update(eval_metrics)  # precision, recall, f1 등 업데이트
            log_entry['gt_count'] = len(gt_schema)
            log_entry['pred_count'] = len(result['selected_items'])
            log_entry['selected_schema'] = result['selected_items']

            if eval_metrics['recall'] < 1.0:
                logger.debug(f"[Low Recall] Low Recall ({eval_metrics['recall']}): Missing {eval_metrics['missing_cols']}")

            # SQL Generation
            generated_sql = sql_generator.generate_query(
                question=question,
                schema_info=final_schema_str
            )
            logger.debug(generated_sql)
            log_entry['generated_sql'] = generated_sql

            if generated_sql:
                df = execute_sql(db_engine, generated_sql)
                if df is not None:
                    log_entry['execution_success'] = True
                    log_entry['row_count'] = len(df)
                    logger.debug(f"[ID: {idx}] Success! ({len(df)} rows")
                else:
                    log_entry['error_msg'] = "Execution Logic Error or Empty Result"
            else:
                log_entry['error_msg'] = "Empty SQL Generated"

        except Exception as e:
            logger.error(f"[ERROR] Failed on item {idx}: {e}")
            logger.error(traceback.format_exc())
            log_entry['error_msg'] = str(e)
            continue

        evaluation_logs.append(log_entry)

        if (idx + 1) % 5 == 0:
            pd.DataFrame(evaluation_logs).to_csv(output_csv, index=False, encoding='utf-8-sig')

    # 최종 저장
    if config['evaluation_log']:
        final_df = pd.DataFrame(evaluation_logs)
        final_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        logger.info(f"Evaluation Completed! Saved to {output_csv}")
    else:
        logger.info(f"Evaluation Completed without saving!")

    # 최종 평균 점수 출력
    if not final_df.empty:
        avg_rec = final_df['recall'].mean()
        avg_prec = final_df['precision'].mean()
        avg_f1 = final_df['f1'].mean()
        avg_jaccard = final_df['jaccard'].mean()
        
        logger.info("Final Result")
        logger.info(f"Avg. Recall: {avg_rec:.4f}")
        logger.info(f"Avg. Precision: {avg_prec:.4f}")
        logger.info(f"Avg. F1: {avg_f1:.4f}")
        logger.info(f"Avg. Jaccard: {avg_jaccard:.4f}")

pbar.close()

if __name__ == "__main__":
    main()
