import json
import yaml
import argparse
import traceback
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sqlalchemy import create_engine, text
import os
import sys
import logging
import re

# === Custom Modules ===
from src.pipeline import Pipeline
from src.utils.evaluation import SchemaEvaluator  # 앞서 만든 평가 클래스
from src.modules.generator import XiYanGenerator, OpenAIGenerator  # 둘 다 필요
from src.utils.logger import setup_logger
from src.utils.data_loader import DataLoader
from src.utils.M_Schema.schema_engine import SchemaEngine  # Full Schema 추출용

# Logger 설정
setup_logger()
# 외부 라이브러리 로그 끄기
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

#TODO: main.py 전체적으로 손봐야 함 -> 지금은 gemini가 준 코드 그대로임
# === Helper Functions ===

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_db_engine(config, db_id):
    """모드(lg55/bird)에 따른 DB 엔진 생성"""
    mode = config['data']['mode']
    if mode == 'lg55':
        db_config = config['data']['lg55']
        conn_str = db_config.get('db_uri')
        return create_engine(conn_str)
    elif mode == 'bird':
        db_root = config['data']['bird']['db_root_path']
        db_path = os.path.join(db_root, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            logger.error(f"❌ DB File Not Found: {db_path}")
            return None
        return create_engine(f"sqlite:///{db_path}")
    return None


def execute_sql(engine, sql_query):
    """SQL 실행 및 결과 반환"""
    if not engine or not sql_query: return None
    clean_sql = sql_query.replace("```sql", "").replace("```", "").strip()
    if clean_sql.endswith(";"): clean_sql = clean_sql[:-1]

    try:
        with engine.connect() as connection:
            result = connection.execute(text(clean_sql))
            rows = result.fetchall()
            if not rows: return pd.DataFrame()
            return pd.DataFrame(rows, columns=result.keys())
    except Exception as e:
        logger.warning(f"⚠️ SQL Execution Failed: {e}")
        return None


def extract_json_from_text(text):
    """LLM 응답에서 JSON 리스트 추출"""
    try:
        match = re.search(r"```json\s*(\[.*?\])\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))

        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1:
            return json.loads(text[start:end + 1])
        return []
    except Exception:
        return []


def generate_gt_schema(question, full_schema_str, gt_generator):
    """GPT-4를 사용하여 GT Schema 생성"""
    system_prompt = """
    You are an expert DB Architect. Select relevant columns from {full_schema} for the question.

    RULES:
    1. Include columns for SELECT, WHERE, GROUP BY.
    2. MUST include PK/FK columns for joining tables.
    3. Output JSON list ONLY: ```json ["Table.Column", ...] ```
    """
    user_prompt = f"Question: {question}\n\nSchema:\n{full_schema_str}"

    try:
        # OpenAIGenerator의 인터페이스에 맞춰 호출 (메서드명 확인 필요)
        # 만약 generate_text 메서드가 없다면 client 직접 호출
        response = gt_generator.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        return extract_json_from_text(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"GT Generation Failed: {e}")
        return []


# === Main Execution ===

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/exp_config.yaml')
    args = parser.parse_args()

    logger.info(f"Loading Configuration from {args.config}...")
    config = load_config(args.config)

    # 1. Initialize Components
    loader = DataLoader(config)

    # Pipeline (Prediction용)
    pipeline = Pipeline(config)

    # SQL Generator (XiYanSQL - 로컬/서버)
    sql_generator = XiYanGenerator(config.get('server_url', 'http://localhost:8001'))

    # GT Generator (GPT-4 - 평가 기준 생성용)
    gt_generator = OpenAIGenerator(model_id="gpt-4o")

    # Evaluator
    evaluator = SchemaEvaluator()

    # 2. Load Data
    dataset = loader.load_data()
    logger.info(f"Loaded {len(dataset)} items.")

    # 3. Process Loop
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = f"./logs/eval_result_{timestamp}.csv"
    evaluation_logs = []

    pbar = tqdm(dataset)

    for idx, item in enumerate(pbar):
        question = item['question']
        db_id = item['db_id']

        pbar.set_description(f"[{db_id}] QID:{idx}")

        log_entry = {
            "id": idx, "db_id": db_id, "question": question,
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "gt_count": 0, "pred_count": 0,
            "missing_cols": [], "extra_cols": [],
            "generated_sql": "", "execution_success": False, "row_count": 0, "error_msg": ""
        }

        # DB Engine 생성
        db_engine = get_db_engine(config, db_id)
        if not db_engine:
            log_entry['error_msg'] = "DB Connection Failed"
            evaluation_logs.append(log_entry)
            continue

        try:
            # --- [Step 1] GT Schema 생성 (GPT) ---
            # 현재 DB의 전체 스키마 추출
            full_mschema = SchemaEngine(engine=db_engine, db_name=db_id).mschema.to_mschema()
            gt_schema_list = generate_gt_schema(question, full_mschema, gt_generator)

            # --- [Step 2] Pipeline 실행 (Prediction) ---
            pipe_result = pipeline.run(question, db_id, db_engine)
            pred_schema_list = pipe_result.get('selected_items', [])
            final_schema_str = pipe_result.get('final_schema_str', '')

            # --- [Step 3] Schema Evaluation ---
            eval_metrics = evaluator.evaluate_single(gt_schema_list, pred_schema_list)

            # 메트릭 기록
            log_entry.update(eval_metrics)  # precision, recall, f1 등 업데이트
            log_entry['gt_count'] = len(gt_schema_list)
            log_entry['pred_count'] = len(pred_schema_list)

            # 디버깅 로그 출력 (Recall이 낮으면 경고)
            if eval_metrics['recall'] < 1.0:
                logger.debug(f"⚠️ Low Recall ({eval_metrics['recall']}): Missing {eval_metrics['missing_cols']}")

            # --- [Step 4] SQL Generation (XiYan) ---
            generated_sql = sql_generator.generate_query(
                question=question,
                schema_info=final_schema_str
            )
            log_entry['generated_sql'] = generated_sql

            # --- [Step 5] SQL Execution ---
            if generated_sql:
                df = execute_sql(db_engine, generated_sql)
                if df is not None:
                    log_entry['execution_success'] = True
                    log_entry['row_count'] = len(df)
                else:
                    log_entry['error_msg'] = "Execution Logic Error or Empty Result"
            else:
                log_entry['error_msg'] = "Empty SQL Generated"

        except Exception as e:
            logger.error(f"Failed on item {idx}: {e}")
            logger.error(traceback.format_exc())
            log_entry['error_msg'] = str(e)

        evaluation_logs.append(log_entry)

        # 중간 저장 (5개마다)
        if (idx + 1) % 5 == 0:
            pd.DataFrame(evaluation_logs).to_csv(output_csv, index=False, encoding='utf-8-sig')

    # 최종 저장
    final_df = pd.DataFrame(evaluation_logs)
    final_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    logger.info(f"Evaluation Completed! Saved to {output_csv}")

    # 최종 평균 점수 출력
    if not final_df.empty:
        avg_rec = final_df['recall'].mean()
        avg_prec = final_df['precision'].mean()
        logger.info(f"📊 Final Average - Recall: {avg_rec:.4f}, Precision: {avg_prec:.4f}")


if __name__ == "__main__":
    main()