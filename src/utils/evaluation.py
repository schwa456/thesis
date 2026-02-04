import pandas as pd
import numpy as np
import logging
import time
import math
import ast
import sqlite3
from typing import List, Any, Dict, Set
from collections import Counter
from sqlalchemy import text, create_engine

logger = logging.getLogger(__name__)

def safe_list_converter(data):
    if isinstance(data, list):
        return data
    if isinstance(data, str):
        try:
            return ast.literal_eval(data.strip())
        except (ValueError, SyntaxError):
            if ',' in data:
                return [x.strip() for x in data.split(',')]
            return []
    if isinstance(data, set):
        return list(data)
    return []

class SchemaEvaluator:
    def __init__(self, db_path: str = None):
        self.db_path = db_path

    def _normalize_set(self, item_list):
        clean_list = safe_list_converter(item_list)
        if not clean_list:
            return set()
        return set(str(item).lower().strip() for item in clean_list)

    def _normalize_sql(self, sql: str) -> str:
        if not sql: return ""
        return " ".join(str(sql).strip().lower().split()).replace(";", "")

    def _normalize_val(self, val):
        if val is None:
            return "none"
        s = str(val).lower().strip()
        s = s.replace(" ", "")
        return s
    
    def _is_flexible_equal(self, val1, val2):
        if val1 == "none" and val2 == "none":
            return True
        if val1 == val2:
            return True
        
        # 숫자 비교 (소수점 처리)
        if val1.replace('.', '', 1).isdigit() and val2.replace('.', '', 1).isdigit():
            try:
                return float(val1) == float(val2)
            except:
                pass
        
        # 문자열 포함 관계 (Substring)
        if len(val1) > len(val2):
            if val2 in val1: return True
        else:
            if val1 in val2: return True
        return False

    def _normalize_result_set(self, result_set):
        normalized = set()
        for row in result_set:
            sorted_row = tuple(sorted(str(val) for val in row))
            normalized.add(sorted_row)
        return normalized
    
    def _fuzzy_row_match(self, gt_row, pred_row):
        gt_vals = [self._normalize_val(v) for v in gt_row]
        pred_vals = [self._normalize_val(v) for v in pred_row]

        if len(pred_vals) <= len(gt_vals):
            subset, superset = pred_vals, gt_vals
        else:
            subset, superset = gt_vals, pred_vals

        matched_indices = set()
        for sub_item in subset:
            found = False
            for i, super_item in enumerate(superset):
                if i in matched_indices:
                    continue
                if self._is_flexible_equal(sub_item, super_item):
                    matched_indices.add(i)
                    found = True
                    break
            if not found:
                return False
        return True

    def _setup_db_timeout(self, conn, engine, timeout_sec=10.0):
            """
            DB 엔진 종류에 따라 알맞은 타임아웃을 설정하는 통합 함수
            """
            dialect = engine.dialect.name.lower()

            # 1. MariaDB / MySQL 인 경우 (Server-side Timeout)
            if 'mysql' in dialect or 'mariadb' in dialect:
                try:
                    # MariaDB: 초 단위 (float 지원)
                    # MySQL: 밀리초 단위일 수 있으나, MariaDB 10.1+는 max_statement_time(초) 지원
                    # 안전하게 MariaDB 전용 구문 사용
                    conn.execute(text(f"SET SESSION max_statement_time = {float(timeout_sec)}"))
                    return None # 별도의 해제 작업 필요 없음
                except Exception as e:
                    # 구버전 MySQL 호환 (밀리초 단위 max_execution_time)
                    try:
                        ms = int(timeout_sec * 1000)
                        conn.execute(text(f"SET SESSION max_execution_time = {ms}"))
                    except:
                        logger.warning(f"Failed to set timeout for {dialect}: {e}")
                return None

            # 2. SQLite 인 경우 (Client-side Interrupt)
            elif 'sqlite' in dialect:
                raw_conn = conn.connection
                if hasattr(raw_conn, 'dbapi_connection'):
                    raw_conn = raw_conn.dbapi_connection
                
                if isinstance(raw_conn, sqlite3.Connection):
                    start_time = time.time()
                    def progress_handler():
                        if time.time() - start_time > timeout_sec:
                            return 1 # 1 반환 시 쿼리 강제 중단
                        return 0
                    
                    # 1000 opcode마다 체크
                    raw_conn.set_progress_handler(progress_handler, 1000)
                    return raw_conn # 해제를 위해 객체 반환
            
            return None

    def _calculate_r_ves(self, ex_score, gt_sql, pred_sql, engine, iterations=20):
        if ex_score != 1.0:
            return 0.0

        def get_clean_average_time(sql, db_engine, runs):
            times = []
            try:
                with db_engine.connect() as conn:
                    self._setup_db_timeout(conn, db_engine, timeout_sec=20.0)

                    # 워밍업
                    conn.execute(text(sql)).fetchall()

                    for _ in range(runs):
                        start = time.perf_counter()
                        conn.execute(text(sql)).fetchall()
                        end = time.perf_counter()
                        times.append(end - start)
            except Exception as e:
                return float('inf')
            
            if not times:
                return float('inf')
        
            q1 = np.percentile(times, 25)
            q3 = np.percentile(times, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            clean_times = [t for t in times if lower_bound <= t <= upper_bound]
            if not clean_times:
                return np.mean(times)
        
            return np.mean(clean_times)
        
        avg_gt_time = get_clean_average_time(gt_sql, engine, iterations)
        avg_pred_time = get_clean_average_time(pred_sql, engine, iterations)

        epsilon = 1e-9
        tau = avg_gt_time / (avg_pred_time + epsilon)

        if tau >= 2.0: return 1.25
        elif 1.0 <= tau < 2.0: return 1.0
        elif 0.5 <= tau < 1.0: return 0.75
        elif 0.25 <= tau < 0.5: return 0.5
        else: return 0.25

    def _normalize_val_for_soft_f1(self, val):
        s = str(val).lower().strip()
        try:
            f_val = float(s)
            if f_val.is_integer():
                return str(int(f_val))
            return str(f_val)
        except:
            return s.replace(" ", "")
    
    def _get_row_multiset(self, row):
        clean_values = []
        for val in row:
            if val is None or str(val).lower == 'none':
                continue
            clean_values.append(self._normalize_val_for_soft_f1(val))
        return Counter(clean_values)

    def _calculate_soft_f1(self, gt_res, pred_res):
        def sort_key(row):
            return tuple(sorted([str(x) for x in row if x is not None]))
        
        gt_rows = sorted(list(gt_res), key=sort_key)
        pred_rows = sorted(list(pred_res), key=sort_key)

        total_tp = 0
        total_fp = 0
        total_fn = 0

        max_len = max(len(gt_rows), len(pred_rows))

        for i in range(max_len):
            # GT Row가 없으면 (Pred가 더 많은 경우) -> 전체가 FP (Pred Only)
            if i >= len(gt_rows):
                pred_counter = self._get_row_multiset(pred_rows[i])
                total_fp += sum(pred_counter.values())
                continue
            
            # Pred Row가 없으면 (GT가 더 많은 경우) -> 전체가 FN (Gold Only)
            if i >= len(pred_rows):
                gt_counter = self._get_row_multiset(gt_rows[i])
                total_fn += sum(gt_counter.values())
                continue

            # 둘 다 있는 경우 -> 교집합(Matched) 계산
            gt_counter = self._get_row_multiset(gt_rows[i])
            pred_counter = self._get_row_multiset(pred_rows[i])

            # Intersection (Matched)
            # Counter의 교집합(&)은 각 요소의 min 개수를 취함
            intersection = gt_counter & pred_counter
            tp = sum(intersection.values())

            # Pred Only (FP) = Pred - Intersection
            # Counter의 뺄셈(-)은 양수 값만 남김
            pred_only = pred_counter - intersection
            fp = sum(pred_only.values())

            # Gold Only (FN) = GT - Intersection
            gold_only = gt_counter - intersection
            fn = sum(gold_only.values())

            total_tp += tp
            total_fp += fp
            total_fn += fn

        # 3. Precision, Recall, F1 계산
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        
        f1 = 0.0
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            
        return round(f1, 4)

    def _execute_sql(self, sql: str, engine):
        start_time = time.time()
        result_set = set()
        error = None

        if not engine:
            return set(), 0.0, "No Engine Provided"

        try:
            with engine.connect() as conn:
                raw_conn = self._setup_db_timeout(conn, engine, timeout_sec=10.0)

                result = conn.execute(text(sql))
                rows = result.fetchall()

                if len(rows) == 0:
                    logger.warning(f"[WARNING] Zero Rows Returned! SQL: {sql}")

                result_set = set(tuple(row) for row in rows)

                # 핸들러 해제
                if raw_conn:
                    raw_conn.set_progress_handler(None, 1000)
                
        except Exception as e:
            result_set = set()
            error = str(e)
            if "interrupted" in error.lower():
                error = "Execution Timeout (10s limit exceeded)"
            logger.error(f"[ERROR] SQL Execution Failed: {error} | Query: {sql}")
        
        exec_time = time.time() - start_time
        return result_set, exec_time, error

    def evaluate_schema(self, gt_list, pred_list):
        gt_set = self._normalize_set(gt_list)
        pred_set = self._normalize_set(pred_list)
        
        tp = len(gt_set.intersection(pred_set))
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        union_len = len(gt_set.union(pred_set))
        jaccard = tp / union_len if union_len > 0 else 0.0

        return {
            "schema_precision": round(precision, 4),
            "schema_recall": round(recall, 4),
            "schema_f1": round(f1, 4),
            "schema_jaccard": round(jaccard, 4),
            "schema_tp_count": tp,
            "schema_fp_count": fp,
            "schema_fn_count": fn,
            "missing_cols": list(gt_set - pred_set),
            "extra_cols": list(pred_set - gt_set)
        }
    
    def evaluation_execution(self, gt_sql, pred_sql, engine):
        # 1. EM
        norm_gt = self._normalize_sql(gt_sql)
        norm_pred = self._normalize_sql(pred_sql)
        em = 1.0 if norm_gt == norm_pred else 0.0

        if not engine:
            return {"ex": 0.0, "em": em, "r_ves": 0.0, "exec_soft_f1": 0.0, "exec_error": "No DB Engine"}
        
        # 2. Execution
        gt_res, gt_time, gt_err = self._execute_sql(gt_sql, engine)
        pred_res, pred_time, pred_err = self._execute_sql(pred_sql, engine)

        final_err = None
        if gt_err: final_err = f"GT Error: {gt_err}"
        if pred_err: final_err = f"Pred Error: {pred_err}"

        # 3. EX
        ex = 0.0
        if not pred_err and not gt_err:
            gt_list = list(gt_res)
            pred_list = list(pred_res)
            match_count = 0

            # Exact Match First
            if gt_res == pred_res:
                ex = 1.0
            else:
                # Fuzzy Match
                if len(gt_res) == len(pred_res):
                    for p_row in pred_list:
                        for g_row in gt_list:
                            if self._fuzzy_row_match(g_row, p_row):
                                match_count += 1
                                break
                    
                    if match_count == len(pred_list):
                        ex = 1.0
                    else:
                        final_err = "Mismatch (Fuzzy check failed)"
                else:
                    final_err = f"Row Count Mismatch: GT={len(gt_res)}, Pred={len(pred_res)}"

        # 4. Soft-F1
        soft_f1 = 0.0
        if not final_err and not pred_err:
            if len(gt_res) > 0 or len(pred_res) > 0:
                soft_f1 = self._calculate_soft_f1(gt_res, pred_res)
        
        # 5. R-VES (Only if EX == 1.0)
        r_ves = 0.0
        if ex == 1.0:
            r_ves = self._calculate_r_ves(ex, gt_sql, pred_sql, engine, iterations=20)

        return {
            "ex": ex,
            "em": em,
            "r_ves": round(r_ves, 4),
            "exec_soft_f1": round(soft_f1, 4),
            "pred_exec_time": round(pred_time, 4),
            "exec_error": final_err
        }
    
    def evaluate_single(self, gt_schema, pred_schema, gt_sql, pred_sql, db_engine=None, db_path=None):
        schema_metrics = self.evaluate_schema(gt_schema, pred_schema)

        # 엔진이 없으면 db_path로 임시 생성
        local_engine = db_engine
        created_locally = False

        if not local_engine and db_path:
            try:
                local_engine = create_engine(f"sqlite:///{db_path}")
                created_locally = True
            except Exception as e:
                logger.error(f"Failed to create engine from path {db_path}: {e}")

        exec_metrics = self.evaluation_execution(gt_sql, pred_sql, local_engine)

        if created_locally and local_engine:
            local_engine.dispose()

        return {**schema_metrics, **exec_metrics}

    def evaluate_batch(self, results):
        df_log = []
        metric_keys = ["schema_precision", "schema_recall", "schema_f1", "ex", "em", "r_ves", "exec_soft_f1"]
        total_metrics = {k: 0.0 for k in metric_keys}
        count = len(results)

        for res in results:
            # [수정] db_path를 정확히 전달
            metrics = self.evaluate_single(
                gt_schema=res.get('gt_schema', []),
                pred_schema=res.get('pred_schema', []),
                gt_sql=res.get('gt_sql', ""),
                pred_sql=res.get('pred_sql', ""),
                db_engine=res.get('db_engine', None),
                db_path=res.get('db_path', None)
            )

            log_entry = {"question": res.get("question", ""), **metrics}
            df_log.append(log_entry)

            for k in metric_keys:
                total_metrics[k] += metrics.get(k, 0.0)
        
        avg_metrics = {k: round(v / count, 4) for k, v in total_metrics.items()} if count > 0 else total_metrics
        return avg_metrics, pd.DataFrame(df_log)
