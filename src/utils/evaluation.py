import pandas as pd
import logging
import time
import math
from typing import List, Any, Dict, Set
from sqlalchemy import text

logger = logging.getLogger(__name__)

class SchemaEvaluator:
    def __init__(self, db_path: str = None):
        self.db_path = db_path

    def _normalize_set(self, item_list):
        if not item_list:
            return set()
        return set(item.lower().strip() for item in item_list)

    def _normalize_sql(self, sql: str) -> str:
        if not sql: return ""
        return " ".join(sql.strip().lower().split()).replace(";", "")

    def _normalize_result_set(self, result_set):
        normalized = set()
        for row in result_set:
            sorted_row = tuple(sorted(str(val) for val in row))
            normalized.add(sorted_row)
        return normalized
    
    def _execute_sql(self, sql: str, engine):
        start_time = time.time()
        result_set = set()
        error = None

        try:
            with engine.connect() as conn:
                result = conn.execute(text(sql))
                rows = result.fetchall()

                if len(rows) == 0:
                    logger.warning(f"[WARNING] Zero Rows Returned! SQL: {sql}")

                result_set = set(tuple(row) for row in rows)
                
        except Exception as e:
            result_set = set()
            error = str(e)
            logger.error(f"[ERROR] SQL Execution Failed: {error} | Query: {sql}")
        finally:
            conn.close()
        
        exec_time = time.time() - start_time
        
        return result_set, exec_time, error

    def evaluate_schema(self, gt_list, pred_list):
        gt_set = self._normalize_set(gt_list)
        pred_set = self._normalize_set(pred_list)

        tp_set = gt_set.intersection(pred_set)
        tp = len(tp_set)

        fp_set = pred_set - gt_set
        fp = len(fp_set)

        fn_set = gt_set - pred_set
        fn = len(fn_set)

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
            "schema_fp_count": fp, # Noise
            "schema_fn_count": fn, # Missing (Critical)
            "missing_cols": list(fn_set),
            "extra_cols": list(fp_set)
        }
    
    def evaluation_execution(self, gt_sql, pred_sql, engine):
        # 1. EM
        norm_gt = self._normalize_sql(gt_sql)
        norm_pred = self._normalize_sql(pred_sql)
        em = 1.0 if norm_gt == norm_pred else 0.0

        if not engine:
            return {"ex": 0.0, "em": em, "r_ves": 0.0, "soft_f1": 0.0, "exec_error": "No DB Path"}
        
        # 2, Execution
        gt_res, gt_time, gt_err = self._execute_sql(gt_sql, engine)
        pred_res, pred_time, pred_err = self._execute_sql(pred_sql, engine)

        final_err = None
        if gt_err: final_err = f"GT Error: {gt_err}"
        if pred_err: final_err = f"Pred Error: {pred_err}"

        # 3. EX
        ex = 0.0
        if not pred_err and not gt_err:
            if gt_res == pred_res:
                ex = 1.0
            else:
                norm_gt_res = self._normalize_result_set(gt_res)
                norm_pred_res = self._normalize_result_set(pred_res)
                
                if norm_gt_res == norm_pred_res:
                    ex = 1.0
                else:
                    gt_sample = list(gt_res)[:3]
                    pred_sample = list(pred_res)[:3]
                    final_err = f"Mismatch! GT(len={len(gt_res)}):{gt_sample} vs Pred(len={len(pred_res)}):{pred_sample}"

        
        # 4. Soft-F1
        soft_f1 = 0.0
        if not final_err and (len(gt_res) > 0 or len(pred_res) > 0):
            intersection = len(gt_res.intersection(pred_res))

            if intersection == 0 and ex == 1.0:
                intersection = len(gt_res)

            len_gt = len(gt_res)
            len_pred = len(pred_res)

            p = intersection / len_pred if len_pred > 0 else 0.0
            r = intersection / len_gt if len_gt > 0 else 0.0

            if p + r > 0:
                soft_f1 = 2 * (p * r) / (p + r)
            
            if ex == 1.0:
                soft_f1 = 1.0
        
        # 5. R-VES
        r_ves = 0.0
        if ex == 1.0:
            ratio = (gt_time + 1e-9) / (pred_time + 1e-9)
            r_ves = math.sqrt(ratio)
            r_ves = min(r_ves, 1.0)

        
        return {
            "ex": ex,
            "em": em,
            "r_ves": round(r_ves, 4),
            "exec_soft_f1": round(soft_f1, 4),
            "pred_exec_time": round(pred_time, 4),
            "exec_error": final_err
        }
    
    def evaluate_single(self, gt_schema, pred_schema, gt_sql, pred_sql, db_engine=None):
        schema_metrics = self.evaluate_schema(gt_schema, pred_schema)

        exec_metrics = self.evaluation_execution(gt_sql, pred_sql, db_engine)

        return {**schema_metrics, **exec_metrics}

    def evaluate_batch(self, results):
        df_log = []

        metric_keys = [
            "schema_precision", "schema_recall", "schema_f1", "schema_jaccard",
            "ex", "em", "r_ves", "exec_soft_f1"
        ]

        total_metrics = {k: 0.0 for k in metric_keys}
        count = len(results)

        for res in results:
            metrics = self.evaluate_single(
                gt_schema=res.get('gt_schema', []),
                pred_schema=res.get('pred_schema', []),
                gt_sql=res.get('gt_sql', ""),
                pred_sql=res.get('pred_sql', ""),
                db_path=res.get('db_path', None)
            )

            log_entry = {
                "question": res.get("question", ""),
                **metrics
            }

            df_log.append(log_entry)

            for k in metric_keys:
                total_metrics[k] += metrics.get(k, 0.0)
        
        avg_metrics = {k: round(v / count, 4) for k, v in total_metrics.items()} if count > 0 else total_metrics

        return avg_metrics, pd.DataFrame(df_log)
