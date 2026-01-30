import os
import json
import sqlglot
import logging
from sqlglot import exp

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.mode = config['data']['mode']

    def load_data(self):
        if self.mode == 'bird':
            return self._load_bird()
        elif self.mode == 'lg55':
            return self._load_lg55()
        else:
            raise ValueError(f"Unknown data mode: {self.mode}")
    
    def _load_bird_tables(self):
        """dev_tables.json을 읽어 DB ID별 메타데이터 매핑"""
        table_path = self.config['data']['bird']['dev_table_path'] # dev.json 경로
        
        if not os.path.exists(table_path):
            return {}

        with open(table_path, 'r', encoding='utf-8') as f:
            tables_list = json.load(f)
        
        return {t['db_id']: t for t in tables_list}

    def _load_bird(self):
        if self.mode != 'bird':
            return []

        data_path = self.config['data']['bird']['dev_json_path']
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        tables_metadata = self._load_bird_tables()
        
        processed_data = []
        for item in raw_data:
            db_id = item['db_id']
            processed_data.append({
                "db_id": db_id,
                "question": item['question'],
                "evidence": item.get('evidence', ''),
                "sql": item['SQL'],
                "gt_schema": self._convert_sql_to_schema(item['SQL']),
                "meta_schema": tables_metadata.get(db_id, {}) 
            })
        
        return processed_data
    
    def _load_lg55(self):
        json_path = self.config['data']['lg55']['question_path']
        logger.info(f"Loading LG55 data from {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        processed_data = []

        for i, item in enumerate(data):
            processed_data.append({
                "question_id": item.get("idx", i),
                "question": item["question"],
                "gt_schema": item.get("gold_schema"),
                "db_id": "lg55_db",
                "evidence": "",
                "sql": item.get("gold_sql"),
            })
        
        logger.info(f"Loaded {len(processed_data)} questions from LG55.")
        return processed_data
    
    def _convert_sql_to_schema(self, query):
        try:
            parsed = sqlglot.parse_one(query, read="mysql")
        except Exception as e:
            logger.error(f"SQL Parsing Failed: {e}")
            return []
        
        alias_map = {}
        tables_found = []

        for table in parsed.find_all(exp.Table):
            real_name = table.name
            alias = table.alias_or_name

            alias_map[alias] = real_name

            tables_found.append(real_name)

            unique_tables = list(set(tables_found))

            extracted_columns = set()

            for col in parsed.find_all(exp.Column):
                col_name = col.name
                table_part = col.table

                final_table_name = None

                if table_part:
                    final_table_name = alias_map.get(table_part, table_part)
                else:
                    if len(unique_tables) == 1:
                        final_table_name = unique_tables[0]
                    else:
                        final_table_name = "UNKNOWN"
                
                if final_table_name and final_table_name != "UNKNOWN":
                    extracted_columns.add(f"{final_table_name}.{col_name}")
        sorted_list = sorted(list(extracted_columns))
        
        return json.dumps(sorted_list)
