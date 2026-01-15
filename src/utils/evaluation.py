from sql_metadata import Parser

def extract_gold_tables(sql_query):

    try:
        tables = Parser(sql_query).tables
        return list(set(tables))
    
    except Exception as e:
        return []
    
def calculate_metrics(gold_tables, predicted_tables):
    gold_set = set(gold_tables)
    pred_set = set(predicted_tables)

    if not gold_set: return 0.0, 0.0
    if not pred_set: return 0.0, 0.0

    tp = len(gold_set.intersection(pred_set))

    recall = tp / len(gold_set)
    precision = tp / len(pred_set)

    return recall, precision