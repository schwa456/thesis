import networkx as nx
from abc import ABC, abstractmethod

class BaseGraphBuilder(ABC):
    @abstractmethod
    def build_graph(self, db_id, table_metadata):
        pass

class SimpleFKGraphBuilder(BaseGraphBuilder):
    """ FK 기반 단순 Table Graph (Pilot 용) """
    def build_graph(self, db_id, table_metadata):
        G = nx.Graph()
        
        # [수정됨] 키값 안전하게 가져오기 (table_names_original 또는 table_names)
        tables = table_metadata.get('table_names_original')
        if not tables:
            tables = table_metadata.get('table_names', [])
        
        # 그래도 없으면 로그 출력 (디버깅용)
        if not tables:
            print(f"[Warning] No tables found for DB: {db_id}")
            return G, []

        G.add_nodes_from(tables)
        
        # FK 연결 로직
        fks = table_metadata.get('foreign_keys', [])
        column_names = table_metadata.get('column_names_original')
        if not column_names:
             column_names = table_metadata.get('column_names', [])

        if column_names: # 컬럼 정보가 있을 때만 엣지 연결
            for fk in fks:
                try:
                    src_table_idx = column_names[fk[0]][0]
                    tgt_table_idx = column_names[fk[1]][0]
                    
                    src_table = tables[src_table_idx]
                    tgt_table = tables[tgt_table_idx]
                    
                    if src_table != tgt_table:
                        G.add_edge(src_table, tgt_table, weight=1.0)
                except IndexError:
                    continue
                except Exception as e:
                    # 혹시 모를 에러 무시
                    continue
                
        return G, tables

class SemanticGraphBuilder(BaseGraphBuilder):
    """TODO: 고도화 예정, Edge에 Text Label을 부여하여 G-Retriever 사용 가능하게끔 """
    def build_graph(self, db_id, table_metadata):
        pass # TODO

    