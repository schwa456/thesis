import pickle
import networkx as nx

# 1. 파일 경로 설정 (SemanticGraphBuilder에서 설정한 경로)
PKL_PATH = "../../data/cache_graphs/lg55_db_semantic.pkl" 

def inspect_graph():
    try:
        # 2. Pickle 파일 로드
        print(f"📂 Loading graph from {PKL_PATH}...")
        with open(PKL_PATH, 'rb') as f:
            saved_data = pickle.load(f)
        
        # 저장 구조가 {'G': G, 'tables': tables} 였으므로 G만 꺼냅니다.
        G = saved_data['G']
        
        print(f"✅ Graph Loaded! Nodes: {len(G.nodes)}, Edges: {len(G.edges)}")
        print("-" * 50)

        # 3. Textual Label이 있는 Edge만 찾아서 출력
        found_labels = False

        # G.edges(data=True)를 하면 (u, v, attribute_dict)가 나옵니다.
        for u, v, data in G.edges(data=True):
            
            # 우리가 생성한 'textual_label' 키가 있는지 확인
            if 'textual_label' in data:
                found_labels = True
                relation_type = data.get('relation', 'unknown')
                label = data['textual_label']
                
                print(f"🔗 [Edge] {u} -> {v}")
                print(f"   📌 Type: {relation_type}")
                print(f"   📝 Label: \"{label}\"")
                print("-" * 50)
                
        
        if not found_labels:
            print("⚠️ 'textual_label' 속성을 가진 Edge가 하나도 없습니다!")
            print("   -> SemanticGraphBuilder가 제대로 실행되었는지, LLM 키가 있었는지 확인해보세요.")

    except FileNotFoundError:
        print(f"❌ Error: 파일을 찾을 수 없습니다: {PKL_PATH}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    inspect_graph()
