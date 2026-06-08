"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


import os
import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def local_semantic_search(query: str, top_k: int) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng NumPy tính cosine similarity cục bộ.
    """
    filepath = Path(__file__).parent.parent / "data" / "vectorstore.json"
    if not filepath.exists():
        print("[WARNING] Khong tim thay local vector store. Hay chay task4 truoc!")
        return []
        
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except Exception as e:
        print(f"[ERROR] Loi doc file vectorstore: {e}")
        return []
        
    if not chunks:
        return []
        
    from sentence_transformers import SentenceTransformer
    # Load model nhẹ
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_emb = model.encode(query)
    
    results = []
    for chunk in chunks:
        if "embedding" not in chunk:
            continue
        emb = np.array(chunk["embedding"])
        # Tránh chia cho 0
        norm_query = np.linalg.norm(query_emb)
        norm_emb = np.linalg.norm(emb)
        if norm_query == 0 or norm_emb == 0:
            score = 0.0
        else:
            score = float(np.dot(query_emb, emb) / (norm_query * norm_emb))
            
        results.append({
            "content": chunk["content"],
            "score": score,
            "metadata": {
                "source": chunk["metadata"].get("source", ""),
                "type": chunk["metadata"].get("type", ""),
                "article": chunk["metadata"].get("article", "N/A"),
                "chunk_index": chunk["metadata"].get("chunk_index", 0),
            }
        })
        
    # Sắp xếp giảm dần theo điểm tương đồng
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def weaviate_semantic_search(query: str, top_k: int) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng Weaviate Cloud.
    """
    weaviate_url = os.getenv("WEAVIATE_URL", "")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY", "")
    
    if not weaviate_url or "xxx" in weaviate_url:
        raise ValueError("WEAVIATE_URL is placeholder or empty")
        
    import weaviate
    from weaviate.classes.query import MetadataQuery
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = model.encode(query).tolist()
    
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key)
    )
    
    try:
        collection = client.collections.get("DrugLawDocs")
        results = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True)
        )
        
        output = []
        for obj in results.objects:
            distance = obj.metadata.distance if obj.metadata.distance is not None else 1.0
            # similarity score: cosine distance = 1 - similarity -> similarity = 1 - distance
            score = 1.0 - distance
            output.append({
                "content": obj.properties["content"],
                "score": float(score),
                "metadata": {
                    "source": obj.properties.get("source", ""),
                    "type": obj.properties.get("type", ""),
                    "article": obj.properties.get("article", "N/A"),
                    "chunk_index": obj.properties.get("chunk_index", 0),
                }
            })
        client.close()
        return output
    except Exception as e:
        client.close()
        raise e


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity (Weaviate Cloud / Local fallback).

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, type, chunk_index, article
        }
        Sorted by score descending.
    """
    try:
        return weaviate_semantic_search(query, top_k)
    except Exception as e:
        print(f"[WARNING] Weaviate Cloud search failed ({e}). Fallback to local numpy search...")
        return local_semantic_search(query, top_k)


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
