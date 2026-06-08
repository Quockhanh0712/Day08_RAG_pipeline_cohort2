"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

from pathlib import Path

import json
from pathlib import Path
from rank_bm25 import BM25Okapi
import numpy as np

# Cache cho BM25 index và corpus
_bm25 = None
_corpus = None


def load_corpus() -> list[dict]:
    """
    Tải danh sách các chunks từ file vectorstore.json được tạo ra ở Task 4.
    """
    filepath = Path(__file__).parent.parent / "data" / "vectorstore.json"
    if not filepath.exists():
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Loi doc vectorstore: {e}")
        return []


def get_bm25_index():
    """
    Khởi tạo lazy BM25 index từ corpus.
    """
    global _bm25, _corpus
    if _bm25 is None:
        _corpus = load_corpus()
        if not _corpus:
            return None, []
        
        # Tokenize đơn giản bằng split tiếng Việt dạng lowercase
        tokenized_corpus = [doc["content"].lower().split() for doc in _corpus]
        _bm25 = BM25Okapi(tokenized_corpus)
    return _bm25, _corpus


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    bm25, corpus = get_bm25_index()
    if bm25 is None:
        print("[WARNING] BM25 Index chua duoc dung. Hay chay Task 4 de sinh vectorstore.json.")
        return []

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Lấy top_k kết quả có điểm cao nhất
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "content": corpus[idx]["content"],
            "score": float(scores[idx]),
            "metadata": {
                "source": corpus[idx]["metadata"].get("source", ""),
                "type": corpus[idx]["metadata"].get("type", ""),
                "article": corpus[idx]["metadata"].get("article", "N/A"),
                "chunk_index": corpus[idx]["metadata"].get("chunk_index", 0),
            }
        })
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
