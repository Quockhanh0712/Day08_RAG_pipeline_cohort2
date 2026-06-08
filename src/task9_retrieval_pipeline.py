"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3   # Nếu best score < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "mmr"  # "cross_encoder" | "mmr" | "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → results_dense
          ├→ Lexical Search  → results_sparse
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If best_score < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm tối thiểu cho hybrid results
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Step 1: Chạy song song semantic + lexical
    # Mở rộng pool tìm kiếm để đảm bảo bắt được các điều luật cụ thể
    search_pool = top_k * 8
    dense_results = semantic_search(query, top_k=search_pool)
    sparse_results = lexical_search(query, top_k=search_pool)

    # Step 2: Merge kết quả bằng RRF
    merged = rerank_rrf([dense_results, sparse_results], top_k=search_pool)

    # Chuẩn hóa điểm RRF về khoảng [0, 1]
    max_possible_rrf = 2.0 / (60 + 1)
    for item in merged:
        item["source"] = "hybrid"
        item["score"] = min(1.0, item["score"] / max_possible_rrf)

    # Tối ưu hóa truy xuất cho văn bản pháp luật
    import re
    legal_keywords = ["luật", "hình phạt", "điều", "bộ luật", "nghị định", "thông tư", "tội", "quy định", "pháp luật"]
    is_legal_query = any(k in query.lower() for k in legal_keywords)

    if is_legal_query:
        for item in merged:
            if item.get("metadata", {}).get("type") == "legal":
                item["score"] = min(1.0, item["score"] * 1.5)

    # Ưu tiên tuyệt đối các điều luật khớp chính xác số điều (ví dụ: "Điều 249")
    article_matches = re.findall(r"Điều\s+(\d+)", query, re.IGNORECASE)
    has_exact_match = False
    if article_matches:
        for match in article_matches:
            target = f"Điều {match}"
            for item in merged:
                if item.get("metadata", {}).get("article") == target:
                    item["score"] = 1.0
                    has_exact_match = True

    # Sắp xếp lại sau khi boost
    merged.sort(key=lambda x: x["score"], reverse=True)

    # Step 3: Rerank
    # Khi có điều luật khớp chính xác → KHÔNG dùng MMR (MMR bỏ qua score đã boost,
    # tính lại từ embedding → đẩy điều luật quan trọng xuống dưới).
    # Thay vào đó dùng sorted list theo score trực tiếp.
    if use_reranking and merged and not has_exact_match:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
    else:
        final_results = merged[:top_k]

    # Step 4: Kiểm tra ngưỡng score_threshold -> Fallback sang PageIndex
    if not final_results or final_results[0]["score"] < score_threshold:
        best_score = final_results[0]["score"] if final_results else 0.0
        print(f"  [WARNING] Hybrid score ({best_score:.3f}) < threshold ({score_threshold}). Fallback -> PageIndex")
        fallback = pageindex_search(query, top_k=top_k)
        return fallback

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
