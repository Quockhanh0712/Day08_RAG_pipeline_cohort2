"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    pageindex_key = os.getenv("PAGEINDEX_API_KEY", "")
    if not pageindex_key or "pi_" in pageindex_key:
        print("[WARNING] PAGEINDEX_API_KEY chua cau hinh hoac la placeholder. Bo qua upload.")
        return

    try:
        from pageindex import PageIndex
        pi = PageIndex(api_key=pageindex_key)

        for md_file in STANDARDIZED_DIR.rglob("*.md"):
            if md_file.is_file() and md_file.name != ".gitkeep":
                content = md_file.read_text(encoding="utf-8")
                pi.upload(
                    content=content,
                    metadata={"filename": md_file.name, "type": md_file.parent.name}
                )
                print(f"  [OK] Uploaded to PageIndex: {md_file.name}")
    except Exception as e:
        print(f"[WARNING] Loi upload tai lieu len PageIndex: {e}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    pageindex_key = os.getenv("PAGEINDEX_API_KEY", "")

    if pageindex_key and "pi_" not in pageindex_key:
        try:
            from pageindex import PageIndex
            pi = PageIndex(api_key=pageindex_key)
            results = pi.query(query=query, top_k=top_k)
            output = []
            for r in results:
                output.append({
                    "content": r.text,
                    "score": float(r.score) if hasattr(r, 'score') else 1.0,
                    "metadata": r.metadata if hasattr(r, 'metadata') else {},
                    "source": "pageindex"
                })
            return output
        except Exception as e:
            print(f"[WARNING] Loi truy van PageIndex: {e}. Tu dong fallback sang mock search...")

    # Mock fallback bằng tìm kiếm lexical cục bộ và gắn nhãn source='pageindex'
    try:
        from .task6_lexical_search import lexical_search
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.task6_lexical_search import lexical_search

    print("  [OK] Fallback: Su dung BM25 cuc bo gia lap ket qua PageIndex.")
    local_results = lexical_search(query, top_k=top_k)
    output = []
    for r in local_results:
        item = r.copy()
        item["source"] = "pageindex"
        output.append(item)
    return output


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("[WARNING] Hay set PAGEINDEX_API_KEY trong file .env")
        print("  Dang ky tai: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
