"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# CHUNK_SIZE = 500: Chọn 500 vì độ dài này vừa đủ chứa nội dung của 1 điều luật ngắn
# hoặc một vài khoản trong điều luật dài, tối ưu cho ngữ cảnh và khả năng đọc của LLM.
CHUNK_SIZE = 500        
# CHUNK_OVERLAP = 50: Chọn 50 để giữ tính liên tục giữa các đoạn trích dẫn.
CHUNK_OVERLAP = 50      
CHUNKING_METHOD = "custom_regex"  # "custom_regex" dùng biểu thức chính quy tách theo Điều luật

# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2": Model nhẹ, chạy nhanh cục bộ,
# tương thích tốt với các bài kiểm thử và môi trường không dùng GPU.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  
EMBEDDING_DIM = 384

VECTOR_STORE = "weaviate"  


# =============================================================================
# IMPLEMENTATION
# =============================================================================

import sys
import re
import os
import json
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        if md_file.is_file() and md_file.name != ".gitkeep":
            content = md_file.read_text(encoding="utf-8")
            doc_type = "legal" if "legal" in str(md_file) else "news"
            documents.append({
                "content": content,
                "metadata": {"source": md_file.name, "type": doc_type}
            })
    return documents


def chunk_legal_document(doc: dict) -> list[dict]:
    """
    Tách tài liệu pháp luật theo tiêu chí: tách chính xác từng 'Điều' bằng Regex.
    Nếu điều luật quá dài (> CHUNK_SIZE), sử dụng RecursiveCharacterTextSplitter để chia nhỏ
    nhưng vẫn giữ nguyên tiêu đề của điều luật ở đầu mỗi chunk con.
    """
    content = doc["content"]
    # Tách theo dòng bắt đầu bằng chữ "Điều" hoặc dòng giới thiệu sửa đổi bổ sung điều luật
    parts = re.split(r'(?m)^([#\s“"*]*Điều\s+\d+.*?|\d+\.\s+Sửa\s+đổi,\s+bổ\s+sung.*?Điều\s+\d+.*?)$', content)
    
    chunks = []
    # Phần giới thiệu/mở đầu trước Điều 1
    preamble = parts[0].strip()
    if preamble:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        splits = splitter.split_text(preamble)
        for split in splits:
            chunks.append({
                "content": split,
                "metadata": {
                    **doc["metadata"],
                    "chunk_index": len(chunks),
                    "article": "Lời nói đầu"
                }
            })
            
    # Các Điều luật tiếp theo
    for j in range(1, len(parts), 2):
        title = parts[j].strip()
        text = parts[j+1].strip() if j+1 < len(parts) else ""
        
        # Trích xuất số Điều (ví dụ: "Điều 1")
        article_match = re.search(r'Điều\s+(\d+)', title)
        article_name = f"Điều {article_match.group(1)}" if article_match else title
        
        full_content = f"{title}\n{text}".strip()
        if len(full_content) <= CHUNK_SIZE:
            chunks.append({
                "content": full_content,
                "metadata": {
                    **doc["metadata"],
                    "chunk_index": len(chunks),
                    "article": article_name
                }
            })
        else:
            # Điều luật quá dài -> Tách nhỏ phần text của điều đó và chèn tiêu đề điều luật ở đầu
            max_text_len = max(50, CHUNK_SIZE - len(title) - 2)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_text_len,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            splits = splitter.split_text(text)
            for split in splits:
                chunks.append({
                    "content": f"{title}\n{split}".strip(),
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": len(chunks),
                        "article": article_name
                    }
                })
    return chunks


def chunk_news_document(doc: dict) -> list[dict]:
    """
    Tách bài báo tin tức bằng RecursiveCharacterTextSplitter thông thường.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    splits = splitter.split_text(doc["content"])
    chunks = []
    for i, split in enumerate(splits):
        chunks.append({
            "content": split,
            "metadata": {
                **doc["metadata"],
                "chunk_index": i,
                "article": "N/A"
            }
        })
    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy phù hợp cho từng loại.
    """
    chunks = []
    for doc in documents:
        if doc["metadata"]["type"] == "legal":
            chunks.extend(chunk_legal_document(doc))
        else:
            chunks.extend(chunk_news_document(doc))
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn cục bộ.
    """
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def save_local_vectorstore(chunks: list[dict]):
    """
    Lưu chunks cục bộ vào file json làm fallback khi Weaviate Cloud không sẵn sàng.
    """
    filepath = Path(__file__).parent.parent / "data" / "vectorstore.json"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    # Lưu bản copy không chứa embedding (nếu muốn tiết kiệm diện tích) hoặc chứa cả
    filepath.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [OK] Saved to local fallback vectorstore: {filepath}")


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store Weaviate Cloud.
    Nếu thất bại sẽ fallback tự động ghi file cục bộ.
    """
    # Lưu cục bộ trước để làm fallback chắc chắn cho Task 5, 6
    save_local_vectorstore(chunks)
    
    weaviate_url = os.getenv("WEAVIATE_URL", "")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY", "")
    
    # Kiểm tra xem cấu hình thật hay chỉ là placeholder "xxx"
    is_placeholder = not weaviate_url or "xxx" in weaviate_url
    if is_placeholder:
        print("[WARNING] WEAVIATE_URL đang là placeholder. Bỏ qua index lên cloud, sử dụng local fallback.")
        return
        
    try:
        import weaviate
        from weaviate.classes.config import Property, DataType, Configure
        
        print(f"Connecting to Weaviate Cloud at {weaviate_url}...")
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key)
        )
        
        # Tạo hoặc ghi đè collection
        collection_name = "DrugLawDocs"
        if client.collections.exists(collection_name):
            client.collections.delete(collection_name)
            
        collection = client.collections.create(
            name=collection_name,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="type", data_type=DataType.TEXT),
                Property(name="article", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
            ]
        )
        
        # Thêm các đối tượng theo batch
        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                batch.add_object(
                    properties={
                        "content": chunk["content"],
                        "source": chunk["metadata"].get("source", ""),
                        "type": chunk["metadata"].get("type", ""),
                        "article": chunk["metadata"].get("article", "N/A"),
                        "chunk_index": chunk["metadata"].get("chunk_index", 0),
                    },
                    vector=chunk["embedding"]
                )
        print("  [OK] Indexed to Weaviate Cloud successfully.")
        client.close()
    except Exception as e:
        print(f"[WARNING] Khong the ket noi/index toi Weaviate Cloud: {e}")
        print("He thong se chay hoan toan bang du lieu local fallback.")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n[OK] Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"[OK] Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"[OK] Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("[OK] Indexing completed.")


if __name__ == "__main__":
    run_pipeline()
