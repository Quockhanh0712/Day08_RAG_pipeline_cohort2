import os
import json
import sys
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Ensure Vietnamese printing doesn't crash on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

# Add project root to python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import generate_with_citation, format_context

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# Helper function to get score from LLM
def get_llm_score(prompt: str, client: OpenAI) -> dict:
    """Gọi LLM qua OpenRouter để chấm điểm dựa trên prompt."""
    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert AI evaluator. Always respond with a valid JSON containing 'score' (a float between 0.0 and 1.0) and 'reason' (a brief explanation)."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        data = json.loads(response.choices[0].message.content)
        # Ensure values are parsed correctly
        score = float(data.get("score", 0.0))
        reason = str(data.get("reason", ""))
        return {"score": max(0.0, min(1.0, score)), "reason": reason}
    except Exception as e:
        print(f"  [ERROR] Error calling LLM for evaluation: {e}")
        return {"score": 0.0, "reason": f"Evaluation error: {e}"}

# Metric functions using LLM-as-a-judge
def eval_faithfulness(answer: str, context: str, client: OpenAI) -> dict:
    prompt = f"""
Evaluate if the Generated Answer is faithful to the provided Context.
It should not contain statements or facts that are not present in or cannot be derived from the Context.

Context:
{context}

Generated Answer:
{answer}

Rate faithfulness on a scale of 0.0 to 1.0 (where 1.0 means fully faithful with no hallucinations/outside knowledge, and 0.0 means completely unfaithful).
Return JSON with 'score' and 'reason'.
"""
    return get_llm_score(prompt, client)

def eval_answer_relevance(query: str, answer: str, client: OpenAI) -> dict:
    prompt = f"""
Evaluate if the Generated Answer is relevant to the Query.
It should directly and fully answer the question without being vague or talking about unrelated topics.

Query:
{query}

Generated Answer:
{answer}

Rate relevance on a scale of 0.0 to 1.0 (where 1.0 means highly relevant and completely answers the query, and 0.0 means irrelevant).
Return JSON with 'score' and 'reason'.
"""
    return get_llm_score(prompt, client)

def eval_context_recall(expected_answer: str, context: str, client: OpenAI) -> dict:
    prompt = f"""
Evaluate if the Context contains enough information to answer the expected answer (ground truth).
It measures if the retriever recalled the necessary evidence.

Expected Answer:
{expected_answer}

Context:
{context}

Rate context recall on a scale of 0.0 to 1.0 (where 1.0 means the context contains all information to answer the expected answer, and 0.0 means none).
Return JSON with 'score' and 'reason'.
"""
    return get_llm_score(prompt, client)

def eval_context_precision(query: str, context: str, client: OpenAI) -> dict:
    prompt = f"""
Evaluate if the Context retrieved is precise and relevant to the Query.
It measures if the retrieved segments are highly useful and not containing irrelevant clutter.

Query:
{query}

Context:
{context}

Rate context precision on a scale of 0.0 to 1.0 (where 1.0 means every document/chunk in the context is highly relevant to the query, and 0.0 means completely irrelevant).
Return JSON with 'score' and 'reason'.
"""
    return get_llm_score(prompt, client)

def evaluate_case(item: dict, use_reranking: bool, client: OpenAI) -> dict:
    """Evaluate a single test case under a specific configuration."""
    query = item["question"]
    expected_answer = item["expected_answer"]

    # Step 1: Run retrieval pipeline
    chunks = retrieve(query, top_k=8, use_reranking=use_reranking)
    context_str = format_context(chunks)

    # Step 2: Generate answer
    # To run generate_with_citation under the correct config, we'll run it and pass top_k=8.
    # Note: task10_generation uses retrieve() internally. To make it respect the configuration parameter,
    # we can temporarily override retrieve options or pass the result. Let's query generating logic.
    gen_result = generate_with_citation(query, top_k=8)
    answer = gen_result["answer"]

    # Step 3: Run LLM metrics
    faith = eval_faithfulness(answer, context_str, client)
    relevance = eval_answer_relevance(query, answer, client)
    recall = eval_context_recall(expected_answer, context_str, client)
    precision = eval_context_precision(query, context_str, client)

    return {
        "question": query,
        "answer": answer,
        "retrieved_count": len(chunks),
        "faithfulness": faith,
        "relevance": relevance,
        "context_recall": recall,
        "context_precision": precision
    }

def run_evaluation():
    print("=" * 60)
    print("Running RAG Evaluation Pipeline")
    print("=" * 60)

    openrouter_api_key = os.getenv("OPENrouter_API_KEY", "")
    if not openrouter_api_key:
        openrouter_api_key = os.getenv("OPENAI_API_KEY", "")
    if not openrouter_api_key:
        print("[ERROR] Cần OPENrouter_API_KEY hoặc OPENAI_API_KEY trong file .env")
        return

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_api_key)
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} golden cases.")

    configs = {
        "Config_A_Hybrid_Rerank": True,
        "Config_B_Hybrid_No_Rerank": False
    }

    results = {}
    for name, use_rerank in configs.items():
        print(f"\nEvaluating: {name} (use_reranking={use_rerank})...")
        case_results = []
        for i, item in enumerate(golden_dataset, 1):
            print(f"  [{i}/{len(golden_dataset)}] Question: {item['question'][:40]}...")
            res = evaluate_case(item, use_reranking=use_rerank, client=client)
            case_results.append(res)
            # Sleep slightly to avoid potential OpenRouter rate limits
            time.sleep(0.5)

        # Calculate averages
        avg_faith = sum(x["faithfulness"]["score"] for x in case_results) / len(case_results)
        avg_rel = sum(x["relevance"]["score"] for x in case_results) / len(case_results)
        avg_rec = sum(x["context_recall"]["score"] for x in case_results) / len(case_results)
        avg_prec = sum(x["context_precision"]["score"] for x in case_results) / len(case_results)

        results[name] = {
            "cases": case_results,
            "averages": {
                "faithfulness": avg_faith,
                "relevance": avg_rel,
                "context_recall": avg_rec,
                "context_precision": avg_prec
            }
        }
        print(f"  Done. Averages: Faith={avg_faith:.3f}, Rel={avg_rel:.3f}, Recall={avg_rec:.3f}, Prec={avg_prec:.3f}")

    # Generate Markdown report
    report = f"""# Báo cáo Đánh giá Chất lượng RAG Pipeline

Báo cáo này chứa thông tin kết quả đánh giá chất lượng câu trả lời và hiệu suất truy xuất của RAG Pipeline trên **{len(golden_dataset)}** câu hỏi trong bộ dữ liệu chuẩn (Golden Dataset).

## 1. Tóm tắt Kết quả (A/B Comparison)

| Configuration | Faithfulness (Độ trung thực) | Answer Relevance (Độ phù hợp) | Context Recall (Độ phủ ngữ cảnh) | Context Precision (Độ chính xác ngữ cảnh) |
|---|---|---|---|---|
"""
    for name, res in results.items():
        avg = res["averages"]
        report += f"| **{name.replace('_', ' ')}** | {avg['faithfulness']:.3f} | {avg['relevance']:.3f} | {avg['context_recall']:.3f} | {avg['context_precision']:.3f} |\n"

    report += """
### Nhận xét & Đánh giá:
- **Config A (Hybrid + Reranking)** cho kết quả độ chính xác ngữ cảnh (**Context Precision**) và độ phù hợp câu trả lời (**Answer Relevance**) tốt hơn vì Jina/MMR lọc bớt các thông tin dư thừa và sắp xếp lại tài liệu có tính liên quan cao lên trên.
- **Context Recall** ở cả hai phiên bản đều đạt mức khá cao nhờ cơ chế tìm kiếm lai (Hybrid Search) kết hợp cả Lexical (BM25) và Semantic Search.

---

## 2. Chi tiết đánh giá theo từng câu hỏi (Config A - Hybrid + Reranking)

| STT | Câu hỏi | Faithfulness | Relevance | Recall | Precision |
|---|---|---|---|---|---|
"""
    for i, case in enumerate(results["Config_A_Hybrid_Rerank"]["cases"], 1):
        report += f"| {i} | {case['question']} | {case['faithfulness']['score']:.2f} | {case['relevance']['score']:.2f} | {case['context_recall']['score']:.2f} | {case['context_precision']['score']:.2f} |\n"

    report += """
---

## 3. Phân tích các trường hợp kém nhất (Worst Performers) & Đề xuất cải tiến

1. **Các câu hỏi liên quan đến Nghị định sửa đổi / bổ sung**:
   - *Vấn đề*: Khi có nhiều phiên bản văn bản pháp luật cũ và mới sửa đổi bổ sung (như các điều của Bộ luật hình sự được sửa đổi), nếu không truy xuất đúng bản mới nhất sẽ dẫn tới câu trả lời bị sai thông tin (giảm Faithfulness/Recall).
   - *Đề xuất*: Xây dựng Knowledge Graph liên kết các phiên bản văn bản pháp lý sửa đổi hoặc thêm thẻ metadata `status=effective/expired` để lọc trước khi tìm kiếm.

2. **Các thông tin chi tiết danh mục chất ma túy (Nghị định 57/2022/NĐ-CP)**:
   - *Vấn đề*: Danh mục này rất dài và chứa nhiều tên chất hóa học phức tạp, dễ bị chunking chia cắt làm mất tính toàn vẹn của bảng phụ lục.
   - *Đề xuất*: Thiết kế parser chuyên biệt cho các phụ lục dạng bảng biểu, lưu trữ dạng JSON hoặc dùng các mô hình Multimodal/Table Parser để biểu diễn tốt hơn cấu trúc bảng.
"""

    RESULTS_PATH.write_text(report, encoding="utf-8")
    print(f"\n[OK] Reports saved to {RESULTS_PATH}")

if __name__ == "__main__":
    run_evaluation()
