# Checklist chuyển đổi RAG (Day 8) sang A2A Multi-Agent

- [x] **Chuẩn bị môi trường & Sao chép các module nền tảng (Base Modules)**
  - [x] Cập nhật `requirements.txt` của Day 8, bổ sung các thư viện A2A (`a2a-sdk[http-server]`, `langgraph`, `uvicorn`, `httpx`, `pydantic`).
  - [x] Sao chép thư mục `registry/` sang `Day08_RAG_pipeline_cohort2/registry/`.
  - [x] Sao chép thư mục `common/` sang `Day08_RAG_pipeline_cohort2/common/`.
- [x] **Xây dựng RAG Retrieval Agent (`rag_retrieval_agent`)**
  - [x] Tạo folder `Day08_RAG_pipeline_cohort2/rag_retrieval_agent/`.
  - [x] Viết `rag_retrieval_agent/agent_executor.py` để bọc hàm `retrieve()` từ `src/task9_retrieval_pipeline.py` dưới dạng một Tool của agent.
  - [x] Viết `rag_retrieval_agent/__main__.py` thiết lập entrypoint (cổng 10101) đăng ký với Registry (cổng 10000). (Đã lược bỏ graph.py vì executor xử lý trực tiếp, tối ưu hóa độ trễ).
- [x] **Xây dựng Generation Agent (`generation_agent`)**
  - [x] Tạo folder `Day08_RAG_pipeline_cohort2/generation_agent/`.
  - [x] Viết `generation_agent/agent_executor.py` chứa Tool `call_retrieval_agent` để giao tiếp với `rag_retrieval_agent` thông qua client A2A.
  - [x] Viết `generation_agent/graph.py` thiết lập LangGraph: nhận câu hỏi -> gọi Tool tìm kiếm -> nhúng kết quả vào prompt và sinh câu trả lời có citation (dùng logic từ `src/task10_generation.py`).
  - [x] Viết `generation_agent/__main__.py` để chạy server trên cổng `10102` đăng ký với Registry.
- [x] **Tích hợp kiểm thử & Khởi động**
  - [x] Tạo file chạy `start_all.sh` (và `start_all.ps1`) cho Day 8 để khởi động Registry, Retrieval Agent, và Generation Agent.
  - [x] Chạy thử nghiệm và kiểm tra kết quả phản hồi cuối cùng có đầy đủ citation và hoạt động ổn định.
