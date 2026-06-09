#!/bin/bash
set -e

# Start all RAG A2A services
PYTHON_EXEC="a:/AIK20_aithucchien/Batch02-Day9_Multi-Agent_MCP-A2A/.venv/Scripts/python"

echo "Starting Registry on port 10000..."
$PYTHON_EXEC -m registry > logs_registry.log 2> logs_registry_err.log &
REGISTRY_PID=$!
sleep 2

echo "Starting RAG Retrieval Agent on port 10101..."
$PYTHON_EXEC -m rag_retrieval_agent > logs_retrieval.log 2> logs_retrieval_err.log &
RETRIEVAL_PID=$!

echo "Starting Generation Agent on port 10102..."
$PYTHON_EXEC -m generation_agent > logs_generation.log 2> logs_generation_err.log &
GENERATION_PID=$!
sleep 3

echo ""
echo "All services started:"
echo "  Registry:         http://localhost:10000"
echo "  Retrieval Agent:  http://localhost:10101"
echo "  Generation Agent: http://localhost:10102"
echo ""
echo "You can run 'test_a2a_rag.py' to verify the flow."
echo "Press Ctrl+C to exit."

wait $REGISTRY_PID $RETRIEVAL_PID $GENERATION_PID
