# Start all RAG A2A services
$venvPython = "a:\AIK20_aithucchien\Batch02-Day9_Multi-Agent_MCP-A2A\.venv\Scripts\python.exe"

Write-Host "Starting Registry on port 10000..." -ForegroundColor Green
Start-Process -FilePath $venvPython -ArgumentList "-m registry" -NoNewWindow -RedirectStandardOutput "logs_registry.log" -RedirectStandardError "logs_registry_err.log"
Start-Sleep -Seconds 2

Write-Host "Starting RAG Retrieval Agent on port 10101..." -ForegroundColor Green
Start-Process -FilePath $venvPython -ArgumentList "-m rag_retrieval_agent" -NoNewWindow -RedirectStandardOutput "logs_retrieval.log" -RedirectStandardError "logs_retrieval_err.log"

Write-Host "Starting Generation Agent on port 10102..." -ForegroundColor Green
Start-Process -FilePath $venvPython -ArgumentList "-m generation_agent" -NoNewWindow -RedirectStandardOutput "logs_generation.log" -RedirectStandardError "logs_generation_err.log"
Start-Sleep -Seconds 3

Write-Host "All services started!" -ForegroundColor Yellow
Write-Host "  Registry:         http://localhost:10000"
Write-Host "  Retrieval Agent:  http://localhost:10101"
Write-Host "  Generation Agent: http://localhost:10102"
Write-Host ""
Write-Host "You can run 'test_a2a_rag.py' to verify the flow."
