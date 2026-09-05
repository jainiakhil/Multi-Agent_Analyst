# PowerShell runner script for Multi-Agent Analyst
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "   Starting Multi-Agent Code & Document Analyst   " -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Cyan

# Check if .venv exists
if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "Virtual environment not found! Run: python -m venv .venv" -ForegroundColor Red
    exit 1
}

Write-Host "Launching FastAPI server on http://localhost:8000..." -ForegroundColor Yellow
Write-Host "Interactive Swagger docs at: http://localhost:8000/docs" -ForegroundColor Gray
Write-Host ""

& ".\.venv\Scripts\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
