@echo off
title Multi-Agent Code and Document Analyst
echo ==================================================
echo    Starting Multi-Agent Code and Document Analyst
echo ==================================================

if not exist ".\.venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found in .\.venv!
    pause
    exit /b 1
)

echo Starting FastAPI server on http://localhost:8000 ...
echo Swagger documentation: http://localhost:8000/docs
echo.
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
pause
