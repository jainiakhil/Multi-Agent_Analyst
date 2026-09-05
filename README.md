# Multi-Agent Code & Document Analyst

[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/Framework-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/Vector%20Store-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/Local%20LLMs-Ollama-black.svg)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade, stateful multi-agent system designed for automated code audits and technical documentation analysis. Built with **LangGraph**, **Ollama**, **ChromaDB**, **SQLite**, and **FastAPI**, it orchestrates specialized local AI models to inspect code structural layouts via Abstract Syntax Trees (AST) and query indexed technical documents via vector RAG.

---

## 🏛️ Architecture & System Overview

The system uses a cyclic multi-agent supervisor architecture with persistent SQLite checkpointing and real-time streaming capabilities:

```
                             ┌────────────────────────┐
                             │   FastAPI Client / UI  │
                             └───────────┬────────────┘
                                         │ (WebSocket / REST)
                                         ▼
                             ┌────────────────────────┐
                             │   LangGraph Engine     │
                             │ (SQLite Checkpointer)  │
                             └───────────┬────────────┘
                                         │
                                         ▼
                             ┌────────────────────────┐
                             │    Supervisor Node     │
                             │ (Ollama: llama3.1:8b)  │
                             └───────────┬────────────┘
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
       ┌───────────────────────┐                   ┌───────────────────────┐
       │   Code Analyst Node   │                   │   Doc Parser Node     │
       │ (qwen2.5-coder:7b)    │                   │  (llama3.1:8b + RAG)  │
       └───────────┬───────────┘                   └───────────┬───────────┘
                   │                                           │
                   ▼                                           ▼
       ┌───────────────────────┐                   ┌───────────────────────┐
       │ AST Parser Tool (`ast`)│                  │ Local ChromaDB Vector │
       └───────────────────────┘                   └───────────────────────┘
```

### Agent Team & Responsibilities

1. **Lead Analyst Supervisor (`llama3.1:8b`)**:
   - Evaluates incoming prompts and ongoing conversation history.
   - Dynamically routes execution to the `code_analyst`, `doc_parser`, or determines completion (`FINISH`).
   - Uses structured outputs (`RouterOutput`) for deterministic control flow.

2. **Senior Code Auditor (`qwen2.5-coder:7b`)**:
   - Focuses on Python code quality, structural signatures, syntax errors, security vulnerabilities, and optimizations.
   - Actively invokes the native `parse_python_ast` tool to structurally inspect classes, methods, and functions before formulating audits.

3. **Technical Documentation Specialist (`llama3.1:8b`)**:
   - Resolves requirement questions, system specifications, and documentation queries.
   - Performs vector similarity search over local ChromaDB embeddings (`nomic-embed-text`) using the `query_documentation` tool.

---

## 📂 Project Structure

```
Multi-Agent Analyst/
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI entry point (REST + WebSockets)
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py               # Pydantic environment configuration
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py                # Graph state definitions & Pydantic schemas
│   │   ├── tools.py                # AST analysis & ChromaDB vector tools
│   │   ├── agents.py               # Supervisor and worker node definitions
│   │   └── workflow.py             # LangGraph construction & SQLite persistence
│   └── services/
│       ├── __init__.py
│       └── vector_store.py         # Ingestion pipelines for PDFs & text files
├── data/
│   ├── chroma/                     # Persistent local ChromaDB store
│   └── checkpoints.db              # SQLite database for state snapshots
├── tests/
│   ├── __init__.py
│   ├── sample_code.py              # Sample code for AST parser verification
│   ├── test_ast.py                 # AST parsing unit tests
│   ├── test_vector_store.py        # Vector ingestion and chunking tests
│   ├── test_graph.py               # StateGraph routing and schema tests
│   └── test_api.py                 # FastAPI REST and WebSocket test suite
├── .env.example                    # Environment variable template
├── .gitignore                      # Git ignore specifications
├── Dockerfile                      # Container build definition
├── docker-compose.yml              # Multi-container orchestration
├── requirements.txt                # Python package dependencies
└── README.md                       # Documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites

- **Python 3.11+** installed.
- **[Ollama](https://ollama.com/)** running locally or accessible via network.
- **Git**.

### 2. Pull Required Ollama Models

Pull the three models required for orchestration, code audit, and embeddings:

```bash
# Pull Supervisor & Documentation LLM
ollama pull llama3.1:8b

# Pull Code Analyst LLM
ollama pull qwen2.5-coder:7b

# Pull Local Embeddings Model
ollama pull nomic-embed-text
```

### 3. Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/jainiakhil/Multi-Agent_Analyst.git
cd Multi-Agent_Analyst

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configuration

Copy the example environment file and customize if needed:

```bash
cp .env.example .env
```

Default settings in `.env`:
```ini
OLLAMA_BASE_URL=http://localhost:11434
SUPERVISOR_MODEL=llama3.1:8b
CODE_ANALYST_MODEL=qwen2.5-coder:7b
DOC_PARSER_MODEL=llama3.1:8b
EMBEDDING_MODEL=nomic-embed-text
CHROMA_PERSIST_DIR=./data/chroma
CHECKPOINT_DB_PATH=./data/checkpoints.db
HOST=0.0.0.0
PORT=8000
DEBUG=False
```

### 5. Launch the API Service

Start the FastAPI application using Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI Swagger documentation will be available at:
`http://localhost:8000/docs`

---

## 🐳 Docker Deployment

To run the application inside Docker with volume persistence for ChromaDB and SQLite checkpoints:

```bash
# Build and start the service
docker compose up --build -d

# View container logs
docker compose logs -f analyst-api

# Stop service
docker compose down
```

*Note: In `docker-compose.yml`, `host.docker.internal` is pre-configured so the container can communicate with Ollama running on your host machine.*

---

## 📡 API Usage & Examples

### Health Check

```bash
curl -X GET "http://localhost:8000/health"
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "ollama_base_url": "http://localhost:11434",
  "provider": "ollama",
  "supervisor_model": "llama3.1:8b",
  "code_analyst_model": "qwen2.5-coder:7b",
  "doc_parser_model": "llama3.1:8b",
  "embedding_model": "nomic-embed-text"
}
```

---

## 🔄 Switching LLM Models & Providers

The system provides complete flexibility to switch models at three different levels without modifying source code:

### 1. Environment / Configuration (`.env`)
Set the default models and provider in your `.env` file:
```ini
LLM_PROVIDER=ollama
SUPERVISOR_MODEL=llama3.1:8b
CODE_ANALYST_MODEL=qwen2.5-coder:7b
DOC_PARSER_MODEL=llama3.1:8b
EMBEDDING_MODEL=nomic-embed-text
```

### 2. Runtime Dynamic Model Switching (REST API)
Switch models and providers on the running server on the fly without restarting:

- **List active models & discover local Ollama models**:
  ```bash
  curl -X GET "http://localhost:8000/api/v1/models"
  ```

- **Switch active models at runtime**:
  ```bash
  curl -X POST "http://localhost:8000/api/v1/models/switch" \
       -H "Content-Type: application/json" \
       -d '{
         "supervisor_model": "mistral:7b",
         "code_analyst_model": "qwen2.5-coder:14b"
       }'
  ```

### 3. Per-Request Model Overrides
Override models on demand in individual requests (`POST /api/v1/analyze` or `WebSocket /ws/analyze`):
```json
{
  "query": "Review tests/sample_code.py for syntax and structure.",
  "thread_id": "custom_session",
  "target_file": "tests/sample_code.py",
  "supervisor_model": "llama3.2:3b",
  "code_analyst_model": "qwen2.5-coder:14b"
}
```

### 4. Using OpenAI-Compatible Providers (vLLM, LMStudio, OpenRouter, Groq, OpenAI)
To use any OpenAI-compatible inference server, configure `.env`:
```ini
LLM_PROVIDER=openai_compatible
OPENAI_API_BASE=http://localhost:1234/v1   # LMStudio, vLLM, or provider URL
OPENAI_API_KEY=your-api-key-if-needed
SUPERVISOR_MODEL=gpt-4o
CODE_ANALYST_MODEL=gpt-4o-mini
```

### 1. Ingest Documentation (PDF / TXT / MD)

Upload a specification, technical manual, or document into the ChromaDB vector database:

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample_spec.pdf"
```

Response:
```json
{
  "status": "success",
  "chunks_indexed": 14,
  "filename": "sample_spec.pdf"
}
```

### 2. Synchronous Code & Document Analysis (REST)

Execute an analysis query over HTTP:

```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Audit app/graph/workflow.py and verify if it matches our indexing spec.",
       "thread_id": "session_001",
       "target_file": "app/graph/workflow.py"
     }'
```

### 3. Real-Time Streaming (WebSocket)

Connect to the WebSocket endpoint for real-time agent transition events:

Using `wscat`:
```bash
npm install -g wscat
wscat -c ws://localhost:8000/ws/analyze
```

Send the analysis request:
```json
{
  "query": "Analyze app/graph/workflow.py for code quality and verify against indexed spec.",
  "thread_id": "session_001",
  "target_file": "app/graph/workflow.py"
}
```

Streaming updates will emit:
```json
{"node": "supervisor", "next_node": "code_analyst", "latest_message": null}
{"node": "code_analyst", "next_node": "", "latest_message": "[Code Analyst]:\nAST Analysis shows..."}
{"node": "supervisor", "next_node": "FINISH", "latest_message": null}
{"status": "completed"}
```

Using Python client:
```python
import asyncio
import json
import websockets

async def stream_analysis():
    uri = "ws://localhost:8000/ws/analyze"
    async with websockets.connect(uri) as ws:
        payload = {
            "query": "Review tests/sample_code.py for syntax and structure.",
            "thread_id": "test_session_1",
            "target_file": "tests/sample_code.py"
        }
        await ws.send(json.dumps(payload))
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            print(f"[{data.get('node', 'Event')}]: {data.get('latest_message') or data.get('status')}")
            if data.get("status") == "completed":
                break

asyncio.run(stream_analysis())
```

---

## 🧪 Testing Suite

All tests are located in `tests/` and run against `pytest`:

```bash
# Run all tests
pytest -v tests/

# Run individual test suites
pytest -v tests/test_ast.py
pytest -v tests/test_graph.py
pytest -v tests/test_vector_store.py
pytest -v tests/test_api.py
```

---

## 🛠️ Contributing

Contributions are welcome! Please follow these guidelines:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/awesome-feature`).
3. Ensure all tests pass (`pytest -v tests/`).
4. Commit your changes with clear messages (`git commit -m "Add awesome feature"`).
5. Push to your branch (`git push origin feature/awesome-feature`).
6. Open a Pull Request.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
