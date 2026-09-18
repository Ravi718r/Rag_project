# Production-Style RAG Question Answering System

A production-oriented **Retrieval-Augmented Generation (RAG)** system built with **FastAPI, LangChain, ChromaDB, Hugging Face embeddings, BM25 hybrid retrieval, Cross-Encoder reranking, and Ollama**.

The system retrieves relevant information from a document knowledge base before generating an answer, reducing hallucinations and keeping responses grounded in the available documents.

---

## 🏗️ Architecture

```text
                    ┌──────────────────┐
                    │      User        │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    FastAPI API   │
                    │      /ask        │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Query Router    │
                    │  Metadata Filter │
                    └────────┬─────────┘
                             │
                 ┌───────────┴───────────┐
                 ▼                       ▼
        ┌─────────────────┐     ┌─────────────────┐
        │ Vector Search   │     │   BM25 Search   │
        │    ChromaDB     │     │ Keyword Search  │
        └────────┬────────┘     └────────┬────────┘
                 └───────────┬───────────┘
                             ▼
                    ┌──────────────────┐
                    │ Hybrid Retrieval │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │ Cross-Encoder    │
                    │    Reranking     │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │ Context Builder  │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │   Ollama LLM     │
                    │   Qwen 2.5 3B    │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │      Answer      │
                    └──────────────────┘
```

---

## ✨ Features

- Document ingestion and chunking
- Hugging Face sentence embeddings
- ChromaDB vector storage
- BM25 keyword retrieval
- Hybrid retrieval
- Query routing and metadata filtering
- Cross-Encoder reranking
- Similarity and reranking thresholds
- Grounded LLM generation
- FastAPI REST API
- Request validation and error handling
- Structured application logging
- RAG evaluation pipeline
- Docker containerization
- Ollama integration for local LLM inference

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| API | FastAPI |
| LLM | Ollama + Qwen 2.5 3B |
| Embeddings | `all-MiniLM-L6-v2` |
| Vector DB | ChromaDB |
| Keyword Retrieval | BM25 |
| Reranking | BAAI BGE Cross-Encoder |
| Framework | LangChain |
| Containerization | Docker + Docker Compose |

---

## 📁 Project Structure

```text
RAG/
│
├── app/
│   ├── main.py
│   ├── schemas.py
│   └── logging_config.py
│
├── docs/
│   └── knowledge documents
│
├── evaluation/
│   ├── evaluator.py
│   ├── ragas_evaluator.py
│   ├── experiment.py
│   ├── regression.py
│   └── reporter.py
│
├── retrievers/
│   ├── hybrid.py
│   ├── reranker.py
│   ├── metadata.py
│   └── compression.py
│
├── routers/
│   └── rule_router.py
│
├── utils/
│   └── formatter.py
│
├── ingestion_pipeline.py
├── rag_pipeline.py
├── rag_setup.py
├── embeddings.py
├── config.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## ⚙️ Configuration

Create a `.env` file from the provided example:

```bash
cp .env.example .env
```

Configure the required environment variables in `.env`.

Example:

```env
OLLAMA_BASE_URL=http://localhost:11434
MODEL_NAME=qwen2.5:3b
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

PERSIST_DIRECTORY=db/chroma_db

TOP_K=3
RETRIEVAL_K=10
FINAL_K=3

SIMILARITY_THRESHOLD=0.45
RERANK_THRESHOLD=0.20
```

---

## 🚀 Running Locally

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd RAG
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Ollama

Make sure Ollama is installed and running.

Pull the required model:

```bash
ollama pull qwen2.5:3b
```

### 5. Start FastAPI

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

---

## 🐳 Running with Docker

Build the Docker image:

```bash
docker compose build
```

Start the application:

```bash
docker compose up
```

Run in detached mode:

```bash
docker compose up -d
```

Stop the application:

```bash
docker compose down
```

The API will be available at:

```text
http://localhost:8000
```

Swagger:

```text
http://localhost:8000/docs
```

---

## 🔌 Ollama + Docker

The RAG API runs inside the Docker container while Ollama runs on the host machine.

Docker Compose connects the container to the host Ollama service using:

```yaml
environment:
  OLLAMA_BASE_URL: http://host.docker.internal:11434

extra_hosts:
  - "host.docker.internal:host-gateway"
```

This allows the containerized FastAPI application to communicate with Ollama running on the host.

---

## 🔄 RAG Pipeline

```text
Documents
    │
    ▼
Document Loading
    │
    ▼
Chunking
    │
    ▼
Hugging Face Embeddings
    │
    ▼
ChromaDB
    │
    ▼
Query
    │
    ▼
Query Routing
    │
    ▼
Metadata Filtering
    │
    ├───────────────┐
    ▼               ▼
Vector Search     BM25 Search
    │               │
    └───────┬───────┘
            ▼
      Hybrid Retrieval
            │
            ▼
     Cross-Encoder
       Reranking
            │
            ▼
      Context Builder
            │
            ▼
       Ollama LLM
            │
            ▼
          Answer
```

---

## 📚 Retrieval Strategy

The system combines multiple retrieval techniques:

### Vector Retrieval

Uses semantic embeddings with:

```text
sentence-transformers/all-MiniLM-L6-v2
```

and stores the embeddings in ChromaDB.

### BM25 Retrieval

Provides keyword-based retrieval and helps handle queries where exact terms are important.

### Hybrid Retrieval

Combines semantic and keyword retrieval to improve recall across different query types.

### Query Routing

Routes queries to relevant document sources before retrieval.

### Metadata Filtering

Uses document metadata such as `source` to restrict retrieval to relevant documents.

### Cross-Encoder Reranking

Retrieved candidates are reranked using:

```text
BAAI/bge-reranker-base
```

A reranking threshold is applied before selecting the final context.

---

## 🧪 Evaluation

The project includes an evaluation pipeline for measuring RAG quality.

Evaluation components include:

- Retrieval evaluation
- RAGAS evaluation
- Experiment configuration
- Regression testing
- Evaluation reporting

The evaluation pipeline is designed to help detect retrieval and generation quality regressions when the RAG system changes.

---

## 🔒 Security

Sensitive configuration should not be committed to Git.

Use:

```text
.env
```

for local secrets and configuration.

Commit only:

```text
.env.example
```

Make sure `.env` is included in `.gitignore`.

---

## 📌 API

The primary API endpoint is:

```text
POST /ask
```

Example request:

```json
{
  "question": "What is the primary focus of Google LLC?"
}
```

Example response:

```json
{
  "question": "What is the primary focus of Google LLC?",
  "answer": "..."
}
```

---

## 🎯 Project Goals

This project demonstrates practical implementation of:

- Retrieval-Augmented Generation
- Retrieval engineering
- Hybrid search
- Metadata-aware retrieval
- Cross-Encoder reranking
- Local LLM inference
- FastAPI API development
- RAG evaluation
- Docker containerization
- Production-oriented project structure

---

## 👨‍💻 Author

**Ravi Yadav**

B.Tech — Electronics & Communication Engineering

Focused on **AI Engineering, RAG Systems, LLM Applications, and Agentic AI**.
