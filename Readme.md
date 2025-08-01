# 🧠 Retrieval-Augmented Generation (RAG) Document QA System

This project is a **Retrieval-Augmented Generation (RAG) pipeline** that allows users to **upload documents** (PDF, TXT, etc.) and **ask questions** based on their content. It combines **semantic retrieval using vector databases** with **LLM-powered generation** to provide accurate, context-aware responses.

The system is fully **containerized with Docker** and can be deployed in both **local and cloud environments**.

---

## 🚀 Features

- Upload and process up to 20 documents (max 1000 pages each)
- Intelligent document chunking and embedding
- Fast semantic search with vector database (FAISS / Qdrant)
- RAG pipeline with LLMs like OpenAI GPT or Gemini
- REST API endpoints for upload, query, and metadata
- Dockerized setup for easy local/cloud deployment

---

## 🧰 Tech Stack

- **Backend:** Python, FastAPI, LangChain
- **LLM APIs:** OpenAI / Gemini / Local (Mistral, Ollama - optional)
- **Embeddings:** SentenceTransformers / OpenAI Embeddings
- **Vector DB:** FAISS (local), Qdrant or Weaviate (optional cloud)
- **Document Parsing:** PyMuPDF, pdfminer.six, Unstructured.io
- **Containerization:** Docker, Docker Compose

---

## 🛠️ Step-by-Step Setup (Before Docker)

### 1. **Project Initialization**
- Set up a Python environment
- Initialize FastAPI app with endpoints:
  - `/upload`: for document uploads
  - `/query`: for user questions
  - `/metadata`: to list uploaded files

### 2. **Document Handling**
- Parse documents (PDFs, TXT)
- Chunk documents using recursive splitting or token-based methods
- Store raw chunks with metadata (title, page number, etc.)

### 3. **Embeddings & Vector Store**
- Generate embeddings using:
  - `sentence-transformers/all-MiniLM-L6-v2` (local)
  - or OpenAI embedding model (`text-embedding-3-small`)
- Store embeddings in:
  - **FAISS** (for local)
  - or **Qdrant** (optional cloud)

### 4. **LLM Integration**
- Accept a user query
- Retrieve top relevant chunks from vector DB
- Send both query and retrieved context to an LLM:
  - OpenAI (via API key)
  - Gemini (via REST API)
- Return generated answer

### 5. **API Testing**
- Test all endpoints using **Postman** or **curl**
- Validate PDF parsing, retrieval accuracy, and LLM output

---

## 🐳 Dockerization & Deployment

### 6. **Dockerfile Setup**
- Create `Dockerfile` with FastAPI + LangChain dependencies
- Include any models or scripts required for local embedding

### 7. **Docker Compose**
- Add `docker-compose.yml` to orchestrate:
  - FastAPI app
  - FAISS or Qdrant container (if not in-memory)
  - Mount volumes for storing uploaded documents

### 8. **Local Deployment**
```bash
docker-compose up --build
```

```
# 🚀 Setup and Installation

# 1. Clone the Repository

# 2. Create and Activate a Virtual Environment
# For Linux/macOS:
python3 -m venv venv
source venv/bin/activate

# For Windows:
python -m venv venv
venv\Scripts\activate

# 3. Install Python Dependencies
pip install -r requirements.txt

# 4. Set Environment Variables
# Create a .env file in the root directory with the following:

LLM_PROVIDER=openai           # or gemini or local
OPENAI_API_KEY=your-openai-key
GEMINI_API_KEY=your-gemini-key
LOCAL_LLM_URL=http://localhost:11434/api/generate

# 5. Run the Application Locally
uvicorn app.main:app --reload

# Access the API at:
# http://localhost:8000/docs

# 🐳 Docker Installation (Optional)

# 1. Build and Start the Services
docker-compose up --build

# 2. Access the API
# Swagger UI: http://localhost:8000/docs
```



