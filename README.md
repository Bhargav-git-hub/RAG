# RAG Application with Ollama

A lightweight Retrieval-Augmented Generation (RAG) application built from scratch using Python, NumPy, and Ollama. The application retrieves relevant information from local Markdown documents using vector embeddings and generates context-aware answers using a locally running LLM.

## Overview

This project demonstrates the fundamentals of building a RAG pipeline without relying on frameworks such as LangChain.

It uses:
- **Ollama** – Runs local embedding and language models.
- **nomic-embed-text** – Converts documents and user queries into vector embeddings.
- **NumPy** – Performs vector similarity calculations and retrieves relevant document chunks.
- **Qwen 3 (4B)** – Generates answers using the retrieved context.
- **uv** – Python dependency and project management.

### How It Works

The application follows a simple RAG pipeline:

1. **Document Loading:** Reads Markdown (`.md`) files from the `data/` directory.
2. **Chunking:** Splits documents into smaller chunks using paragraph breaks.
3. **Embedding Generation:** Converts each chunk into a numerical vector using `nomic-embed-text`.
4. **Similarity Search:** Converts the user's query into an embedding and calculates Euclidean distances against stored document embeddings.
5. **Context Retrieval:** Retrieves the top 3 most relevant document chunks.
6. **Answer Generation:** Passes the retrieved context and user query to Qwen 3 to generate an answer.

```text
             Markdown Documents
                     |
                     v
              Document Loading
                     |
                     v
                  Chunking
                     |
                     v
             Embedding Generation
              (nomic-embed-text)
                     |
                     v
              Document Vectors
                     |
                     |
User Query ----------+
     |
     v
Query Embedding
     |
     v
Similarity Search (NumPy)
     |
     v
Top 3 Relevant Chunks
     |
     v
  Context + Query
     |
     v
   Qwen 3 (4B)
     |
     v
  Generated Answer
```

## Project Structure

```text
RAG/
│
├── data/
│   └── handbook.md          # Source documents
│
├── src/
│   └── rag/
│       └── __init__.py
│
├── rag.py                   # Main RAG application
├── pyproject.toml           # Project configuration and dependencies
├── uv.lock                  # Dependency lockfile
├── .python-version          # Python version configuration
├── .gitignore
└── README.md
```

## Prerequisites

Before running the application, make sure you have the following installed:

| Requirement | Purpose |
|---|---|
| Python 3.14 | Programming language |
| uv | Python package and environment management |
| Ollama | Local LLM and embedding model runtime |
| Git | Repository management |

### 1. Install uv

If you don't already have uv installed:

**macOS / Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify installation:

```bash
uv --version
```

Installation instructions for other platforms: https://docs.astral.sh/uv/

### 2. Install Ollama

Download and install Ollama:

https://ollama.com/download

Verify installation:

```bash
ollama --version
```

## Installation & Setup

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Navigate into the project directory:

```bash
cd RAG
```

### 2. Install Python Dependencies

Use uv to create the virtual environment and install the dependencies defined in `pyproject.toml` and `uv.lock`.

```bash
uv sync
```

This installs the required packages and creates the `.venv` environment automatically.

If `requests` and `numpy` are not already listed in your project dependencies, add them:

```bash
uv add requests numpy
```

### 3. Download Ollama Models

The application requires two models.

**Embedding Model:**

```bash
ollama pull nomic-embed-text:latest
```

**Language Model:**

```bash
ollama pull qwen3:4b-instruct
```

Verify that both models are available:

```bash
ollama list
```

### 4. Prepare Your Documents

Place your source Markdown documents inside the `data/` directory.

Example:

```text
data/
├── handbook.md
├── company_info.md
└── policies.md
```

The application currently reads `.md` files and splits their content into chunks using blank lines.

You can add your own Markdown documents to experiment with retrieval and question answering.

## Running the Application

Make sure Ollama is running. On macOS, the Ollama application normally starts its local service automatically.

If necessary, start the service:

```bash
ollama serve
```

From the project root, execute:

```bash
uv run rag.py
```

The application will load your documents, generate embeddings, and start an interactive question-answering session.

### Example Interaction

```text
Loaded 11 chunks

You: When does Byte Bistro open?

The cafe opens at 7 AM from Monday to Friday
and 8 AM on weekends.

You: What is the signature drink?

The signature drink is the Stack Overflow Latte,
which consists of a double shot of espresso,
oat milk, maple syrup, and a pinch of sea salt.

You: exit
```

*Example responses are illustrative; actual answers depend on the documents and retrieved context.*

## Configuration

The application uses the following configuration:

| Configuration | Value |
|---|---|
| Ollama API | `http://localhost:11434` |
| Embedding model | `nomic-embed-text:latest` |
| Chat model | `qwen3:4b-instruct` |
| Embedding dimensions | 768 |
| Number of retrieved chunks (Top K) | 3 |
| Similarity metric | Euclidean distance |

You can modify the `k` parameter in `search_chunks()` to retrieve more or fewer document chunks.

## Troubleshooting

### Ollama connection refused

Make sure Ollama is running:

```bash
ollama serve
```

### Model not found

Pull the required models:

```bash
ollama pull nomic-embed-text:latest
ollama pull qwen3:4b-instruct
```

### No document chunks loaded

Verify that:
- The `data/` directory exists.
- Your Markdown files are inside `data/`.
- Your files have the `.md` extension.
- You are running the application from the project root.

### Missing Python dependencies

Run:

```bash
uv sync
```

If necessary:

```bash
uv add requests numpy
```

## Learning Objectives

This project provides hands-on experience with:

- Understanding Retrieval-Augmented Generation (RAG).
- Document ingestion and text chunking.
- Generating vector embeddings using local embedding models.
- Implementing semantic retrieval using NumPy.
- Understanding vector distance calculations.
- Connecting embedding models with generative LLMs.
- Building a basic local question-answering system.

## Future Improvements

Potential enhancements include:

- [ ] Support for PDF and TXT document ingestion.
- [ ] Improved text chunking with configurable chunk sizes and overlap.
- [ ] Cosine similarity-based retrieval.
- [ ] Persistent vector storage to avoid regenerating embeddings on every application startup.
- [ ] Integration with vector databases such as ChromaDB or FAISS.
- [ ] Streaming LLM responses.
- [ ] Source attribution and citations in generated answers.
- [ ] Improved prompt engineering and retrieval evaluation.

## License

This project is open-source. Add a license file (such as MIT) if you intend to distribute it under that license.