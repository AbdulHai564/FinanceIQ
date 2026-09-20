# FinanceIQ 📈
> Chat with financial documents using production-grade RAG

🔗 **[Live Demo](https://financeiq-rag.streamlit.app)**

---

## What it does
FinanceIQ lets you ask natural language questions about Nvidia's 2023 Annual Report and get accurate, cited answers in seconds.

**Example:**
> "What was Nvidia's total revenue in 2023?"
> → *"NVIDIA's fiscal 2023 revenue was approximately $27 billion..."* + sources shown

---

## Why it's different from basic RAG
Most RAG tutorials use simple semantic search. This project uses a production-grade pipeline:

| Component | Basic RAG | FinanceIQ |
|-----------|-----------|-----------|
| Search | Semantic only | Hybrid (BM25 + Semantic) |
| Relevance | Raw retrieval | Cohere Reranking |
| Speed | Reloads every time | Cached models |
| API | None | FastAPI wrapper |

---

## Tech Stack
- **Vector DB** — Qdrant Cloud
- **Embeddings** — HuggingFace `all-MiniLM-L6-v2`
- **Search** — Hybrid (BM25 + Semantic via LangChain EnsembleRetriever)
- **Reranking** — Cohere `rerank-english-v3.0`
- **LLM** — Groq (fast inference)
- **UI** — Streamlit
- **API** — FastAPI

---

## Architecture
```
User Question
     ↓
Hybrid Search (BM25 + Semantic)
     ↓
Cohere Reranker (top 5 chunks)
     ↓
Groq LLM → Answer + Sources
```

---

## Features
- Hybrid retrieval for accuracy + coverage
- Cohere reranking filters low quality chunks
- Source citations show exactly which chunks were used
- Chat history persists across the session
- Models cached on startup for fast responses
- FastAPI wrapper for programmatic access

---

## Run Locally
```bash
git clone https://github.com/AbdulHai564/FinanceIQ.git
cd FinanceIQ
pip install -r requirements.txt
```

Add a `.env` file:
```
QDRANT_URL=your-url
QDRANT_API_KEY=your-key
GROQ_API_KEY=your-key
COHERE_API_KEY=your-key
```

```bash
streamlit run appp.py
```
