from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from retriever import (
    get_retriever,
    chunking,
    BM25_retrieval,
    hybrid_search,
    cohere_reranker,
    generate_answer
)
from config import PDF_PATH

app = FastAPI(title="Nvidia RAG API")

# Load resources once on startup
chunks = chunking(PDF_PATH)
semantic = get_retriever()
bm25 = BM25_retrieval(chunks)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        hybrid_results = hybrid_search(chunks, request.question, semantic, bm25)
        reranked = cohere_reranker(hybrid_results, request.question)
        answer = generate_answer(reranked, request.question)
        sources = [doc.page_content for doc in reranked]

        return QueryResponse(answer=answer, sources=sources)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


    