from config import QDRANT_URL, QDRANT_API_KEY, PDF_PATH, COLLECTION_NAME, GROQ_API_KEY, COHERE_API_KEY
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_cohere import CohereRerank
from langchain_groq import ChatGroq

def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = QdrantVectorStore.from_existing_collection(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    return retriever

def chunking(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    return chunks

def BM25_retrieval(chunks):
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 10
    return bm25

def hybrid_search(chunks, question, semantic, bm25):
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25, semantic],
        weights=[0.5, 0.5]
    )

    return hybrid_retriever.invoke(question)


def cohere_reranker(hybrid_results, question):
    reranker = CohereRerank(
        model="rerank-english-v3.0",
        top_n=5,
        cohere_api_key=COHERE_API_KEY
    )
    reranked = reranker.compress_documents(hybrid_results, question)
    return reranked

def generate_answer(reranked, question):
    llm = ChatGroq(model="openai/gpt-oss-120b", api_key=GROQ_API_KEY)
    context = "\n\n".join([doc if isinstance(doc, str) else doc.page_content for doc in reranked])
    prompt = f"""Answer the following question based on the context below:

Context:
{context}

Question: {question}"""
    
    answer = llm.invoke(prompt)
    return answer.content

if __name__ == "__main__":
    question = "What was Nvidia's total revenue in 2023?"

    chunks = chunking(PDF_PATH)

    hybrid_results = hybrid_search(chunks, question)

    reranked = cohere_reranker(hybrid_results, question)

    print(generate_answer(reranked, question))

