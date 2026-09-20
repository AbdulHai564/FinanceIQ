
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from config import QDRANT_URL,QDRANT_API_KEY,PDF_PATH,COLLECTION_NAME




def load_chunk(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks: {len(chunks)}")
    return chunks

def store_in_qdrant(chunks):

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    
    
    
    vectorstore = QdrantVectorStore.from_documents(
        chunks,
        embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME
    )
    
    print("Stored in Qdrant!")
    return vectorstore

def main():
    print("Loading and chunking PDF...")
    chunks = load_chunk(PDF_PATH)
    
    print("Storing in Qdrant...")
    store_in_qdrant(chunks)
    
    print("Done! PDF ingested successfully.")

if __name__ == "__main__":
    main()
    