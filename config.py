import os
from dotenv import load_dotenv
import streamlit as st


load_dotenv()

def get_secret(key):
    try:
        return st.secrets[key]
    except:
        return os.getenv(key)


QDRANT_URL=os.getenv("QDRANT_URL")
QDRANT_API_KEY=os.getenv("QDRANT_API_KEY")
PDF_PATH="reports/Nvidia 2023 Annual Report.pdf"
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
COLLECTION_NAME="nvidia_10k"
COHERE_API_KEY=os.getenv("COHERE_API_KEY")
