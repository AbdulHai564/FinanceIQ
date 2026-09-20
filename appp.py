import streamlit as st
from retriever import (
    get_retriever,
    chunking,
    BM25_retrieval,
    hybrid_search,
    cohere_reranker,
    generate_answer
)
from config import PDF_PATH

st.title("📈 FinanceIQ")
st.caption("Chat with Nvidia's 2023 Annual Report")

@st.cache_resource
def load_resources():
    chunks = chunking(PDF_PATH)
    semantic = get_retriever()
    bm25 = BM25_retrieval(chunks)
    return chunks, semantic, bm25

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

chunks, semantic, bm25 = load_resources()

if question := st.chat_input("Ask about Nvidia's financials..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching and analyzing..."):
            hybrid_results = hybrid_search(chunks, question, semantic, bm25)
            reranked = cohere_reranker(hybrid_results, question)
            answer = generate_answer(reranked, question)
            st.markdown(answer)

            with st.expander("📄 Sources used"):
                for i, chunk in enumerate(reranked):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.caption(chunk.page_content)

    st.session_state.messages.append({"role": "assistant", "content": answer})