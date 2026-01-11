import streamlit as st

from app.ingest import load_and_chunk_pdf
from app.bm25_store import BM25Retriever
from app.vectorstore import create_vectorstore
from app.hybrid_search import HybridRetriever
from app.rag import HybridRAG


st.set_page_config(page_title="Hybrid PDF Search", layout="centered")

st.title("📄 Hybrid PDF Search (BM25 + Semantic)")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully")

    with st.spinner("Processing document..."):
        docs = load_and_chunk_pdf("temp.pdf")
        bm25 = BM25Retriever(docs)
        vs = create_vectorstore(docs)
        hybrid = HybridRetriever(bm25, vs)
        rag = HybridRAG(hybrid)

    query = st.text_input("Ask a question about the document")

    if query:
        with st.spinner("Generating answer..."):
            answer = rag.answer(query)

        st.subheader("Answer")
        st.write(answer)
