

```markdown
# Hybrid PDF Search Engine (BM25 + Semantic RAG)

A production-style **Hybrid Search system** that combines:
- **Keyword-based retrieval (BM25)**
- **Semantic vector search (Embeddings)**
- **Retrieval-Augmented Generation (RAG)**

Built with **LangChain + ChromaDB + Streamlit**, designed to run on **CPU-only systems**.

---

## 🚀 Features
- Upload any PDF document
- Ask natural language questions
- Hybrid retrieval:
  - Lexical keyword search (BM25)
  - Semantic similarity search (vector embeddings)
- Context-aware answers grounded in document content
- Lightweight and local-first (no GPU required)

---

## 🧠 Architecture Overview

The system uses **two parallel retrievers** to improve recall and precision:

1. **BM25 Retriever**
   - Captures exact keyword and phrase matches
2. **Vector Retriever**
   - Captures semantic similarity using embeddings

Both results are **merged** and passed to a **Retrieval-Augmented Generation (RAG)** pipeline for answer generation.

---

## 🏗️ Project Structure

```

genai-hybrid-search/
│
├── app/
│   ├── ingest.py           # PDF loading & text chunking
│   ├── bm25_store.py       # BM25 keyword retriever
│   ├── vectorstore.py      # Chroma vector database
│   ├── hybrid_search.py    # Hybrid retrieval logic
│   ├── rag.py              # RAG pipeline
│
├── data/
│   └── sample.pdf          # Example document
│
├── streamlit_app.py        # Streamlit UI entry point
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
└── venv/                   # Virtual environment

````

---

## 🔄 End-to-End Pipeline

1. User uploads a PDF
2. PDF text is chunked
3. Parallel retrieval:
   - BM25 keyword search
   - Vector similarity search
4. Results are merged
5. LLM generates a grounded answer
6. Answer is displayed in Streamlit UI

---

## 🧪 Run Locally

### 1. Create virtual environment
```bash
python -m venv venv
````

### 2. Activate environment

```bash
venv/Scripts/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch app

```bash
streamlit run streamlit_app.py
```

---

## 🛠 Tech Stack

* Python
* LangChain
* ChromaDB
* Sentence-Transformers
* Streamlit

---

## 📌 Use Cases

* PDF Question Answering
* Internal document search
* Resume-ready GenAI project
* Hybrid Retrieval experimentation
* RAG system prototyping

---

## 🔮 Future Improvements

* Cross-encoder reranking
* Metadata-based filtering
* Multi-document support
* FastAPI backend
* Cloud deployment

---

## 📄 License

This project is for educational and demonstration purposes.

````

---

### ✅ What to do next
1. Save this as `README.md`
2. Run:
```bash
git add README.md
git commit -m "docs: add project README"
git push
````

When ready, say **“next”** and we’ll proceed to **deployment or the next advanced project**.
