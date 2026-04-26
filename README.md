# ✦ RAG PDF Q&A

A Retrieval-Augmented Generation (RAG) system that lets you upload any PDF and ask natural language questions about it. Built end-to-end with a custom pipeline and deployed as an interactive web app.

**🔗 Live Demo:** [huggingface.co/spaces/saadhnasoman/rag-pdf-qa](https://huggingface.co/spaces/saadhnasoman/rag-pdf-qa)

---

## How It Works

1. Upload any PDF document
2. The document is chunked, embedded using `BAAI/bge-small-en-v1.5`, and indexed with FAISS
3. You ask a question — the system retrieves the 3 most semantically relevant chunks
4. Groq's LLaMA 3 generates an answer strictly from those chunks

```
Question
   ↓
BGE embeds the question (cosine similarity via IndexFlatIP)
   ↓
FAISS retrieves top-3 relevant chunks
   ↓
Chunks injected into prompt as context
   ↓
Groq LLaMA 3.3 70B answers from context only
   ↓
Answer displayed in chat UI
```

---

## Tech Stack

| Component | Tool |
|---|---|
| PDF parsing | `pdfplumber` |
| Text chunking | `LangChain RecursiveCharacterTextSplitter` |
| Embeddings | `BAAI/bge-small-en-v1.5` (sentence-transformers) |
| Vector store | `FAISS` (IndexFlatIP) |
| LLM inference | `Groq API` — LLaMA 3.3 70B Versatile |
| UI | `Streamlit` |
| Deployment | Hugging Face Spaces (Docker) |

---

## Run Locally

```bash
git clone https://github.com/saadhna25/rag-pdf-qa.git
cd rag-pdf-qa
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
streamlit run app.py
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

---

## Project Structure

```
rag-pdf-qa/
├── app.py               # Streamlit UI + full RAG pipeline
├── rebuild_index.py     # Standalone script to rebuild FAISS index
├── requirements.txt
└── README.md
```

---

## Features

- Upload any PDF — not hardcoded to one document
- Session-based chat history
- Dark purple UI with gold accents
- Answers grounded strictly in document context — no hallucination outside source material
- Fast inference via Groq (sub-second response times)
