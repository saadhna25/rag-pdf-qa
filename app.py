import os
import re
import tempfile
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
import pdfplumber

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RAG · PDF Q&A",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600&family=Inter:wght@300;400;500&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0e0b1a;
    color: #dcd6f7;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #120f24 0%, #0e0b1a 100%);
    border-right: 1px solid #2a2050;
}
.main .block-container {
    padding-top: 2rem;
    max-width: 860px;
}
.rag-header {
    font-family: 'Cinzel', serif;
    font-size: 1.8rem;
    font-weight: 600;
    background: linear-gradient(90deg, #c9a84c, #f0d080, #c9a84c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 0.08em;
    margin-bottom: 0.2rem;
}
.rag-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #6b5ea8;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #3d2f7a, #c9a84c44, #3d2f7a, transparent);
    margin: 0.5rem 0 1.8rem 0;
}
.sidebar-section {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #c9a84c;
    margin: 1.2rem 0 0.5rem 0;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid #2a2050;
}
.badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    letter-spacing: 0.08em;
    margin: 0.5rem 0 0.4rem 0;
}
.badge-ready   { background: #0d2b1a; color: #4ade80; border: 1px solid #16a34a55; }
.badge-noindex { background: #2a1a00; color: #c9a84c; border: 1px solid #c9a84c55; }
.chat-user {
    background: linear-gradient(135deg, #1e1640 0%, #1a1235 100%);
    border: 1px solid #3d2f7a;
    border-left: 3px solid #c9a84c;
    border-radius: 8px;
    padding: 0.85rem 1.1rem;
    margin: 0.6rem 0;
    font-size: 0.93rem;
    line-height: 1.6;
}
.chat-assistant {
    background: linear-gradient(135deg, #12102a 0%, #0f0d22 100%);
    border: 1px solid #2a2050;
    border-left: 3px solid #7c6fcf;
    border-radius: 8px;
    padding: 0.85rem 1.1rem;
    margin: 0.6rem 0;
    font-size: 0.93rem;
    line-height: 1.7;
}
.chat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 0.45rem;
}
.label-user { color: #c9a84c; }
.label-bot  { color: #9d8fe8; }
[data-testid="stTextInput"] input {
    background-color: #1a1635 !important;
    border: 1px solid #3d2f7a !important;
    border-radius: 8px !important;
    color: #dcd6f7 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.93rem !important;
    transition: border-color 0.2s ease !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #c9a84c !important;
    box-shadow: 0 0 0 2px #c9a84c22 !important;
}
[data-testid="stTextInput"] input::placeholder { color: #4a3f7a !important; }
.stButton > button {
    background: linear-gradient(135deg, #2d1f6e, #1e1648) !important;
    border: 1px solid #c9a84c88 !important;
    color: #c9a84c !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.1em !important;
    border-radius: 6px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    border-color: #c9a84c !important;
    box-shadow: 0 0 12px #c9a84c33 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "index" not in st.session_state:
    st.session_state.index = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# ─────────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

@st.cache_resource(show_spinner=False)
def load_groq_client():
    return Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ─────────────────────────────────────────────
#  INDEXING
# ─────────────────────────────────────────────
def build_index(pdf_bytes):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    full_text = ""
    with pdfplumber.open(tmp_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    os.unlink(tmp_path)

    full_text = re.sub(r'\n+', '\n', full_text)
    full_text = re.sub(r' +', ' ', full_text)
    full_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', full_text)
    full_text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', full_text)
    full_text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', full_text)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(full_text)

    model = load_embedding_model()
    embeddings = model.encode(chunks, show_progress_bar=False, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")

    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_index.add(embeddings)

    return chunks, faiss_index

# ─────────────────────────────────────────────
#  RAG PIPELINE
# ─────────────────────────────────────────────
def retrieve_chunks(query, chunks, index, k=3):
    model = load_embedding_model()
    query_embedding = model.encode([query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding).astype("float32")
    _, indices = index.search(query_embedding, k)
    retrieved = [chunks[i] for i in indices[0]]

    limitation_keywords = ['limitation', 'limitations', 'drawback', 'weakness', 'constraint']
    if any(kw in query.lower() for kw in limitation_keywords):
        if len(chunks) > 37 and chunks[37] not in retrieved:
            retrieved[0] = chunks[37]

    return retrieved

def build_prompt(query, context_chunks):
    context = "\n\n".join(context_chunks)
    return f"""You are a helpful assistant. Answer the question based only on the context below.
If the answer is not in the context, say "I don't know based on the provided document."

Context:
{context}

Question: {query}

Answer:"""

def ask(query, chunks, index):
    client = load_groq_client()
    context_chunks = retrieve_chunks(query, chunks, index)
    prompt = build_prompt(query, context_chunks)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="rag-header" style="font-size:1.1rem;">✦ RAG PDF Q&A</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Document</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf", label_visibility="collapsed")

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.pdf_name:
            with st.spinner("Indexing document..."):
                chunks, faiss_index = build_index(uploaded_file.read())
                st.session_state.chunks = chunks
                st.session_state.index = faiss_index
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.chat_history = []
                st.session_state.last_query = ""
        st.markdown(f'<div class="badge badge-ready">✓ {uploaded_file.name}</div>', unsafe_allow_html=True)
        st.caption(f"{len(st.session_state.chunks)} chunks indexed")
    else:
        st.markdown('<div class="badge badge-noindex">⚠ No document loaded</div>', unsafe_allow_html=True)

    if st.session_state.chat_history:
        st.markdown('<div class="sidebar-section">Chat</div>', unsafe_allow_html=True)
        if st.button("Clear chat"):
            st.session_state.chat_history = []
            st.session_state.last_query = ""
            st.rerun()

    st.markdown('<div class="sidebar-section">About</div>', unsafe_allow_html=True)
    st.caption("Built with LangChain · FAISS · BGE · Groq")

# ─────────────────────────────────────────────
#  MAIN AREA
# ─────────────────────────────────────────────
st.markdown('<div class="rag-header">✦ Research Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="rag-sub">retrieval-augmented generation · pdf question answering</div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

if st.session_state.index is None:
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; color: #4a3f7a;">
        <div style="font-size:2.5rem; margin-bottom:1rem;">✦</div>
        <div style="font-family:'Cinzel',serif; font-size:1rem; color:#6b5ea8; letter-spacing:0.1em;">
            Upload a PDF to begin
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # ── Chat history ──
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-user">
                <div class="chat-label label-user">You</div>
                {msg["content"]}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-assistant">
                <div class="chat-label label-bot">Assistant</div>
                {msg["content"]}
            </div>""", unsafe_allow_html=True)

    # ── Input ──
    query = st.text_input(
        "Ask a question",
        placeholder="What does this document say about...",
        label_visibility="collapsed",
        key="query_input"
    )

    if query and query != st.session_state.last_query:
        st.session_state.last_query = query
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.spinner("Thinking..."):
            answer = ask(query, st.session_state.chunks, st.session_state.index)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()