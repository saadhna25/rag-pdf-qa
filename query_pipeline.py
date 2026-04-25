import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# ---- Load chunks and FAISS index ----
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

index = faiss.read_index("faiss_index.index")
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ---- Retrieve using FAISS ----
def retrieve_chunks(query, k=3):
    query_embedding = embedding_model.encode([query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding).astype("float32")
    distances, indices = index.search(query_embedding, k)
    retrieved = [chunks[i] for i in indices[0]]
    
    # Fallback: always include limitations chunk if query mentions limitations
    limitation_keywords = ['limitation', 'limitations', 'drawback', 'weakness', 'constraint']
    if any(kw in query.lower() for kw in limitation_keywords):
        if chunks[37] not in retrieved:
            retrieved[0] = chunks[37]  # put it first, guaranteed in context
    
    return retrieved

# ---- Build prompt ----
def build_prompt(query, context_chunks):
    context = "\n\n".join(context_chunks)
    return f"""You are a helpful assistant. Answer the question based only on the context below.
If the answer is not in the context, say "I don't know based on the provided document."

Context:
{context}

Question: {query}

Answer:"""

# ---- Ask ----
def ask(query):
    context_chunks = retrieve_chunks(query)
    prompt = build_prompt(query, context_chunks)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    print("RAG Pipeline ready. Type 'quit' to exit.\n")
    while True:
        query = input("Ask a question: ")
        if query.lower() == "quit":
            break
        answer = ask(query)
        print(f"\nAnswer: {answer}\n")
        print("-" * 50)