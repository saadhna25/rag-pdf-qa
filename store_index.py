import pdfplumber
import re
import faiss
import numpy as np
import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

#----step1: extract text (same as before)----
with pdfplumber.open("PriorCCI.pdf") as pdf:
    full_text = ""
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

full_text = re.sub(r'\n+', '\n', full_text)
full_text = re.sub(r' +', ' ', full_text)

#----step2: chunk the text (same as before)----
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_text(full_text)

#----step3: load the embedding model, embed all chunks----
print("loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"Embedding {len(chunks)} chunks...")
embeddings = model.encode(chunks, show_progress_bar = True)

#----step4: build faiss index----
dimension = embeddings.shape[1] #384
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))
print(f"faiss index built. Total vectores stored: {index.ntotal}")

#----step5: save index and chunks to disk----
faiss.write_index(index, "faiss_index.index")
with open("chunks.pkl","wb") as f:
    pickle.dump(chunks, f)
print(f"faiss_index.index and chunks.pkl saved")

#----step6: test a search----
query = "What is PriorCCI used for?"
query_vector = model.encode([query])
distances, indices = index.search(np.array(query_vector), k = 3)

print(f"\ntop 3 chunks for: '{query}' ---")
for i, idx in enumerate(indices[0]):
    print(f"\nresult {i+1} (distance: {distances[0][i]:.4f}):")
    print(chunks[idx][:300])