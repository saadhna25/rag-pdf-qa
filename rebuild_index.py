import pdfplumber
import re
import pickle
import numpy as np
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

#----step 1: extract text----
with pdfplumber.open("PriorCCI.pdf") as pdf:
    full_text = ""
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

#----step 2: clean text----
full_text = re.sub(r'\n+', '\n', full_text)
full_text = re.sub(r' +', ' ', full_text)
#fix merged words — insert space before capital letters mid-word
full_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', full_text)
#fix merged numbers and words
full_text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', full_text)
full_text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', full_text)

print(f"total characters: {len(full_text)}")
print("\n--- sample (first 500 chars) ---")
print(full_text[:500])

#----step 3: chunk----
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
chunks = splitter.split_text(full_text)
print(f"\nTotal chunks: {len(chunks)}")
print("\n--- Chunk 1 ---")
print(chunks[0])


#----step 4: embed----
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
embeddings = model.encode(chunks, show_progress_bar=True, normalize_embeddings=True)
embeddings = np.array(embeddings).astype("float32")

#----step 5: save new faiss index----
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "faiss_index.index")

with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print(f"\nrebuilt index with {index.ntotal} vectors saved!")