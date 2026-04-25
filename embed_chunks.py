import pdfplumber
import re
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

#----step3: load the embedding model----
print("loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

#----step4: embed all chunks----
print(f"Embedding {len(chunks)} chunks...")
embeddings = model.encode(chunks, show_progress_bar = True)

#----step5: print results----
print(f"\nTotal chunks embedded: {len(chunks)}")
print(f"Shape of embeddings: {embeddings.shape}")
print(f"\n--- Sample: chunk 1 text ---")
print(chunks[0])
print(f"n--- Sample: chunk 1 vector (first 8 values) ---")
print(chunks[0][:8])