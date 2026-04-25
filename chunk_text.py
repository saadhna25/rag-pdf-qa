import pdfplumber
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

#---Step 1 extract from pdf----

#open the pdf using pdfplumber (handles spacing better than PyPDF2)
with pdfplumber.open("PriorCCI.pdf") as pdf:

    #find out how many pages the pdf has
    num_pages = len(pdf.pages)
    print(f"Total pages: {num_pages}")

    #loop through every page and extract its text
    full_text = " "
    for page in pdf.pages:
        text = page.extract_text()
        if text: #some pages might be images with no text - skip those
            full_text += text + "\n"

#clean up white space
full_text = re.sub(r'\n+','\n', full_text)
full_text = re.sub(r' +', ' ', full_text)

#----Step 2 split text into chunks----
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)
chunks = splitter.split_text(full_text)

#----Step 3 print results----
print(f"Total chunks created: {len(chunks)}")
print(f"\n ---- Chunk 1 ----")
print(chunks[0])
print(f"\n --- Chunk 2 ---")
print(chunks[1])
print(f"\n --- Chunk 3 ---")
print(chunks[2])