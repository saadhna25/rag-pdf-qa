import pdfplumber
import re

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
        
#print the first 2000 characters to check quality
print(full_text[:2000])
print(f"Total characters extracted: {len(full_text)}")