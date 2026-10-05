from pypdf import PdfReader
import os

def load_pdf_text(filepath):
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def load_all_pdfs(folder_path):
    documents = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            documents[filename] = load_pdf_text(path)
    return documents

if __name__ == "__main__":
    docs = load_all_pdfs("data/raw_pdfs")
    print(f"Loaded {len(docs)} documents")
    for name, text in docs.items():
        print(f"{name}: {len(text)} characters")