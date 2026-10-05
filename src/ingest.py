from load_documents import load_all_pdfs
from chunking import chunk_text
from vector_store import build_index, save_index

def run_ingestion():
    docs = load_all_pdfs("data/raw_pdfs")
    all_chunks = []
    for name, text in docs.items():
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
    print(f"Total chunks: {len(all_chunks)}")
    index, embeddings = build_index(all_chunks)
    save_index(index, all_chunks)
    print("Index saved.")

if __name__ == "__main__":
    run_ingestion()