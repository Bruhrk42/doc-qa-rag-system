import os
import time
from dotenv import load_dotenv
import google.generativeai as genai
import faiss
import numpy as np
import pickle

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

EMBED_MODEL = "models/gemini-embedding-001"

def embed_texts(texts, task_type="retrieval_document", batch_delay=1):
    """Embed a list of texts one at a time (free tier has low rate limits)."""
    embeddings = []
    for i, text in enumerate(texts):
        result = genai.embed_content(
            model=EMBED_MODEL,
            content=text,
            task_type=task_type
        )
        embeddings.append(result["embedding"])
        if i < len(texts) - 1:
            time.sleep(batch_delay)  # avoid hitting rate limits
    return np.array(embeddings).astype("float32")

def build_index(chunks):
    embeddings = embed_texts(chunks, task_type="retrieval_document")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

def save_index(index, chunks, path="data/index"):
    faiss.write_index(index, f"{path}.faiss")
    with open(f"{path}_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

def load_index(path="data/index"):
    index = faiss.read_index(f"{path}.faiss")
    with open(f"{path}_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def search(query, index, chunks, k=3):
    query_vec = embed_texts([query], task_type="retrieval_query")
    distances, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0]]