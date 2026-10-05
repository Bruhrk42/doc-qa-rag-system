from vector_store import load_index, search
from generate import generate_answer

def ask(query, k=3):
    index, chunks = load_index()
    relevant_chunks = search(query, index, chunks, k=k)
    answer = generate_answer(query, relevant_chunks)
    return answer, relevant_chunks

if __name__ == "__main__":
    q = input("Ask a question: ")
    answer, sources = ask(q)
    print("\nAnswer:", answer)
    print("\nSources used:")
    for i, s in enumerate(sources):
        print(f"[{i+1}] {s[:150]}...")