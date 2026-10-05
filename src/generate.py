import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-3.8-flash")
def generate_answer(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}

Answer:"""
    
    response = model.generate_content(prompt)
    return response.text