import os
import time
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
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        if "429" in str(e) or "ResourceExhausted" in str(e):
            return "⚠️ Rate limit reached (free tier allows 5 requests/minute). Please wait about 30 seconds and try again."
        return f"⚠️ Something went wrong: {str(e)}"