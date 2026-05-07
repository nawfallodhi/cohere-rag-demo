import os
import cohere
import numpy as np

from dotenv import load_dotenv

load_dotenv()

co = cohere.ClientV2(os.getenv("COHERE_API_KEY"))

documents = [
    "Nawfal built an AI tutoring platform with FastAPI and OpenAI API that adapts quiz difficulty based on performance.",
    "The BC Homeless Identity Portal uses Face-api.js for 1:N facial recognition without physical ID.",
    "Habitat Builder is a 3D space habitat design tool built with React and Three.js for NASA Space Apps 2025.",
    "Minesweeper was implemented in C++ with OpenGL, using flood-fill revealing a 24x24 grid in under 50ms.",
    "ImmunoLab is a live medical clinic frontend serving 13 locations built with Next.js and TypeScript.",
]

def embed(texts):
    res = co.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type="search_document",
        embedding_types=["float"]
    )
    return np.array(res.embeddings.float)

def search(query, doc_embeddings, top_k=2):
    q_embed = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query",
        embedding_types=["float"]
    ).embeddings.float[0]
    scores = np.dot(doc_embeddings, q_embed)
    return np.argsort(scores)[::-1][:top_k]

def answer(query, context_docs):
    context = "\n".join(f"- {d}" for d in context_docs)
    res = co.chat(
        model="command-r-plus-08-2024",
        messages=[{
            "role": "user",
            "content": f"Answer this question using only the context below.\n\nContext:\n{context}\n\nQuestion: {query}"
        }]
    )
    return res.message.content[0].text

print("Embedding documents...")
doc_embeddings = embed(documents)

queries = [
    "What has Nawfal built with AI?",
    "What graphics projects has Nawfal worked on?",
    "What production software is Nawfal currently maintaining?"
]

for q in queries:
    print(f"\nQ: {q}")
    top_idx = search(q, doc_embeddings)
    context = [documents[i] for i in top_idx]
    print(f"A: {answer(q, context)}")