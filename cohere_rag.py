import os
import cohere
import numpy as np

from dotenv import load_dotenv

load_dotenv()

co = cohere.ClientV2(os.getenv("COHERE_API_KEY"))

documents = [
    # Education
    "Nawfal is a Computer Science student at the University of Victoria (UVic), entering third year in Fall 2026 with a 4.0 GPA.",
    "Nawfal previously studied at Lahore University of Management Sciences (LUMS) in Pakistan, achieving a 3.68 GPA and making the Dean's Honor List in Year 1 before transferring to UVic.",
 
    # Experience
    "Nawfal works as a Frontend Developer at ImmunoLab (Allergo Express Med), building and maintaining a production medical clinic frontend in Next.js and TypeScript serving 13 branch locations.",
    "At ImmunoLab, Nawfal built an online order portal, same-day results dashboard, and secure patient authentication flows with structured IgE panel result displays, deployed to Vercel.",
    "Nawfal completed a Data Engineering internship at Love for Data (LFD) in Karachi, where he automated a literacy data pipeline using Python and Power BI across 50+ districts in Pakistan.",
    "At Love for Data, Nawfal's pipeline processed 1,200+ records and reduced manual data processing time by 60%, enabling regional trend analysis.",
    "At Love for Data, Nawfal built an OpenCV image preprocessing pipeline for an Animal Passport Control MVP, achieving 80 accuracy in cattle identification through mobile image capture.",
 
    # Cybersecurity
    "Nawfal has completed a certificate in Vulnerability Management and has used Qualys for vulnerability scanning at a basic level.",
    "Nawfal has foundational knowledge of cybersecurity concepts including vulnerability assessment, network security, and risk management.",
 
    # Cloud
    "Nawfal has AWS certification and has used AWS for cloud deployment and storage across multiple projects.",
    "Nawfal is familiar with cloud infrastructure concepts including EC2, S3, and deployment pipelines on AWS.",
 
    # AI Tutor Project
    "Nawfal is developing a full-stack AI tutoring platform called Digital Studies AI Tutor using React, Vite, FastAPI, and PostgreSQL.",
    "The AI tutor platform has a FastAPI Python backend with 12 RESTful CRUD endpoints and a PostgreSQL schema supporting 10,000+ flashcards.",
    "The AI tutor platform integrates the OpenAI API with JWT authentication and role-based access control (RBAC) for students and instructors.",
    "The AI tutor uses prompt engineering strategies that adapt quiz difficulty and study materials based on individual student performance metrics.",
 
    # Lisp Interpreter
    "Nawfal is building a Lisp interpreter from scratch in C, following the Build Your Own Lisp book (buildyourownlisp.com).",
    "The Lisp interpreter currently supports a REPL, arithmetic expressions, S-Expressions, and Q-Expressions using the mpc parser combinator library.",
    "The Lisp interpreter uses a custom lval type system with manual heap allocation and a recursive delete function for memory management.",
    "A key challenge in the Lisp interpreter was implementing correct memory ownership across lval_pop and lval_take operations to avoid use-after-free bugs.",
    "The Lisp interpreter implements Q-Expressions by reusing the lval struct with a type tag, and builtin_eval converts Q-Expressions back to S-Expressions by flipping the type field.",
 
    # Habitat Builder
    "Habitat Builder is a 3D interactive space habitat design tool built with React, TypeScript, and Three.js for NASA Space Apps Challenge 2025.",
    "Habitat Builder placed 2nd locally in Victoria, BC and became a Global Nominee among 15,000+ teams worldwide at NASA Space Apps 2025.",
    "Habitat Builder maintains 60 FPS with 50+ concurrent modules through optimized spatial data structures and a real-time rendering pipeline.",
    "Habitat Builder includes a resource simulation engine modelling 5 interconnected life-support systems with constraint validation processing 100+ interdependency checks per second.",
 
    # BC Homeless Identity Portal
    "The BC Homeless Identity Portal was built at UVec X Inspire 2026 Hackathon and won the Best Use of AI award.",
    "The BC Homeless Identity Portal enables shelter outreach workers to register individuals and perform facial identity verification without physical ID.",
    "The portal uses Face-api.js for 1:N facial identification with privacy-aware biometric handling in real time.",
    "The portal was built with Next.js, TypeScript, and Prisma ORM with a relational schema supporting multi-role access control for outreach workers and administrators.",
 
    # Minesweeper
    "Nawfal implemented Minesweeper in C++ using OOP design and an optimized flood-fill algorithm revealing a 24x24 grid in under 50ms.",
    "The Minesweeper project uses an OpenGL rendering system sustaining 50 FPS through draw call batching and state caching.",
    "The Minesweeper project features an event-driven architecture handling real-time input across variable grid configurations.",
 
    # AVR LED Message Board
    "Nawfal programmed an AVR ATmega2560 microcontroller in assembly to drive an LED matrix display supporting 256-character scrollable messages at 5Hz refresh rate.",
    "The AVR project uses interrupt-service routines and hardware timers with microsecond precision, optimizing memory utilization through efficient buffer management.",
 
    # Technical Skills
    "Nawfal's programming languages include Python, C, C++, Java, Assembly (AVR), Unix, Bash, JavaScript, and TypeScript.",
    "Nawfal's web frameworks include React.js, Next.js, Node.js, Three.js, Express.js, and FastAPI.",
    "Nawfal's developer tools include AWS, Git, Docker, Linux, PostgreSQL, Prisma, OpenCV, and OpenGL.",
    "Nawfal's CS concepts include Object Oriented Programming, Data Structures and Algorithms, Unit Testing, and Design Patterns.",
 
    # Personal
    "Nawfal grew up in Pakistan and has lived across three countries, giving him firsthand experience with cross-border financial and logistical friction.",
    "Nawfal is based in Victoria, BC, Canada and is available for co-op work terms starting September 2026.",
    "Nawfal can be contacted at nlodhi@uvic.ca or +1 (236) 882-0088.",
    "Nawfal's GitHub is github.com/nawfallodhi and his LinkedIn is linkedin.com/in/nawfal-lodhi.",
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
            "content": f"You are a helpful assistant that answers questions about Nawfal Lodhi based only on the context below. Be concise and direct.\n\nContext:\n{context}\n\nQuestion: {query}"
        }]
    )
    return res.message.content[0].text

print("Loading Nawfal's knowledge base...")
doc_embeddings = embed(documents)
print("Ready! Ask anything about Nawfal.\n")

queries = [
    "What has Nawfal built with AI?",
    "What low level programming has Nawfal done?",
    "What production systems is Nawfal currently maintaining?",
    "What hackathons has Nawfal won?",
    "What backend frameworks does Nawfal know?",
]

while True:
    query = input("Q: ").strip()
    if query.lower() in ["exit", "quit", "q"]:
        break
    top_idx = search(query, doc_embeddings)
    context = [documents[i] for i in top_idx]
    print(f"A: {answer(query, context)}\n")