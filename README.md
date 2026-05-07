# Ask Me Anything (Powered by Cohere)

A RAG pipeline that lets you have a conversation with my resume.

Built with Cohere's Embed v3 and Command R. You ask a question, it finds the most relevant facts from my knowledge base, and Command R answers using only that context. 

## How it works

1. **Embed** : Every document in the knowledge base gets converted into a vector using `embed-english-v3.0`
2. **Search** : Your question gets embedded too, then compared against all document vectors using dot product similarity
3. **Answer** : The top matching documents get passed to `command-r-plus-08-2024` as context, which generates a grounded response

This is the same pattern enterprises use to make their internal docs, contracts, and wikis searchable with natural language. Except the corpus here is just me.

## Try it

```bash
git clone https://github.com/nawfallodhi/cohere-rag-demo
cd cohere-rag-demo
pip install cohere numpy python-dotenv
```

Create a `.env` file:
```
COHERE_API_KEY=your_key_here
```

Run it:
```bash
python cohere_rag.py
```

Then ask things like:
- `What has Nawfal built with AI?`
- `What low level programming has Nawfal done?`
- `What cybersecurity experience does Nawfal have?`
- `What is Nawfal currently building?`
- `How can I contact Nawfal?`

Type `exit` to quit.

## What's in the knowledge base

Pretty much everything. Education, work experience, every project, technical skills, certifications, personal background. 40 documents total. If it's on my resume or GitHub it's probably in here.

## Stack

- [Cohere Embed v3](https://docs.cohere.com/docs/embed) for semantic search
- [Command R+](https://docs.cohere.com/docs/command-r-plus) for grounded generation
- NumPy for the vector math
- python-dotenv for keeping the API key off GitHub

---

Built by Nawfal Mansoor Lodhi while applying to Cohere.  
[github.com/nawfallodhi](https://github.com/nawfallodhi) | [linkedin.com/in/nawfal-lodhi](https://linkedin.com/in/nawfal-lodhi)