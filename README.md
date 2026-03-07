# GenAI Chat Assistant

## 🏗️ Architecture Diagram

```
User <-> HTML/JS Chat UI <-> Flask Backend (Python)
        |                |
        |                v
        |         RAG Pipeline:
        |         - Load docs.json
        |         - Chunk documents
        |         - Generate embeddings (Mistral API)
        |         - Store embeddings (in-memory)
        |         - Similarity search (cosine)
        |         - Inject context into LLM prompt
        |         - Generate response (Mistral LLM API)
        v
    Display reply
```

---

## 🔄 RAG Workflow Explanation

1. User sends a question via chat UI.
2. Backend chunks documents and generates embeddings for each chunk.
3. User query is embedded.
4. Cosine similarity is used to find top 3 relevant chunks.
5. Retrieved chunks are injected into the LLM prompt.
6. LLM generates a grounded response.
7. Response is sent back to the frontend.

---

## 🧠 Embedding Strategy

- Each document is split into chunks (300–500 tokens).
- Each chunk is sent to Mistral’s embedding API (`mistral-embed`) to get a vector.
- Embeddings are stored in memory for fast retrieval.

---

## 🔍 Similarity Search Explanation

- User query is embedded using Mistral.
- Cosine similarity is calculated between the query embedding and all chunk embeddings.
- Top 3 chunks above a similarity threshold are selected.
- If no chunk is similar enough, a fallback response is returned.

---

## 📝 Prompt Design Reasoning

- Prompt includes:
    - Retrieved context chunks
    - Last 3–5 message pairs (conversation history)
    - User question
- This ensures the LLM response is grounded in real documents and context, reducing hallucinations.

---

## ⚙️ Setup Instructions

1. Clone the repository.
2. Create a virtual environment and activate it.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Add your Mistral API key in app.py.
5. Run the Flask app:
   ```
   python app.py
   ```
6. Open your browser at `http://localhost:5000`.

---
