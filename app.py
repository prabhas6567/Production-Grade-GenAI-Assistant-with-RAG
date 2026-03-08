# app.py
# Entry point for the GenAI Chat Assistant application


from flask import Flask, render_template, request, jsonify
import json

# Gemini API integration
import requests
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Load knowledge base
def load_docs():
    with open('docs.json', 'r', encoding='utf-8') as f:
        return json.load(f)

docs = load_docs()

# Chunk documents
def chunk_document(text, chunk_size=400):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]



# Mistral embedding function
def generate_embedding(text):
    api_key = "yE7eHb8u4EtnDEN23h8bYnt5ohRcvkoG"
    url = "https://api.mistral.ai/v1/embeddings"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "mistral-embed",
        "input": text
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()["data"][0]["embedding"]
    else:
        print(f"Embedding API error {response.status_code}: {response.text}")
        return [0.0]*768  # fallback

# Vector storage (in-memory)
embeddings = []
chunks = []
chunk_metadata = []
for doc in docs:
    for chunk in chunk_document(doc['content']):
        emb = generate_embedding(chunk)
        embeddings.append(emb)
        chunks.append(chunk)
        chunk_metadata.append({"title": doc["title"], "content": chunk})

# Similarity search with threshold
def find_similar_chunks(user_query, threshold=0.5):
    user_emb = generate_embedding(user_query)
    sim = cosine_similarity([user_emb], embeddings)[0]
    top_indices = np.argsort(sim)[-3:][::-1]
    top_scores = sim[top_indices]
    filtered = [(chunk_metadata[i], top_scores[idx]) for idx, i in enumerate(top_indices) if top_scores[idx] >= threshold]
    return filtered, sim


# Mistral LLM API call
def get_llm_response(context_chunks, user_message, history):
    api_key = "yE7eHb8u4EtnDEN23h8bYnt5ohRcvkoG"
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    context = '\n'.join([f"{c['title']}: {c['content']}" for c, _ in context_chunks])
    history_text = '\n'.join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in history])
    prompt = f"Context:\n{context}\nHistory:\n{history_text}\nUser: {user_message}\nAssistant:"
    payload = {
        "model": "mistral-small",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        try:
            reply = response.json()["choices"][0]["message"]["content"]
            tokens_used = response.json().get("usage", {}).get("total_tokens", 0)
            return reply, tokens_used
        except Exception:
            return "Sorry, I couldn't get a response.", 0
    else:
        return "Sorry, I couldn't get a response.", 0

conversation_history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json()
    session_id = data.get('sessionId')
    user_input = data.get('message')
    if not user_input:
        return jsonify({"error": "Missing user message."}), 400
    # Maintain last 5 pairs
    conversation_history.append({'user': user_input, 'assistant': ''})
    if len(conversation_history) > 5:
        conversation_history.pop(0)
    similar_chunks, sim_scores = find_similar_chunks(user_input)
    # Log similarity scores for monitoring/debugging
    print(f"[Similarity Scores] User input: {user_input}")
    print(f"[Similarity Scores] Scores: {sim_scores.tolist()}")
    if not similar_chunks:
        conversation_history[-1]['assistant'] = "Sorry, I don't have enough information to answer that."
        return jsonify({
            "reply": "Sorry, I don't have enough information to answer that.",
            "tokensUsed": 0,
            "retrievedChunks": 0,
            "similarityScores": sim_scores.tolist()
        })
    response, tokens_used = get_llm_response(similar_chunks, user_input, conversation_history[:-1])
    conversation_history[-1]['assistant'] = response
    return jsonify({
        "reply": response,
        "tokensUsed": tokens_used,
        "retrievedChunks": len(similar_chunks),
        "similarityScores": [score for _, score in similar_chunks]
    })

if __name__ == '__main__':
    app.run(debug=True)
