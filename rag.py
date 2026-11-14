import requests
import numpy as np
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"

def get_embedding(text):
    response = requests.post(f"{OLLAMA_URL}/api/embed", json={"model": "nomic-embed-text:latest", "input": text})
    return np.array(response.json()["embeddings"][0])

# def get_similar_texts(query, k=5):
#     embedding = get_embedding(query)
#     response = requests.post(f"{OLLAMA_URL}/api/search", json={"model": "nomic-embed-text", "input": embedding, "k": k})
#     return [match["text"] for match in response.json()["matches"]]

# def load_documents(directory):
#     documents = []
#     for file in Path(directory).glob("*.txt"):
#         with open(file, "r") as f:
#             documents.append(f.read())
#     return documents

# def save_documents(documents, directory):
#     for i, document in enumerate(documents):
#         with open(Path(directory) / f"document_{i}.txt", "w") as f:

def ask_llm(prompt):
    response = requests.post(f"{OLLAMA_URL}/api/chat", json={"model": "qwen3:4b-instruct", "messages": [{"role": "user", "content": prompt}],"stream": False})
    return response.json()["message"]["content"]

chunks = []
for file in Path("data").glob("*.md"):
    chunks+= [p.strip() for p in file.read_text().split("\n\n") if p.strip()]

chunks_embeddings = [get_embedding(chunk) for chunk in chunks]
chunks_embeddings = np.array(chunks_embeddings)
print(f"Loaded {len(chunks_embeddings)} chunks")

def search_chunks(query, k=3):
    query_embedding = get_embedding(query)
    distances = np.linalg.norm(chunks_embeddings - query_embedding, axis=1)
    return distances.argsort()[:k]

print(search_chunks("What is the capital of France?"))
def main():
    while True:
        query = input("You: ")
        if query == "exit":
            break
        indices = search_chunks(query)
        context = "\n\n".join([chunks[i] for i in indices])
        print(context)
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        response = ask_llm(prompt)
        print(response)

if __name__ == "__main__":
    main()