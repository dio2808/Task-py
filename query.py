import os
import json
import faiss
import numpy as np
from typing import List, Dict
import subprocess

# Local embedding model
LOCAL_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_local_model = None

# Embed query or chunks
def get_local_embeddings(texts: List[str]) -> np.ndarray:
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer(LOCAL_MODEL_NAME)
    embeddings = _local_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype("float32")

# Load FAISS store and metadata
def load_store(store_dir="store"):
    index = faiss.read_index(os.path.join(store_dir, "index.faiss"))
    meta = []
    with open(os.path.join(store_dir, "meta.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return index, meta

# Search top-k chunks for a query
def search(query: str, k=5):
    index, meta = load_store()
    qv = get_local_embeddings([query])
    faiss.normalize_L2(qv)
    D, I = index.search(qv, k)
    hits = []
    for idx, score in zip(I[0], D[0]):
        m = meta[int(idx)]
        hits.append({"score": float(score), **m})
    return hits

# Format prompt for ChatGPT or local LLM
def make_prompt(question: str, chunks: List[Dict]) -> str:
    context = "\n\n".join(
        f"[{i+1}] ({c['source']} p.{c['page']})\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    return f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"

# Optional: call local Ollama model 
def answer_with_ollama(prompt: str, model="mistral") -> str:
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True, text=True
        )
        return result.stdout.strip()
    except FileNotFoundError:
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str, nargs="+", help="Your question")
    parser.add_argument("--k", type=int, default=5, help="top-k passages to retrieve")
    parser.add_argument("--ollama", action="store_true", help="Use local Ollama model")
    args = parser.parse_args()
    question = " ".join(args.question)

    hits = search(question, k=args.k)
    if not hits:
        print(" No results found.")
        return

    # Show top passages
    print("\n Top passages:")
    for i, h in enumerate(hits, 1):
        print(f"{i}. {h['source']} p.{h['page']}  score={h['score']:.3f}")

    # Build prompt
    prompt = make_prompt(question, hits)

    if args.ollama:
        answer = answer_with_ollama(prompt)
        if answer:
            print("\n Ollama Answer:\n")
            print(answer)
        else:
            print("\n Ollama not found. Install from https://ollama.ai/")
    else:
        print("\n--- Copy this into ChatGPT (free) to get an answer ---\n")
        print(prompt)

if __name__ == "__main__":
    main()
