import os
import json
import uuid
import faiss
import numpy as np
from pypdf import PdfReader
from typing import List, Dict

# Local embedding model
LOCAL_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_local_model = None  # Global variable to hold the model once loaded

# Split text into manageable chunks for embedding
def chunk_text(text: str, max_tokens=400, overlap=60) -> List[str]:
    words = text.split()
    chunks, cur = [], []
    cur_tokens = 0
    for w in words:
        cur.append(w)
        cur_tokens += 1
        if cur_tokens >= max_tokens:
            chunks.append(" ".join(cur))
            cur = cur[-overlap:]  # overlap previous words
            cur_tokens = len(cur)
    if cur:
        chunks.append(" ".join(cur))
    return chunks

# Load PDF and extract text chunks
def load_pdfs(folder: str) -> List[Dict]:
    docs = []
    for filename in os.listdir(folder):
        if not filename.lower().endswith(".pdf"):
            continue
        path = os.path.join(folder, filename)
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            for chunk in chunk_text(text):
                docs.append({
                    "id": str(uuid.uuid4()),
                    "source": filename,
                    "page": i + 1,
                    "text": chunk.strip()
                })
    return docs

# Embed text chunks using local model
def get_local_embeddings(texts: List[str]) -> np.ndarray:
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer(LOCAL_MODEL_NAME)
    embeddings = _local_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype("float32")

# Save FAISS vector store and metadata
def save_store(vectors: np.ndarray, metadatas: List[Dict], out_dir="store"):
    os.makedirs(out_dir, exist_ok=True)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product = cosine similarity for normalized vectors
    faiss.normalize_L2(vectors)
    index.add(vectors)
    faiss.write_index(index, os.path.join(out_dir, "index.faiss"))
    # Save metadata for each chunk
    with open(os.path.join(out_dir, "meta.jsonl"), "w", encoding="utf-8") as f:
        for m in metadatas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f" Saved {len(metadatas)} chunks to {out_dir}/")

# Main pipeline
def main():
    docs = load_pdfs("data")  # folder with your PDF(s)
    if not docs:
        raise SystemExit(" No PDFs found in ./data")
    texts = [d["text"] for d in docs]
    print(f"Embedding {len(texts)} chunks locally...")
    vectors = get_local_embeddings(texts)
    save_store(vectors, docs)

if __name__ == "__main__":
    main()
