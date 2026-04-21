from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle

from config import CHUNKS_PATH, EMBEDDING_MODEL, FAISS_INDEX_PATH

# ── 1. Load dataset ─────────────────────
train_dataset = load_dataset("taidng/UIT-ViQuAD2.0", split="train")
# dataset = load_dataset("taidng/UIT-ViQuAD2.0")["train"]

seen = set()
corpus = []

for item in train_dataset:
    text = item["context"]
    if text and len(text.strip()) > 50:
        if text not in seen:
            seen.add(text)
            corpus.append({"text": text})

print("Corpus size:", len(corpus))

# ── 2. Encode ───────────────────────────
model = SentenceTransformer(EMBEDDING_MODEL)

texts = [c["text"] for c in corpus]

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    batch_size=32
).astype("float32")

print("Embedding shape:", embeddings.shape)

# ── 3. FAISS index ──────────────────────
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

print("FAISS size:", index.ntotal)

# ── 4. Save ─────────────────────────────
faiss.write_index(index, FAISS_INDEX_PATH)

with open(CHUNKS_PATH, "wb") as f:
    pickle.dump(corpus, f)

print("Saved OK")