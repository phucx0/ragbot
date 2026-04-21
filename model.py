import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config import CHUNKS_PATH, EMBEDDING_MODEL, FAISS_INDEX_PATH, GENERATOR_MODEL

# ── Load index & data ─────────────────
index = faiss.read_index(FAISS_INDEX_PATH)

with open(CHUNKS_PATH, "rb") as f:
    corpus = pickle.load(f)

# ── Embedding model ───────────────────
embed_model = SentenceTransformer(EMBEDDING_MODEL)

# ── Generator (T5) ────────────────────
tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
model_gen = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_MODEL)


# ── Retrieval ─────────────────────────
def search(query, k=3):
    q_vec = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(q_vec, k)
    results = []
    for i, score in zip(idxs[0], scores[0]):
        results.append({
            "text": corpus[i]["text"],
            "score": float(score)
        })
    return results


# ── Generate ──────────────────────────
def generate_answer(question, contexts):
    context = "\n".join([c["text"][:500] for c in contexts])
    prompt = f"question: {question} context: {context}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    output = model_gen.generate(
        **inputs,
        max_new_tokens=80,
        num_beams=4
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)


# ── RAG pipeline ──────────────────────
def rag(query):
    docs = search(query)
    answer = generate_answer(query, docs)
    return answer, docs