import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import CHUNKS_PATH, EMBEDDING_MODEL, FAISS_INDEX_PATH, GENERATOR_MODEL, MAX_CONTEXT_CHARS, MAX_CONTEXT_CHUNKS, TOP_K

_index = None
_corpus = None
_embed_model = None
_tokenizer = None
_model_gen = None

# ── Load index & data ─────────────────

with open(CHUNKS_PATH, "rb") as f:
    corpus = pickle.load(f)

def load_model():
    global _embed_model, _tokenizer, _model_gen, _index
    
    _index = faiss.read_index(FAISS_INDEX_PATH)
    # ── Embedding model ───────────────────
    _embed_model = SentenceTransformer(EMBEDDING_MODEL)

    # ── Generator (T5) ────────────────────
    _tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
    _model_gen = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_MODEL)

# ── Retrieval ─────────────────────────
def search(query, k=TOP_K):
    q_vec = _embed_model.encode([query], normalize_embeddings=True).astype("float32")
    scores, idxs = _index.search(q_vec, k)
    results = []
    for i, score in zip(idxs[0], scores[0]):
        results.append({
            "text": corpus[i]["text"],
            "score": float(score)
        })
    return results


# ── Generate ──────────────────────────
def generate_answer(question, contexts):
    context = "\n".join([c["text"][:MAX_CONTEXT_CHARS] for c in contexts[:MAX_CONTEXT_CHUNKS]])
    prompt = f"question: {question} context: {context}"
    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    output = _model_gen.generate(
        **inputs,
        max_new_tokens=80,
        num_beams=4
    )

    return _tokenizer.decode(output[0], skip_special_tokens=True)


# ── RAG pipeline ──────────────────────
def rag(query):
    load_model()
    docs = search(query)
    answer = generate_answer(query, docs)
    return answer, docs