from pathlib import Path

# ─────────────────────────────
# MODEL CONFIG
# ─────────────────────────────
EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"
GENERATOR_MODEL = "VietAI/vit5-base"

# ─────────────────────────────
# FILE PATH CONFIG
# ─────────────────────────────
BASE_DIR = Path(".")

FAISS_INDEX_PATH = BASE_DIR / "vector-v2.index"
CHUNKS_PATH      = BASE_DIR / "chunks.pkl"
META_PATH        = BASE_DIR / "index_meta.json"

# ─────────────────────────────
# RAG CONFIG
# ─────────────────────────────
TOP_K = 3
MAX_CONTEXT_CHARS = 500
MAX_CONTEXT_CHUNKS = 2

# FAISS CONFIG
INDEX_TYPE = "IndexFlatIP"   # cosine similarity