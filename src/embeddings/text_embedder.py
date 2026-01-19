from sentence_transformers import SentenceTransformer
from functools import lru_cache
from typing import List
from src.config import TEXT_EMBED_MODEL

@lru_cache(maxsize=1)
def _model():
    return SentenceTransformer(TEXT_EMBED_MODEL)

def embed_text(texts: List[str]):
    if isinstance(texts, str):
        texts = [texts]
    return _model().encode(texts, normalize_embeddings=True)
