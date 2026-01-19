from sentence_transformers import SentenceTransformer
from functools import lru_cache
from typing import Optional
from PIL import Image
import io
from src.config import IMAGE_EMBED_MODEL

@lru_cache(maxsize=1)
def _model():
    return SentenceTransformer(IMAGE_EMBED_MODEL)

def embed_image(image_bytes: bytes) -> Optional[list]:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        emb = _model().encode([img], normalize_embeddings=True)  # returns ndarray (1, dim)
        return emb[0].tolist()
    except Exception:
        return None
