import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

TEXT_EMBED_MODEL = os.getenv("TEXT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
IMAGE_EMBED_MODEL = os.getenv("IMAGE_EMBED_MODEL", "sentence-transformers/clip-ViT-B-32")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

ASR_LANGUAGE_HINT = os.getenv("ASR_LANGUAGE_HINT", "en")

# Qdrant collection names
COLL_KB = "gbv_knowledge_base"      # text vectors (384-d default)
COLL_IMG = "toxic_imagery"          # image vectors (512-d for CLIP ViT-B/32)
COLL_REP = "user_reputation"        # user profile vectors (384-d default)
