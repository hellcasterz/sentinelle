from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, MatchText, PayloadSchemaType, TextIndexParams
)
import uuid

from src.config import (
    QDRANT_URL, QDRANT_API_KEY,
    COLL_KB, COLL_IMG, COLL_REP
)

# Dimensions aligned with the chosen models
DIM_TEXT = 384   # all-MiniLM-L6-v2
DIM_IMAGE = 512  # CLIP ViT-B/32

def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)

def ensure_collections():
    c = get_client()

    # Knowledge base (text vectors + full-text index on 'text')
    if COLL_KB not in [col.name for col in c.get_collections().collections]:
        c.create_collection(
            collection_name=COLL_KB,
            vectors_config={"text": VectorParams(size=DIM_TEXT, distance=Distance.COSINE)}
        )
        c.create_payload_index(
            collection_name=COLL_KB,
            field_name="text",
            field_schema=PayloadSchemaType.TEXT,
            params=TextIndexParams(tokenizer="word")
        )

    # Toxic imagery (image vectors)
    if COLL_IMG not in [col.name for col in c.get_collections().collections]:
        c.create_collection(
            collection_name=COLL_IMG,
            vectors_config={"image": VectorParams(size=DIM_IMAGE, distance=Distance.COSINE)}
        )
        # Optional: maintain captions/notes for hybrid
        c.create_payload_index(
            collection_name=COLL_IMG,
            field_name="caption",
            field_schema=PayloadSchemaType.TEXT,
            params=TextIndexParams(tokenizer="word")
        )

    # User reputation (text vector representing profile summary; plus payload with user_id, score)
    if COLL_REP not in [col.name for col in c.get_collections().collections]:
        c.create_collection(
            collection_name=COLL_REP,
            vectors_config={"profile": VectorParams(size=DIM_TEXT, distance=Distance.COSINE)}
        )
        c.create_payload_index(
            collection_name=COLL_REP,
            field_name="user_id",
            field_schema=PayloadSchemaType.KEYWORD
        )

def upsert_kb(text: str, text_vec: List[float], tags: Optional[List[str]] = None):
    c = get_client()
    pid = str(uuid.uuid4())
    c.upsert(
        collection_name=COLL_KB,
        points=[PointStruct(id=pid, vector={"text": text_vec}, payload={"text": text, "tags": tags or []})]
    )

def upsert_toxic_image(image_vec: List[float], caption: str, label: str):
    c = get_client()
    pid = str(uuid.uuid4())
    c.upsert(
        collection_name=COLL_IMG,
        points=[PointStruct(id=pid, vector={"image": image_vec}, payload={"caption": caption, "label": label})]
    )

def upsert_user_profile(user_id: str, profile_vec: List[float], toxicity_score: float, summary: str):
    c = get_client()
    pid = str(uuid.uuid4())
    c.upsert(
        collection_name=COLL_REP,
        points=[PointStruct(
            id=pid,
            vector={"profile": profile_vec},
            payload={"user_id": user_id, "toxicity_score": toxicity_score, "summary": summary}
        )]
    )

def check_history(user_id: str) -> Dict[str, Any]:
    c = get_client()
    res = c.scroll(
        collection_name=COLL_REP,
        limit=100,
        scroll_filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])
    )
    points = res[0]
    if not points:
        return {"flags": 0, "toxicity_score": 0.0, "notes": []}
    flags = len(points)
    avg_score = sum(p.payload.get("toxicity_score", 0.0) for p in points) / max(flags, 1)
    notes = [p.payload.get("summary", "") for p in points][:5]
    return {"flags": flags, "toxicity_score": round(avg_score, 2), "notes": notes}

def check_similarity_image(image_vec: Optional[List[float]], limit: int = 5) -> Dict[str, Any]:
    if not image_vec:
        return {"matches": []}
    c = get_client()
    hits = c.search(
        collection_name=COLL_IMG,
        query_vector=("image", image_vec),
        limit=limit
    )
    return {
        "matches": [
            {
                "score": float(h.score),
                "label": h.payload.get("label", ""),
                "caption": h.payload.get("caption", "")
            } for h in hits
        ]
    }

def retrieve_text_context(query_text: str, text_vec: Optional[List[float]], limit: int = 5) -> List[Dict[str, Any]]:
    c = get_client()
    filt = Filter(must=[FieldCondition(key="text", match=MatchText(text=query_text))]) if query_text else None
    hits = c.search(
        collection_name=COLL_KB,
        query_vector=("text", text_vec) if text_vec else None,
        query_filter=filt,
        limit=limit
    )
    return [{"score": float(h.score), "text": h.payload.get("text", ""), "tags": h.payload.get("tags", [])} for h in hits]
