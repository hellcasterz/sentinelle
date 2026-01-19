"""
Seed small demo data:
- A few GBV knowledge base entries
- Optional: encode a few local images into toxic_imagery (place sample images and adjust paths)
- A couple of user reputation points
"""
import os
from src.config import COLL_KB, COLL_IMG, COLL_REP
from src.embeddings.text_embedder import embed_text
from src.embeddings.image_embedder import embed_image
from src.agents.archivist import ensure_collections, upsert_kb, upsert_toxic_image, upsert_user_profile

def main():
    ensure_collections()

    # Seed KB
    kb_texts = [
        "Definition: Image-based sexual abuse (often called 'revenge porn') is a form of GBV.",
        "Helpline: If you are in danger, contact local authorities. Seek support from trusted organizations.",
        "Threats, stalking, and doxxing are forms of GBV when directed based on gender.",
        "Common obfuscations include numeric substitutions (e.g., r4pe) and leetspeak."
    ]
    for t, vec in zip(kb_texts, embed_text(kb_texts)):
        upsert_kb(text=t, text_vec=vec.tolist(), tags=["kb"])

    # Optional: seed images (put a few sample images in ./data/toxic/*.jpg)
    toxic_dir = "./data/toxic"
    if os.path.isdir(toxic_dir):
        for fname in os.listdir(toxic_dir):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            with open(os.path.join(toxic_dir, fname), "rb") as f:
                emb = embed_image(f.read())
                if emb:
                    upsert_toxic_image(emb, caption=f"seed:{fname}", label="toxic")

    # Seed user reputation examples
    profiles = [
        ("user123", "Prior harassment reports in July.", 0.8),
        ("user456", "No prior flags.", 0.1),
    ]
    for uid, summary, score in profiles:
        vec = embed_text([summary])[0].tolist()
        upsert_user_profile(user_id=uid, profile_vec=vec, toxicity_score=score, summary=summary)

    print("Seeding complete.")

if __name__ == "__main__":
    main()
