from typing import Dict, Any, Optional
from src.agents.observer import process_input
from src.agents.archivist import (
    ensure_collections, check_history, check_similarity_image, retrieve_text_context
)
from src.agents.judge import judge

def run_pipeline(
    user_id: str,
    text: str = "",
    image_bytes: Optional[bytes] = None,
    audio_bytes: Optional[bytes] = None
) -> Dict[str, Any]:
    ensure_collections()

    # Step 1: Observe
    observed = process_input(user_id=user_id, text=text, image_bytes=image_bytes, audio_bytes=audio_bytes)

    # Step 2: Context lookup
    history = check_history(user_id)
    img_sim = check_similarity_image(observed["vectors"]["image"])
    kb_hits = retrieve_text_context(observed.get("combined_text", ""), observed["vectors"]["text"])

    archivist_ctx = {
        "history": history,
        "image_similarity": img_sim,
        "text_context": kb_hits
    }

    # Step 3: Judge
    verdict = judge(observed, archivist_ctx)

    # Summarized step log (safe to display)
    steps = [
        "Image analyzed." if image_bytes else "No image provided.",
        "OCR text extracted." if observed["raw"].get("ocr_text") else "No OCR text.",
        "Audio transcribed." if observed["raw"].get("asr_text") else "No audio provided or transcription.",
        "Qdrant Memory: User history fetched.",
        f"Qdrant Memory: {len(img_sim.get('matches', []))} similar imagery hits.",
        f"Qdrant Memory: {len(kb_hits)} KB context hits.",
        f"Final Verdict computed."
    ]

    return {
        "steps": steps,
        "observed": observed,
        "archivist": archivist_ctx,
        "verdict": verdict
    }
