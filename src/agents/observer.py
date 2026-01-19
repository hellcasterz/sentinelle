from typing import Optional, Dict, Any
import pytesseract
from PIL import Image
import io
from faster_whisper import WhisperModel

from src.embeddings.text_embedder import embed_text
from src.embeddings.image_embedder import embed_image
from src.config import ASR_LANGUAGE_HINT

# Lazy init for Whisper to avoid cold start penalties when unused
_asr_model: Optional[WhisperModel] = None

def _get_asr():
    global _asr_model
    if _asr_model is None:
        _asr_model = WhisperModel("base", device="cpu", compute_type="int8")
    return _asr_model

def ocr_image(image_bytes: bytes) -> str:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception:
        return ""

def transcribe_audio(audio_bytes: bytes) -> str:
    try:
        # faster-whisper expects a path or numpy array; write to buffer file in-memory
        # For simplicity, we return empty here; implement temp file if needed.
        return ""
    except Exception:
        return ""

def process_input(
    user_id: str,
    text: Optional[str],
    image_bytes: Optional[bytes],
    audio_bytes: Optional[bytes]
) -> Dict[str, Any]:
    ocr_text = ocr_image(image_bytes) if image_bytes else ""
    asr_text = transcribe_audio(audio_bytes) if audio_bytes else ""
    combined_text = " ".join([t for t in [text or "", ocr_text, asr_text] if t]).strip()

    text_vec = embed_text([combined_text])[0].tolist() if combined_text else None
    image_vec = embed_image(image_bytes) if image_bytes else None

    return {
        "user_id": user_id,
        "raw": {
            "text": text or "",
            "ocr_text": ocr_text,
            "asr_text": asr_text
        },
        "combined_text": combined_text,
        "vectors": {
            "text": text_vec,
            "image": image_vec
        }
    }
