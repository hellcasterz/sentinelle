import json
from typing import Dict, Any
from openai import OpenAI
from src.config import OPENAI_API_KEY, LLM_MODEL

SYS_PROMPT = """You are a safety moderator focused on detecting online gender-based violence (OGBV).
Return a compact JSON object only. Do not include any other text.

Consider:
- Multimodal context (image similarity to toxic imagery, OCR/ASR text)
- Prior behavior from user history
- Slang and obfuscations
- Ambiguity and benign use (e.g., gaming banter)

JSON schema:
{
  "is_gbv": true|false,
  "severity": integer 1-10,
  "reason": string,
  "evidence": string[]
}
"""

def judge(sample: Dict[str, Any], archivist_ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    sample: output from observer.process_input
    archivist_ctx: dict with keys: history, image_similarity, text_context
    """
    if not OPENAI_API_KEY:
        # Fallback if no LLM key is configured
        return {"is_gbv": False, "severity": 1, "reason": "LLM not configured", "evidence": []}

    client = OpenAI(api_key=OPENAI_API_KEY)
    user_content = {
        "user_id": sample.get("user_id"),
        "text": sample.get("combined_text", ""),
        "raw": sample.get("raw", {}),
        "archivist": archivist_ctx
    }

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)}
        ]
    )
    content = resp.choices[0].message.content.strip()
    # Ensure we return JSON only
    try:
        return json.loads(content)
    except Exception:
        return {"is_gbv": False, "severity": 1, "reason": "Invalid LLM response", "evidence": [content[:200]]}
