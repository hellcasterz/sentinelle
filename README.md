# SentinELLE: A Multimodal Multi-Agent Guardian Against Gender-Based Violence

This MVP adds a minimal, working scaffold:

- Observer: OCR + embeddings (text/image), optional audio transcription stub
- Archivist (Qdrant): user reputation, toxic imagery, knowledge base
- Judge (LLM): strict JSON verdicts `{is_gbv, severity, reason, evidence}`
- Pipeline wrapper and Streamlit dashboard

## Tech Stack
- Orchestration: simple pipeline (LangGraph-like pattern)
- Vector DB: Qdrant (Docker)
- Embeddings: all-MiniLM-L6-v2 (text), CLIP ViT-B/32 (image)
- LLM: OpenAI gpt-4o-mini (swap to Groq/Llama-3 by editing `src/agents/judge.py`)
- UI: Streamlit

## Setup
1. Start Qdrant locally:
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```
2. Create `.env` from `.env.example` and set your keys.
3. Install Tesseract OCR binary (required for image text extraction):
   - macOS: `brew install tesseract`
   - Ubuntu: `sudo apt-get install tesseract-ocr`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Seed demo data:
   ```bash
   python scripts/seed_qdrant.py
   ```
6. Run the UI:
   ```bash
   streamlit run app.py
   ```

## Usage
Upload text, image, and optional audio. The dashboard displays:
- Analysis steps
- Qdrant context snapshots (history, imagery similarity, KB hits)
- Final JSON verdict

## Notes & Next Steps
- Add curated safe/toxic image sets in `./data/toxic` to improve similarity checks.
- Tune thresholds (e.g., consider GBV if top-3 image match > 0.8, or prior flags > 2 plus textual evidence).
- Expand `gbv_knowledge_base` with jurisdiction-specific guidance and evolving slang.
- Consider sparse-dense hybrid search (BM25/SPLADE) for obfuscated text.
- Add rate-limiting, audit logs, and pseudonymous user IDs for privacy.