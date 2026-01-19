import streamlit as st
from src.graph.pipeline import run_pipeline

st.set_page_config(page_title="SentinELLE Moderator", layout="centered")
st.title("SentinELLE: Multimodal GBV Moderator (MVP)")

with st.form("upload"):
    user_id = st.text_input("User ID", value="user123", help="Pseudonymous ID recommended.")
    text = st.text_area("Post Text (optional)")
    image_file = st.file_uploader("Image (optional)", type=["png", "jpg", "jpeg"])
    audio_file = st.file_uploader("Audio (optional)", type=["mp3", "wav", "m4a"])
    submitted = st.form_submit_button("Analyze")

if submitted:
    image_bytes = image_file.read() if image_file else None
    audio_bytes = audio_file.read() if audio_file else None

    with st.spinner("Analyzing..."):
        result = run_pipeline(
            user_id=user_id,
            text=text,
            image_bytes=image_bytes,
            audio_bytes=audio_bytes
        )

    st.subheader("Process")
    for step in result["steps"]:
        st.write(f"- {step}")

    st.subheader("Context Snapshots")
    hist = result["archivist"]["history"]
    st.write(f"User flags: {hist['flags']} | Avg toxicity score: {hist['toxicity_score']}")
    st.write("Image similarity (top matches):")
    for m in result["archivist"]["image_similarity"]["matches"][:3]:
        st.write(f"- score={m['score']:.3f}, label={m['label']}, caption={m['caption']}")

    st.write("KB hits (top):")
    for h in result["archivist"]["text_context"][:3]:
        st.write(f"- score={h['score']:.3f} | {h['text']}")

    st.subheader("Final Verdict")
    st.json(result["verdict"])
