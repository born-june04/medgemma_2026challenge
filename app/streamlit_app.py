import json
import tempfile
from pathlib import Path

import streamlit as st

from pipeline import load_config, run_pipeline_modes


st.set_page_config(page_title="Healthcare Demo Pipeline", layout="wide")

st.title("Healthcare Demo Pipeline")
st.markdown(
    "This demo is **not** diagnostic. It surfaces signal-level evidence and transparent physiology proxies."
)

config = load_config()

col_left, col_right = st.columns(2)
with col_left:
    audio_file = st.file_uploader("Cough/Breath audio (.wav)", type=["wav"])
with col_right:
    image_file = st.file_uploader("Chest X-ray (.png/.jpg)", type=["png", "jpg", "jpeg"])

intake_text = st.text_area("Optional intake text", placeholder="Short clinical context...")

if st.button("Run Pipeline"):
    if not audio_file or not image_file:
        st.error("Please provide both audio and image files.")
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / audio_file.name
            image_path = Path(tmpdir) / image_file.name
            audio_path.write_bytes(audio_file.read())
            image_path.write_bytes(image_file.read())

            outputs = run_pipeline_modes(
                audio_path=str(audio_path),
                image_path=str(image_path),
                intake_text=intake_text,
                config=config,
            )

        st.subheader("Ablation Modes")
        cols = st.columns(3)
        for idx, mode in enumerate(["llm_only", "signals", "signals+physiology"]):
            with cols[idx]:
                st.markdown(f"### {mode}")
                st.json(outputs[mode]["llm_output"])

        with st.expander("Structured Inputs"):
            st.json({mode: outputs[mode]["input"] for mode in outputs})

        with st.expander("Encoder Metadata"):
            st.json({mode: outputs[mode]["encoder_metadata"] for mode in outputs})
