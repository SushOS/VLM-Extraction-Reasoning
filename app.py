from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st
import yaml

from extract import MODEL_CONFIGS, VLMExtractor
from pdf_to_image import load_document_pages, preprocess_image


st.set_page_config(page_title="VLM PDF Extraction Demo", layout="wide")
st.title("VLM PDF Extraction Demo")
st.caption("Use CORD receipts for receipt extraction tests and PDF samples for end-to-end PDF ingestion demos.")

config = yaml.safe_load(Path("config.yaml").read_text())

model_name = st.selectbox("Model", list(MODEL_CONFIGS))
task_name = st.selectbox(
    "Task",
    ["generic_document", "key_value_pairs", "signature_check", "form_fields", "receipt_summary", "cord_receipt"],
)
uploaded = st.file_uploader("Upload a PDF or image", type=["pdf", "png", "jpg", "jpeg"])

if uploaded:
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded.getbuffer())
        temp_path = Path(handle.name)

    pages = [
        preprocess_image(page, **config["preprocessing"])
        for page in load_document_pages(temp_path, dpi=config["dpi"])
    ]
    cols = st.columns(len(pages))
    for index, page in enumerate(pages):
        cols[index].image(page, caption=f"Page {index + 1}", use_container_width=True)

    if st.button("Run Extraction"):
        extractor = VLMExtractor(model_name=model_name, max_new_tokens=config["max_new_tokens"])
        results = []
        with tempfile.TemporaryDirectory() as temp_dir:
            for index, page in enumerate(pages, start=1):
                page_path = Path(temp_dir) / f"page_{index}.png"
                page.save(page_path)
                results.append(extractor.extract_page(page_path, task_name=task_name))
        st.subheader("JSON Output")
        st.code(json.dumps(results, indent=2), language="json")
