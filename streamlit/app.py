## ex) streamlit run app.py --server.port=30001 -- --submission_csv <path> --gt_json <path> --dataset_path <path>

import streamlit as st

st.set_page_config(page_title="cv12 data production visualizer", layout="wide")

st.title("CV12 data production")
st.markdown(
    """
    <--- Pages
    1. annotation \n
    2. OCR text
"""
)
