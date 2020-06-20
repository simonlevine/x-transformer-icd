from pathlib import Path
import streamlit as st

from app.lib.load_model import get_kiss_model
model = get_kiss_model()

ASSETS_DIR = Path(__file__).parent.parent/"assets"

with (ASSETS_DIR/"blurb.md").open() as f:
    st.markdown(f.read())

free_text_user = st.text_area("Free Text Doctors Note")
if free_text_user:
    st.write("ICD Predictions:", model(free_text_user))