from pathlib import Path
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import time

from app.lib.models import KissModel


model = KissModel()

ROOT_DIR = Path(__file__).parent.parent.parent
APP_DIR = ROOT_DIR/"app"
ASSETS_DIR = APP_DIR/"assets"
NOTES2DIAGNOSIS_ICD_TRAIN_FP = ROOT_DIR/"intermediary-data"/"notes2diagnosis-icd-train.json.gz"

@st.cache
def load_assets():
    dataset_test = pd.read_json(NOTES2DIAGNOSIS_ICD_TRAIN_FP, orient="split")
    text_samples_test = dataset_test.head(2).TEXT.tolist()
    return {
        "text_samples_test": text_samples_test
    } 
assets = load_assets()

st.markdown("""# autoICD
Developed By Jeremy Fisher and Simon Levine""")

choices = assets["text_samples_test"] + ["Input your own"]
text_sample = st.selectbox("Choose example discharge summary or input your own. We'll predict the ICD code!", choices)
if text_sample == "Input your own":
    text_sample = st.text_area("")

if text_sample:
    predicted_codes_proba = model(text_sample)
    fig = alt.Chart(
        pd.DataFrame({
            "code_name": model.icd9_codes,
            "code_proba": predicted_codes_proba
        })
    ).mark_bar().encode(
        x=alt.X('code_name', axis=alt.Axis(title='ICD9 Code')),
        y=alt.Y('code_proba',
            scale=alt.Scale(domain=(0, 1)),
            axis=alt.Axis(format='%', title='Probability')
        ),
        tooltip=['code_proba']
    ).interactive()
    st.write("ICD Predictions:", fig)