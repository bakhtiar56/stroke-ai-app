import os
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.title("Stroke AI (Demo)")
st.write("API base URL:", API_BASE_URL)
st.info("UI placeholder. Next milestone: collect inputs + call POST /predict.")