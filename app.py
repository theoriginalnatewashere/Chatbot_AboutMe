import streamlit as st
from groq import Groq
from datetime import datetime
from streamlit_pills import pills

MODEL_NAME = "llama-3.3-70b-versatile"

st.set_page_config(
    page_title="DataPod Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load only the AI assistant page for managing personal data pods
with open("_pages/home.py", encoding="utf-8") as file:
    exec(file.read())
