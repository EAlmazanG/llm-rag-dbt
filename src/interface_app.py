import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import os
import sys
import importlib

sys.path.append(os.path.abspath(os.path.join('..')))
@st.cache_data

def init_session():
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

def render_sidebar():
    st.sidebar.title("Configuración")
    
    repo_option = st.sidebar.selectbox(
        "Select dbt project repository",
        ["Local", "Online", "Already used"]
    )
    
    if repo_option == "Already used":
        processed_file = st.sidebar.file_uploader("Seleccionar archivo procesado")
    
    llm_option = st.sidebar.selectbox(
        "LLM run: ",
        ["Local", "Online"]
    )
    
    if llm_option == "Local":
        model_option = st.sidebar.selectbox(
            "Available models",
            ["model-1", "model-2"]
        )

def render_chat():
    st.title("dbt agents flow")
    for msg in st.session_state.conversation:
        if msg["role"] == "user":
            st.markdown(f"**User**: {msg['content']}")
        else:
            st.markdown(f"**dbt agent**: {msg['content']}")

    user_input = st.text_input("Write your request", "")
    if st.button("Request"):
        st.session_state.conversation.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            # Aquí vendría tu lógica de CrewAI o LLM
            response_text = "Response (placeholder)"
            st.session_state.conversation.append({"role": "assistant", "content": response_text})

def run_app():
    init_session()
    render_sidebar()
    render_chat()

if __name__ == "__main__":
    run_app()