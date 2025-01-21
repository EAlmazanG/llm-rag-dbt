import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import os
import sys
import importlib
from streamlit_extras.file_browser import file_browser

sys.path.append(os.path.abspath(os.path.join('..')))

def render_sidebar():
    st.sidebar.title("⚙️ Config your dbt project repo and LLM")
    
    st.sidebar.markdown("---")
    
    repo_option = st.sidebar.selectbox(
        "Select dbt project repository",
        ["Local", "Online", "Already used"]
    )
    
    if repo_option == "Already used":
        knowledge_file = st.sidebar.file_uploader("Select processed repo file")
        is_online = False
        print(knowledge_file)

    elif repo_option == "Local":
        repo_path = file_browser("Select local repo folder")
        is_online = False
        print(repo_path)

    elif repo_option == "Online":
        repo_path = st.sidebar.text_input("Enter repo URL")
        is_online = False
        print(repo_path)

    st.sidebar.markdown("---")
    
    llm_option = st.sidebar.selectbox(
        "LLM run: ",
        ["Local LLM with LM Studio", "OpenAI"]
    )
    
    if llm_option == "Local":
        model_option = st.sidebar.selectbox(
            "Available models",
            ["model-1", "model-2"]
        )

def render_chat():
    st.markdown(
        """
        <style>
            body {
                font-family: 'Arial', sans-serif;
            }
            .chat-container {
                padding: 20px;
                border-radius: 10px;
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                margin-top: 20px;
            }
            .user-msg, .assistant-msg {
                display: flex;
                align-items: flex-start;
                gap: 10px;
                padding: 12px;
                border-radius: 10px;
                margin-bottom: 15px;
                max-width: 75%;
                font-size: 16px;
            }
            .user-msg {
                background-color: #e1f5fe;
                color: #333;
                text-align: right;
                margin-left: auto;
                flex-direction: row-reverse;
            }
            .assistant-msg {
                background-color: #eceff1;
                color: #333;
                text-align: left;
                margin-right: auto;
            }
            .user-msg strong, .assistant-msg strong {
                font-size: 18px;
                color: #004d40;
            }
            .avatar {
                width: 40px;
                height: 40px;
                border-radius: 50%;
            }
            .input-container {
                margin-top: 30px;
            }
            .stButton>button {
                width: 100%;
                border-radius: 8px;
                background-color: #00695c;
                color: white;
                border: none;
                padding: 12px;
                font-size: 16px;
            }
            .stButton>button:hover {
                background-color: #004d40;
            }
            .header-title {
                font-size: 32px;
                font-weight: bold;
                color: #004d40;
                margin-bottom: 10px;
            }
            .subheader {
                font-size: 20px;
                color: #555;
                margin-bottom: 20px;
            }
            .divider {
                border-top: 1px solid #ddd;
                margin: 20px 0;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="header-title">💬 dbt Agent Chat</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Ask about your dbt project or get help with model changes! </div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.conversation:
        if msg["role"] == "user":
            st.markdown(
                f'''
                <div class="user-msg">
                    <img src="https://cdn-icons-png.flaticon.com/512/847/847969.png" class="avatar">
                    <div><strong>You:</strong><br>{msg["content"]}</div>
                </div>
                ''',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'''
                <div class="assistant-msg">
                    <img src="https://images.seeklogo.com/logo-png/43/1/dbt-logo-png_seeklogo-431112.png?v=1957906038962209040" class="avatar">
                    <div><strong>dbt Agent:</strong><br>{msg["content"]}</div>
                </div>
                ''',
                unsafe_allow_html=True
            )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-size: 16px; color: #6c757d; margin-bottom: 5px;">
            Enter your request:
        </div>
        """,
        unsafe_allow_html=True
    )
    user_input = st.text_input(
        "",
        key="user_input",
        placeholder="Type your message here...",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([0.75, 0.25])
    with col1:
        if st.button("Send"):
            handle_submit()

    with col2:
        if st.button("🗑 Clear Chat"):
            st.session_state.conversation = []
            st.session_state.user_input_key = ""
            st.rerun()

def handle_submit():
    user_input = st.session_state.user_input
    if user_input.strip():
        st.session_state.conversation.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            response_text = "Response (placeholder)"
            st.session_state.conversation.append({"role": "assistant", "content": response_text})

        st.session_state.user_input_key = ""
        st.rerun()

def init_session():
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'user_input_key' not in st.session_state:
        st.session_state.user_input_key = ""

def run_app():
    init_session()
    render_sidebar()
    render_chat()

if __name__ == "__main__":
    run_app()