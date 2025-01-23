import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import os
import sys
import importlib
import yaml
import sqlparse
import requests
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

sys.path.append(os.path.abspath(os.path.join('..')))
from src import generate_knowledge
from src import create_rag_db
from src import llm_chain_tools
from src.enhanced_retriever import EnhancedRetriever

import importlib
import src.generate_knowledge
importlib.reload(src.generate_knowledge)

generate_knowledge.add_repo_root_path()
import openai_setup

OPENAI_API_KEY = openai_setup.conf['key']
OPENAI_PROJECT = openai_setup.conf['project']
OPENAI_ORGANIZATION = openai_setup.conf['organization']
DEFAULT_LLM_MODEL = "gpt-4o-mini"

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ['OPENAI_MODEL_NAME'] = DEFAULT_LLM_MODEL


def create_chromadb(dbt_repo_knowledge_df, repo_path):
    _, repo_name = generate_knowledge.extract_owner_and_repo(repo_path)
    
    CHROMADB_DIRECTORY = '../chromadb'
    COLLECTION_NAME = repo_name

    dbt_repo_knowledge_df['contextual_info'] = dbt_repo_knowledge_df.apply(create_rag_db.combine_contextual_fields, axis=1)
    documents = create_rag_db.create_documents_from_df(dbt_repo_knowledge_df)
    langchain_openai_embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")

    documents_cleaned = create_rag_db.clean_metadata(documents)
    documents_chunked = create_rag_db.chunk_documents(documents_cleaned, chunk_size=500, chunk_overlap=100)
    create_rag_db.save_vectorstore_to_chroma(documents_chunked, langchain_openai_embeddings)
    print("chromadb for " + repo_name + " successfully created!", CHROMADB_DIRECTORY, COLLECTION_NAME)

    return True

def load_chroma_db(repo_name):
    CHROMADB_DIRECTORY = '../chromadb'
    COLLECTION_NAME = repo_name

    langchain_openai_embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
    loaded_vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMADB_DIRECTORY,
        embedding_function=langchain_openai_embeddings
    )

    files = {
        'agents': '../config/agents.yml',
        'tasks': '../config/tasks.yml'
    }

    return loaded_vectorstore, files

def load_repo(repo_option, uploaded_file = None, repo_path = None):

    if repo_option == "Already used":
        if uploaded_file is not None:
            dbt_models_df = pd.read_csv(uploaded_file)
            file_name = uploaded_file.name
            match = re.match(r"dbt_models_(.+)\.csv", file_name)
            if match:
                repo_name = match.group(1)
                dbt_project_df = pd.read_csv('../data/dbt_project_' + str(repo_name) + '.csv')
                dbt_repo_knowledge_df = create_rag_db.merge_dbt_models_and_project_dfs(dbt_models_df, dbt_project_df)
                loaded_vectorstore, files = load_chroma_db(repo_name)
                enable_chat = True

    elif repo_option == "Local":
        repo_path = st.sidebar.text_input("Enter repo folder path")
        repo_elements = generate_knowledge.list_local_repo_structure(repo_path)



    elif repo_option == "Online":
        owner, repo_name = generate_knowledge.extract_owner_and_repo(repo_path)
        repo_elements = generate_knowledge.list_online_repo_structure(owner, repo_name)
        print(repo_elements)
        print("\n")
        dbt_models_enriched_df, dbt_project_df = generate_knowledge.generate_knowledge_from_repo_elements(repo_elements, True, repo_path)
        print("save models and project knowledge from " + repo_path)

        dbt_repo_knowledge_df = create_rag_db.merge_dbt_models_and_project_dfs(dbt_models_enriched_df, dbt_project_df)
        files = {
            'agents': '../config/agents.yml',
            'tasks': '../config/tasks.yml'
        }
        is_db_created = create_chromadb(dbt_repo_knowledge_df, repo_path)
        if is_db_created:
            loaded_vectorstore, files = load_chroma_db(repo_name)
            enable_chat = True

    return enable_chat, dbt_repo_knowledge_df, loaded_vectorstore, files


def render_sidebar():
    enable_chat = False
    st.sidebar.title("⚙️ Config your dbt project repo and LLM")
    
    st.sidebar.markdown("---")
    
    repo_option = st.sidebar.selectbox(
        "Select dbt project repository",
        ["Local", "Online", "Already used"]
    )

    if repo_option == "Already used":
        uploaded_file = st.sidebar.file_uploader("Select processed models file")
        repo_path = None
    elif repo_option == "Local":
        uploaded_file = None
        repo_path = st.sidebar.text_input("Enter repo folder path")
    elif repo_option == "Online":
        uploaded_file = None
        repo_path = st.sidebar.text_input("Enter repo URL", 'https://github.com/dbt-labs/jaffle-shop')

    if st.sidebar.button("Load Repo") and (repo_path is not None or uploaded_file is not None):
        with st.spinner("Loading repository..."):
            enable_chat = load_repo(repo_option, uploaded_file, repo_path)

    st.sidebar.markdown("---")
    
    llm_option = st.sidebar.selectbox(
        "LLM run: ",
        ["Local LLM with LM Studio", "OpenAI"]
    )
    
    if llm_option == "OpenAI":



        langchain_openai_embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
        langchain_openai_llm = ChatOpenAI(model=DEFAULT_LLM_MODEL, temperature=0.1, openai_api_key=OPENAI_API_KEY, openai_organization = OPENAI_ORGANIZATION)

    else:
        model_option = st.sidebar.selectbox(
            "Available models",
            ["model-1", "model-2"]
        )
    return enable_chat


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
    enable_chat, dbt_repo_knowledge_df, loaded_vectorstore, files = render_sidebar()
    if enable_chat:
        render_chat()
    else:
        st.title("Please select the dbt project repo and LLM config to start")

if __name__ == "__main__":
    run_app()