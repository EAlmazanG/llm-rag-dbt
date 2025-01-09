import pandas as pd
import re
import yaml
import sqlparse
import os
import pandas as pd
import numpy as np
import requests
import networkx as nx
import matplotlib.pyplot as plt
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def merge_dbt_models_and_project_dfs(dbt_models_df, dbt_project_df):
    dbt_models_df['knowledge_type'] = 'models'
    dbt_project_df['knowledge_type'] = 'project'

    dbt_models_df.rename(columns = {'sql_code':'code'}, inplace = True)

    all_columns = set(dbt_models_df.columns).union(set(dbt_project_df.columns))

    for col in all_columns:
        if col not in dbt_models_df:
            dbt_models_df[col] = None
        if col not in dbt_project_df:
            dbt_project_df[col] = None

    merged_df = pd.concat([dbt_models_df, dbt_project_df], ignore_index=True)
    columns_order = ['knowledge_type'] + [col for col in merged_df.columns if col != 'knowledge_type']
    return merged_df[columns_order]

def combine_contextual_fields(row):
    combined = f"""
        <MODEL CODE>:
        .sql Code:
        {row['code'] if pd.notna(row['code']) else 'N/A'}

        .yml Code:
        {row['yml_code'] if pd.notna(row['yml_code']) else 'N/A'}

        <MODEL INFO>
        Primary Key:
        {row['primary_key'] if pd.notna(row['primary_key']) else 'N/A'}

        IDS:
        {row['sql_ids'] if pd.notna(row['sql_ids']) else 'N/A'}

        Columns used to Filter the model throuhg JOINS, HAVING, WHERE...:
        {row['filters'] if pd.notna(row['filters']) else 'N/A'}

        Tests:
        {row['tests'] if pd.notna(row['tests']) else 'N/A'}

        Description for project files:
        {row['description'] if pd.notna(row['description']) else 'N/A'}

        dbt Model description:
        {row['model_description'] if pd.notna(row['model_description']) else 'N/A'}

        Jinja inside the dbt model description:
        {row['jinja_description'] if pd.notna(row['jinja_description']) else 'N/A'}

        <MODEL DEPENDENCIES>:
        Downstream models:
        {row['children_models'] if pd.notna(row['children_models']) else 'N/A'}

        Upstream models:
        {row['parent_models'] if pd.notna(row['parent_models']) else 'N/A'}

        {'The model is in the first layer of the dbt model, connected directly to the sources: ' + row['source'] if row.get('is_source_model') else ''}
        {'Also, The model is in the last layer, so is consider as a business output' if row['is_end_model'] else ''}
            
    """
    return combined.strip()

def clean_metadata(documents):
    cleaned_documents = []
    for doc in documents:
        cleaned_metadata = {
            key: ("" if value is None else value) if isinstance(value, (str, int, float, bool, type(None))) else str(value)
            for key, value in doc.metadata.items()
        }
        cleaned_documents.append(Document(page_content=doc.page_content, metadata=cleaned_metadata))
    return cleaned_documents

def chunk_documents(documents, chunk_size=500, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunked_documents = []
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        for chunk in chunks:
            chunked_documents.append(Document(page_content=chunk, metadata=doc.metadata))
    return chunked_documents

def save_vectorstore_to_chroma(documents, embeddings, persist_directory, collection_name):
    vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory, collection_name=collection_name)
    print(f"Vectorstore saved to {persist_directory}")
    return persist_directory

def create_documents_from_df(df):
    documents = df.apply(
        lambda row: Document(
            page_content=row["contextual_info"],
            metadata={
                "knowledge_type": row["knowledge_type"],
                "name": row["name"],
                "path": row["path"],
                "source": row["source"],
                "parents": row["parent_models"],
                "children": row["children_models"],
                "config": row["config"],
                "materialized": row["materialized"],
                "is_snapshot": row["is_snapshot"],
                "model_category": row["model_category"],
                "vertical": row["vertical"],
                "has_tests": row["has_tests"],
                "has_select_all_in_last_select": row["has_select_all_in_last_select"],
                "has_group_by": row["has_group_by"],
                "is_filtered": row["is_filtered"],
                "is_source_model": row["is_source_model"],
                "is_seed": row["is_seed"],
                "is_end_model": row["is_end_model"],
                "is_macro": row["is_macro"],
                "is_test": row["is_test"],
                "macros": row["macros"],
                "packages": row["packages"]
            }
        ), axis=1
    ).tolist()
    return documents

def plot_dbt_lineage(dbt_repo_knowledge_df):
    # Generate lineage df
    lineage_df = dbt_repo_knowledge_df[(dbt_repo_knowledge_df['knowledge_type'] == 'models') & (dbt_repo_knowledge_df['extension'] == '.sql')][['name','parent_models','children_models','source']]
    lineage_df['model_name'] = lineage_df['name'].apply(lambda x: x[:-4])
    lineage_df = lineage_df.drop(columns=['name'])[['model_name','source','parent_models','children_models']]

    # Ensure valid lists in columns
    lineage_df["source"] = lineage_df["source"].apply(lambda x: x if isinstance(x, list) else eval(x) if isinstance(x, str) and x.startswith('[') else [])
    lineage_df["parent_models"] = lineage_df["parent_models"].apply(lambda x: x if isinstance(x, list) else eval(x) if isinstance(x, str) and x.startswith('[') else [])
    lineage_df["children_models"] = lineage_df["children_models"].apply(lambda x: x if isinstance(x, list) else eval(x) if isinstance(x, str) and x.startswith('[') else [])

    # Create directed graph
    G = nx.DiGraph()

    # Add source nodes
    all_sources = lineage_df["source"].sum()
    unique_sources = list(set(all_sources))
    G.add_nodes_from(unique_sources, layer=0)

    # Add nodes and edges for models
    for _, row in lineage_df.iterrows():
        layer = 1 if row["source"] and not row["parent_models"] else 2 if row["source"] and row["parent_models"] else 3 if not row["source"] and row["parent_models"] and row["children_models"] else 4 if not row["children_models"] and row["parent_models"] else None
        if layer:
            G.add_node(row["model_name"], layer=layer)
            for source in row["source"]:
                G.add_edge(source, row["model_name"])
            for parent in row["parent_models"]:
                G.add_edge(parent, row["model_name"])
            for child in row["children_models"]:
                G.add_edge(row["model_name"], child)

    # Assign colors based on model type
    def get_color(node):
        if node in unique_sources:
            return "lightgreen"
        elif G.out_degree(node) == 0:
            return "lightcoral"
        elif node.startswith("stg"):
            return "lightblue"
        elif node.startswith("base"):
            return "orange"
        elif node.startswith("int"):
            return "pink"
        return "gray"

    node_colors = [get_color(n) for n in G.nodes]

    # Layout to minimize edge crossings
    pos = nx.multipartite_layout(G, subset_key="layer")

    # Draw graph with rectangular nodes
    plt.figure(figsize=(16, 10))
    nx.draw(
        G, pos, with_labels=True, node_size=3000, font_size=10, font_weight="bold",
        arrowsize=20, node_color=node_colors, edgecolors="black",
    )
    ax = plt.gca()
    for node, (x, y) in pos.items():
        ax.text(x, y, node, fontsize=10, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor=get_color(node)))

    plt.title("DBT Models Lineage", fontsize=16)
    plt.show()
    return lineage_df