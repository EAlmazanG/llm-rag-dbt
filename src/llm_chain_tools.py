import ast
from langchain.schema import Document

def gather_upstream(lineage_df, model_name):
    visited = set()
    frontier = [model_name]
    while frontier:
        current = frontier.pop()
        row = lineage_df[lineage_df["model_name"] == current]
        if not row.empty:
            for source in row["source"].iloc[0]:
                if source not in visited:
                    visited.add(source)
                    frontier.append(source)
            for parent in row["parent_models"].iloc[0]:
                if parent not in visited:
                    visited.add(parent)
                    frontier.append(parent)
    return sorted(visited)

def gather_downstream(lineage_df, model_name):
    visited = set()
    frontier = [model_name]
    while frontier:
        current = frontier.pop()
        row = lineage_df[lineage_df["model_name"] == current]
        if not row.empty:
            for child in row["children_models"].iloc[0]:
                if child not in visited:
                    visited.add(child)
                    frontier.append(child)
    return sorted(visited)

def get_affected_models(lineage_df, model_name):
    up = gather_upstream(lineage_df, model_name)
    down = gather_downstream(lineage_df, model_name)
    return {
        "upstream": list(up),
        "downstream": list(down),
    }

def select_documents(documents, filtered_models):
    affected_dbt_models_documents = [
        doc for doc in documents
        if hasattr(doc, 'metadata') and doc.metadata.get("name") in filtered_models
    ]

    macro_names = set()
    for doc in affected_dbt_models_documents:
        macros = doc.metadata.get("macros", [])
        if isinstance(macros, str):
            macros = ast.literal_eval(macros)
        macro_names.update(f"{macro}.sql" for macro in macros if macro)

    unique_macros_list = list(macro_names)
    unique_macros_documents = [
        doc for doc in documents
        if hasattr(doc, 'metadata') and doc.metadata.get("name") in unique_macros_list
    ]

    retriever_documents = affected_dbt_models_documents + unique_macros_documents
    
    csv_sources_documents = [
        doc for doc in documents
        if hasattr(doc, 'metadata') and doc.metadata.get("knowledge_type") == "project" and doc.metadata.get("name")[-4:] == ".csv"
    ]

    yml_project_documents = [
        doc for doc in documents
        if hasattr(doc, 'metadata') and doc.metadata.get("knowledge_type") == "project" and doc.metadata.get("name")[-4:] == ".yml"
    ]

    return {
        "retriever_documents": retriever_documents,
        "csv_sources_documents": csv_sources_documents,
        "yml_project_documents": yml_project_documents
    }

def extract_documents_from_vectorstore(vectorstore):
    vectorstore_documents = vectorstore.get()
    documents_formatted = [
        Document(page_content=content, metadata=metadata)
        for content, metadata in zip(vectorstore_documents["documents"], vectorstore_documents["metadatas"])
    ]
    return documents_formatted