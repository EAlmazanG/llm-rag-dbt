from langchain.vectorstores import Chroma
from sklearn.metrics.pairwise import cosine_similarity

class EnhancedRetriever:
    def __init__(self, vectorstore, embedding_function, max_tokens=2048):
        self.vectorstore = vectorstore
        self.embedding_function = embedding_function
        self.max_tokens = max_tokens

    def dynamic_filter(self, query):
        """
        Apply dynamic filters based on the query content.
        """
        filter_criteria = None
        query_lower = query.lower()

        if "source" in query_lower or "staging" in query_lower:
            filter_criteria = {"model_category": "stg", "is_source_model": True}
        elif "business" in query_lower or "output" in query_lower:
            filter_criteria = {"is_end_model": True}
        elif "test" in query_lower:
            filter_criteria = {"has_tests": True}
        elif "macro" in query_lower:
            filter_criteria = {"is_macro": True}
        elif "dependencies" in query_lower or "upstream" in query_lower:
            filter_criteria = {"parent_count": {"$gt": 0}}  # Valida padres según conteo
        elif "downstream" in query_lower or "children" in query_lower:
            filter_criteria = {"child_count": {"$gt": 0}}  # Valida hijos según conteo

        if "sql" in query_lower or "code" in query_lower:
            filter_criteria = {"knowledge_type": "code"}
        elif "description" in query_lower or "explain" in query_lower:
            filter_criteria = {"knowledge_type": "description"}

        return filter_criteria

    def expand_context(self, primary_docs, k=3):
        """
        Expand the context by retrieving related documents for each primary document.
        """
        expanded_documents = {doc.page_content: doc for doc in primary_docs}  # Use page_content as a unique key
        for doc in primary_docs:
            related_docs = self.vectorstore.similarity_search(doc.page_content, k=k)
            for related_doc in related_docs:
                # Add to the dictionary to avoid duplicates
                expanded_documents[related_doc.page_content] = related_doc

        # Return the unique documents as a list
        return list(expanded_documents.values())

    def combine_chunks(self, documents):
        combined_content = ""
        for doc in documents:
            if len(combined_content.split()) + len(doc.page_content.split()) <= self.max_tokens:
                combined_content += f"\n{doc.page_content}"
            else:
                break
        return combined_content

    def rank_documents(self, documents, query_embedding):
        """
        Rank documents based on their similarity to the query embedding.
        """
        ranked_docs = sorted(
            documents,
            key=lambda doc: cosine_similarity(
                [query_embedding],
                [self.embedding_function.embed_documents([doc.page_content])[0]]
            )[0][0],
            reverse=True,
        )
        return ranked_docs

    def retrieve(self, query, top_k=5, expansion_depth=3):
        """
        Retrieve the best documents based on the query, applying dynamic filters, expansion, and ranking.
        """
        # Apply dynamic filtering based on query content
        filter_criteria = self.dynamic_filter(query)

        # Retrieve initial documents
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k, "filter": filter_criteria},
        )
        primary_docs = retriever.invoke(query)

        # Expand context by retrieving related documents
        expanded_docs = self.expand_context(primary_docs, k=expansion_depth)

        # Rank documents based on query similarity
        query_embedding = self.embedding_function.embed_query(query)  # Fixed line
        ranked_docs = self.rank_documents(expanded_docs, query_embedding)

        # Combine top-ranked chunks for the final context
        final_context = self.combine_chunks(ranked_docs[:top_k])

        return final_context, ranked_docs[:top_k]
