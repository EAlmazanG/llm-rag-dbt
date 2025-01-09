from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
from langchain.output_parsers import PydanticOutputParser
from enhanced_retriever import EnhancedRetriever
from langchain.vectorstores import Chroma

import llm_chain_tools
import create_rag_db

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class Step:
    """Base class for a step in the processing pipeline."""
    def __init__(self, name):
        self.name = name

    def execute(self, input_data, context):
        raise NotImplementedError("Each step must implement the execute method.")

class InterpretationStep(Step):
    """Evaluate the user's request and decide the type of action needed."""
    def __init__(self, llm):
        super().__init__("Interpretation")
        self.llm = llm
        self.prompt_template = PromptTemplate(
            input_variables=["request"],
            template=("""
                <<>ROLE>>>
                You are an expert in interpreting user requests and translating them into clear and concise decisions and actions, based on your deep expertise as an Analytics Engineer, knowledge of dbt and in interpreting and translating problems into solutions.      

                <<<TASK>>>      
                Evaluate the request: {request}
                --------------------
                and based on the evaluation, determine the required action:
                1) adding a field -> return ADD_COLUMN
                2) modifying an existing model -> return MODIFY_MODEL
                3) retrieving and returning specific information. -> return RETRIVE_INFO
                
                Reflect on the request and provide a concise plan for the approach:
                - If the action involves adding a field: Identify where the field is currently available, if provided.
                - Determine how to propagate the field through the necessary models or transformations to integrate it into the target model.
                - Consider the impact on related models and dependencies.
                - If the action involves modifying an existing model: Identify the specific changes required.
                - Assess how these changes affect the structure, relationships, and downstream dependencies of the model.
                - If the action involves retrieving or returning information: Identify the models containing the relevant data.
                - Analyze how these models are related, and determine the queries or transformations needed to extract the requested information.
                      
                <<<OUPUT>>>
                Return the required action from the three proposed, and a short explanaition of the required work.
                REMEMBER: Provide no extra commentary or explanation, just the minimal information required,  
                return only useful information, no additional filler text or unnecessary explanations, get to the point.
            """)   
        )

    def execute(self, input_data, context):
        print(f"Step: {self.name} | Input: {input_data}")
        prompt = self.prompt_template.format(request=input_data)
        interpretation = self.llm.invoke(prompt)
        context["interpretation"] = interpretation
        print(f"Step: {self.name} | Output: {interpretation}")
        return interpretation

class EvaluationStep(Step):
    """Evaluates on the user's request based on evaluation."""
    def __init__(self, llm):
        super().__init__("Evaluation")
        self.llm = llm
        self.prompt_template = PromptTemplate(
            input_variables=["request", "interpretation"],
            template=("""
                <<>ROLE>>>
                You are an expert in evaluating user requests and translating them into clear and concise decisions and actions, based on your deep expertise as an Analytics Engineer, knowledge of dbt and in interpreting and translating requests into solutions.      

                <<<TASK>>>      
                Based on the interpretation of an expert dbt and problem interpreter: {interpretation}
                --------------------
                of the original request
                ORIGINAL REQUEST: {request}
                --------------------
                Include only these topics if relevant:
                - Target models or files
                - Field existence
                - Documentation needs
                - Dependencies and relationships
                - Performance/design considerations
                - Tests
                - dbt project config
                - Code or logic generation
                
                Summarize only the necessary actions, no filler. Think about all the considerations and steps required to handle this request effectively.
                Your evaluation could include the following actions:
                - Identify the target model or models: Analyze the dependency tree to locate the model where the change should occur.
                - Understand its upstream sources (seeds, sources or base models)
                - Understand downstream dependencies of the model.
                - Documentation: Assess if it's going to be necessary to add adjustments to the documentation, and which would be this changes.
                - Unique keys and IDs: Examine the available unique keys and identifiers in the initial, intermediate, and final models.
                - Decide how these can be used to integrate the field or establish relationships between models.
                - Check if the granularity of the model can be altered through the changes added.
                - Evaluate performance and design: Review the data pipeline from start to finish.
                - Decide where the change or addition would be most optimal in terms of performance, data modeling, and maintainability.
                - Project state and impact: Consider the current state of the dbt project and how the changes might affect the broader model chain.
                - Macros and seeds: Check if relevant macros or seed data exist that can help transform or derive the required field or model.
                - Tests: Identify existing tests for the field or model and determine whether new tests need to be added or adjusted to validate the changes.
                - dbt project configuration: Review the general configuration (e.g., variables, environments, conventions) in the dbt_project.yml file to ensure the changes align with project standards and won't disrupt the schema.
                - Code generation: fragments of SQL logic needed, including CTEs or columns.
                - Evaluate whether an intermediate model is necessary or if the logic can be handled within the existing pipeline.
                - Documentation generation: Specify the documentation needed for any fields, models, or logic added or updated as part of this request. Only the things to add or change.
                
                <<<OUPUT>>>
                Provide a concise summary of the high-level tasks based on your analysis, with the reflection of each one of them to be prepared to completed in the next steps when context is provided. If you don't have the info to perform the action because it's necessary context of the project, code or the lineage of the models, don't answer it, in next steps context will be provide as input to answer it properly. Mark it as it's necessart the context to responde it, and only do the reflection part using your logic as dbt expert.
                No extra checks or steps that are not on this list, select only the needed actions for the request.
                Return only useful information, no additional filler text or unnecessary explanations, get to the point.
                      
                REMEMBER: Provide no extra commentary or explanation, just the minimal information required.
            """)
        )

    def execute(self, input_data, context):
        print(f"Step: {self.name} | Input: {input_data}")
        interpretation = context.get("interpretation", "")
        prompt = self.prompt_template.format(request=input_data, interpretation=interpretation)
        evaluation = self.llm.invoke(prompt)
        context["evaluation"] = evaluation
        print(f"Step: {self.name} | Output: {evaluation}")
        return evaluation

class LineageResponse(BaseModel):
    model: str
    scope: Literal['UP', 'DOWN', 'ALL']

class LineageStep(Step):
    """Identify which models or documents should be consulted or modified using dbt lineage."""
    def __init__(self, llm, vectorstore, embeddings, dbt_repo_knowledge_df):
        super().__init__("Lineage")
        self.llm = llm
        self.vectorstore = vectorstore
        self.embedding_function = embeddings
        self.dbt_repo_knowledge_df = dbt_repo_knowledge_df

        self.parser = PydanticOutputParser(pydantic_object=LineageResponse)
        self.prompt_template = PromptTemplate(
            input_variables=["evaluation", "retrieved_context"],
            template=("""
                <<>ROLE>>>
                You are an expert in identifying the models and changes needed in a dbt repo given a request, based on your deep expertise as an Analytics Engineer, knowledge of dbt and in interpreting and translating requests into solutions, and using context and context information from the repository code and other models.

                <<<TASK>>>      
                Based on this evaluation: {evaluation}
                --------------------
                and the retrieved context, remember the meaning of all the context data:
                CONTEXT INFO AND METADATA MEANING:
                - knowledge_type: Specifies whether the resource is about files within the dbt project configuration, or data models used in the pipeline.
                - name: Name of the file.
                - path: Path within the repository where the file or resource is located, relative to the project root.
                - source: The original dataset, from which the model or resource pulls its data. Only for the first layer of the model.
                - parents: dbt models  that serve as input dependencies for the current model or file.
                - children: dbt models that depend on this resource as an input for their logic or data processing.
                - config: Configuration parameters defined in the file, specifying behaviors, sort keys, materialization...
                - materialized: Indicates how the data model is materialized (e.g., as a table, view, or ephemeral model) in the pipeline.
                - is_snapshot: Boolean flag that identifies whether the file represents a snapshot dbt model.
                - model_category: The logical dbt categorization of the model inside the project, such as base, staging, intermediate, marts...
                - vertical: Business domain or vertical to which the resource belongs, such as e-commerce, finance, supply...
                - has_tests: Boolean flag indicating if there are tests associated with this resource in the model .yaml file or in the tests folder.
                - has_select_all_in_last_select: Specifies if the final SQL query in the file uses a SELECT * statement. So all columns of previous CTEs will be consider as output of the model.
                - has_group_by: Boolean flag indicating if the SQL code includes a GROUP BY clause for aggregating data.
                - is_filtered: Boolean flag that specifies whether the resource applies filters to its data with WHERE, HAVING, JOIN...
                - is_source_model: The model uses the source macro to extract the data from the dbt project sources. It belongs to the first layer of all dbt models.
                - is_seed: Specifies if the resource is a seed file.
                - is_end_model: Boolean flag identifying if this is a terminal model in the data pipeline, representing the final output.
                - is_macro: Boolean flag indicating if the file defines a macro, used for reusable logic across the project. It does not count macros ref to call other models or source to connect to sources.
                - is_test: Indicates whether the file is a test sql definition used for testing purposes of other dbt models.
                - macros: Name of the macros used in the dbt model if any. It does not count macros ref to call other models or source to connect to sources.
                - packages: List of external packages or dependencies required for the resource or project functionality. Only not None in the packages.yml	row.

                      
                RETRIEVED CONTEXT: {retrieved_context}
                --------------------
                Determine:
                - The primary dbt model directly affected (e.g., where a field is added, a modification is made, or information is requested).
                - Whether upstream models (UP), downstream models (DOWN), or both (ALL) are necessary to handle this request correctly.
                
                Consider the following cases:
                1. If a new field is added, specify the model where the field will be added and indicate UP for upstream models needed to populate the field.
                2. If an existing field is modified, specify the model where the change occurs and indicate DOWN for downstream models affected by the change.
                3. If information is requested, specify the model containing the requested information and indicate UP, DOWN, or ALL based on the context of the data needed.
                4. If a field or model is removed, specify the model being affected and indicate DOWN for downstream dependencies impacted.
                      
                <<<OUPUT>>>
                Return the following json format: {{'model':model name, 'scope':UP/DOWN/ALL}}
                REMEMBER: Provide no extra commentary or explanation, just the minimal information required.
            """)
        )

    def execute(self, input_data, context):
        print(f"Step: {self.name} | Input: {input_data}")
        evaluation = context.get("evaluation", "")
        
        # Adjusted retriever
        #retriever = self.vectorstore.as_retriever()
        #retrieved_documents = retriever.invoke(input_data)
        retriever = EnhancedRetriever(vectorstore = self.vectorstore, embedding_function= self.embedding_function)
        retrieved_context, retrieved_documents = retriever.retrieve(input_data)

        retrieved_context = "\n".join([doc.page_content for doc in retrieved_documents if hasattr(doc, 'page_content')])
        context["documents_for_lineage"] = retrieved_documents
        print(f"Step: {self.name} | Retrieved Context: {retrieved_context}")

        prompt = self.prompt_template.format(evaluation=evaluation, retrieved_context=retrieved_context)

        response = self.llm.invoke(prompt)
        lineage_analysis_raw = response.dict()
        content = lineage_analysis_raw.get("content", "")

        json_string = content.replace("```json", "").replace("```", "").strip()
        lineage_analysis = eval(json_string)  # Use eval cautiously; use json.loads if the string is valid JSON
        
        context["lineage_analysis"] = lineage_analysis
        print(f"Step: {self.name} | Output: {lineage_analysis}")

        # ADD DOC FILTERING
        model_name = lineage_analysis.get("model")
        scope = lineage_analysis.get("scope", "").upper()

        lineage_df = create_rag_db.plot_dbt_lineage(self.dbt_repo_knowledge_df)
        affected_models = llm_chain_tools.get_affected_models(lineage_df, model_name)
        if scope == "UP":
            filtered_models = affected_models["upstream"]
        elif scope == "DOWN":
            filtered_models = affected_models["downstream"]
        elif scope == "ALL":
            filtered_models = affected_models["upstream"] + affected_models["downstream"]

        filtered_models = list(set(f"{model}.sql" for model in filtered_models + [model_name]))
        
        documents = llm_chain_tools.extract_documents_from_vectorstore(self.vectorstore)
        selected_documents = llm_chain_tools.select_documents(documents, filtered_models)
        context["selected_documents"] = selected_documents
        print(f"Step: {self.name} | Filtered Models: {filtered_models} | Filtered Documents: {len(selected_documents)}")
        return response

class PlanStep(Step):
    """Plan the steps needed based on evaluation and retrieved context."""
    def __init__(self, llm, embedding_function):
        super().__init__("Plan")
        self.llm = llm
        self.embedding_function = embedding_function

        self.prompt_template = PromptTemplate(
            input_variables=["evaluation", "retrieved_context", "lineage_analysis","retrieved_csv_sources_context"],
            template=("""
                <<>ROLE>>>
                You are an expert in planning and deciding the necessary changes in the affected models of a dbt project given a request, based on your deep expertise as an Analytics Engineer, knowledge of dbt and in interpreting and translating requests into solutions, and using context and context information from the repository code and other models.      

                <<<TASK>>>      
                Based on this evaluation: {evaluation}
                The analysis of the most impacted model and the part of the lineage that is affected or has related info 
                (UP = Upstream models(parent models)/DOWN = Downstream models(childen models)/ALL = both): 
                {lineage_analysis}
                --------------------
                the retrieved context, remember the meaning of all the context data:
                CONTEXT INFO AND METADATA MEANING:
                - knowledge_type: Specifies whether the resource is about files within the dbt project configuration, or data models used in the pipeline.
                - name: Name of the file.
                - path: Path within the repository where the file or resource is located, relative to the project root.
                - source: The original dataset, from which the model or resource pulls its data. Only for the first layer of the model.
                - parents: dbt models  that serve as input dependencies for the current model or file.
                - children: dbt models that depend on this resource as an input for their logic or data processing.
                - config: Configuration parameters defined in the file, specifying behaviors, sort keys, materialization...
                - materialized: Indicates how the data model is materialized (e.g., as a table, view, or ephemeral model) in the pipeline.
                - is_snapshot: Boolean flag that identifies whether the file represents a snapshot dbt model.
                - model_category: The logical dbt categorization of the model inside the project, such as base, staging, intermediate, marts...
                - vertical: Business domain or vertical to which the resource belongs, such as e-commerce, finance, supply...
                - has_tests: Boolean flag indicating if there are tests associated with this resource in the model .yaml file or in the tests folder.
                - has_select_all_in_last_select: Specifies if the final SQL query in the file uses a SELECT * statement. So all columns of previous CTEs will be consider as output of the model.
                - has_group_by: Boolean flag indicating if the SQL code includes a GROUP BY clause for aggregating data.
                - is_filtered: Boolean flag that specifies whether the resource applies filters to its data with WHERE, HAVING, JOIN...
                - is_source_model: The model uses the source macro to extract the data from the dbt project sources. It belongs to the first layer of all dbt models.
                - is_seed: Specifies if the resource is a seed file.
                - is_end_model: Boolean flag identifying if this is a terminal model in the data pipeline, representing the final output.
                - is_macro: Boolean flag indicating if the file defines a macro, used for reusable logic across the project. It does not count macros ref to call other models or source to connect to sources.
                - is_test: Indicates whether the file is a test sql definition used for testing purposes of other dbt models.
                - macros: Name of the macros used in the dbt model if any. It does not count macros ref to call other models or source to connect to sources.
                - packages: List of external packages or dependencies required for the resource or project functionality. Only not None in the packages.yml	row.

                RETRIEVED CONTEXT: {retrieved_context}
                --------------------
                and the examples about the some of the sources and the seeds (if any):

                {retrieved_csv_sources_context}
                --------------------
                Create a detailed step-by-step plan of the changes required in the existing models or files within the repository.
                To implement the requested change accurately.
                1. Ensure that you only refer to files, models, or fields that are explicitly mentioned in the retrieved information.
                2. Do not invent new models, fields, or dependencies. 
                3. Focus on:
                  - Identifying the exact files or models that need modifications, based on the retrieved context.
                  - Specifying what changes should be made, such as adding fields, updating logic, or modifying relationships.
                  - Highlighting any dependencies between models or files and describing how these should be handled.
                  - Extract children or parent models affected.
                  - If code fragments are provided in the retrieved context, incorporate them where applicable and explain their role.
                  - If no specific code or file is mentioned in the retrieved information, state that no changes should be made to existing files.
                  - Ensure the changes align with the dbt project's standards, such as conventions in `dbt_project.yml`, and do not introduce schema-breaking modifications.
                
                4. No extra checks or steps that are not on this list.
                5. Provide precise and actionable recommendations, avoiding any assumptions beyond the retrieved information.
                      
                <<<OUPUT>>>
                Return a summary of all the process, with the reflection, plan and context and the changes tht are neeeded to perfor
                      - The original request.
                      - The interpretarion of the request.
                      - The affected models with the info of the context and the lineage.
                      - All the necceasry changes step by step with a clear and short explanaiton of why is needed.
                Return only useful information, no additional filler text or unnecessary explanations, get to the point.
                REMEMBER: Provide no extra commentary or explanation, just the minimal information required.
            """)
        )

    def execute(self, input_data, context):
        print(f"Step: {self.name} | Input: {input_data}")
        evaluation = context.get("evaluation", "")
        selected_documents = context.get("selected_documents")
        lineage_analysis = context.get("lineage_analysis")

        print(selected_documents)
        retriever_documents = selected_documents["retriever_documents"]
        csv_sources_documents = selected_documents["csv_sources_documents"]
        yml_project_documents = selected_documents["yml_project_documents"]

        # Create a new vectorstore with the filtered documents
        new_vectorstore = Chroma.from_documents(retriever_documents, self.embedding_function)

        # Adjusted retriever
        #new_retriever = new_vectorstore.as_retriever()
        #retrieved_documents = new_retriever.invoke(input_data)
        new_retriever = EnhancedRetriever(vectorstore = new_vectorstore, embedding_function= self.embedding_function)
        retrieved_context, retrieved_documents = new_retriever.retrieve(input_data)

        combined_documents =  yml_project_documents + retrieved_documents

        retrieved_context = "\n".join([doc.page_content for doc in combined_documents if hasattr(doc, 'page_content')])
        retrieved_csv_sources_context = "\n".join([doc.page_content for doc in csv_sources_documents if hasattr(doc, 'page_content')])

        context["documents_for_plan"] = combined_documents
        print(f"Step: {self.name} | Retrieved Context: {retrieved_context}")

        prompt = self.prompt_template.format(evaluation=evaluation, retrieved_context=retrieved_context, lineage_analysis=lineage_analysis, retrieved_csv_sources_context = retrieved_csv_sources_context)

        plan = self.llm.invoke(prompt)
        context["plan"] = plan
        print(f"Step: {self.name} | Output: {plan}")
        return plan

class ChatPipeline:
    """Manages the pipeline of steps."""
    def __init__(self, steps):
        self.steps = steps

    def execute(self, input_data):
        context = {}
        for step in self.steps:
            print(f"\nExecuting step: {step.name}")
            result = step.execute(input_data, context)
            print(f"Finished step: {step.name} | Result: {result}\n")
        return context

def main(user_input, llm, vectorstore, embeddings, dbt_repo_knowledge_df):
    # Define the steps
    steps = [
        InterpretationStep(llm),
        EvaluationStep(llm),
        LineageStep(llm, vectorstore, embeddings, dbt_repo_knowledge_df),
        PlanStep(llm, embeddings),
    ]

    # Create the pipeline
    pipeline = ChatPipeline(steps)

    context = pipeline.execute(user_input)
    print("Context at the end of pipeline:", context)

    return context