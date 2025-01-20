import pandas as pd
import re
import yaml
import sqlparse
import os
import pandas as pd
import numpy as np
import requests
from IPython.display import display, Markdown

from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


def add_repo_root_path():
    import os
    import sys
    repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    if repo_root not in sys.path:
        sys.path.append(repo_root)
        
add_repo_root_path()
from src import generate_knowledge
from src import create_rag_db
from src import llm_chain_tools
from src.enhanced_retriever import EnhancedRetriever

def update_tasks_and_agents_config(files):
    # Load configurations from YAML files
    configs = {}
    for config_type, file_path in files.items():
        with open(file_path, 'r') as file:
            configs[config_type] = yaml.safe_load(file)

    # Assign loaded configurations to specific variables
    agents_config = configs['agents']
    tasks_config = configs['tasks']

    print(agents_config)
    print(tasks_config)
    return agents_config, tasks_config

files = {
    'agents': '../config/agents.yml',
    'tasks': '../config/tasks.yml'
}
agents_config, tasks_config = update_tasks_and_agents_config(files)

from crewai import Agent, Task, Crew
from crewai import Flow
from crewai.flow.flow import listen, start, and_, or_, router

# Agents
check_model_agent = Agent(
  config=agents_config['check_model_agent'],
)

search_model_agent = Agent(
  config=agents_config['search_model_agent'],
)

interpretation_agent = Agent(
  config=agents_config['interpretation_agent'],
)

generate_info_report_agent = Agent(
  config=agents_config['generate_info_report_agent'],
)

search_involved_models_agent = Agent(
  config=agents_config['search_involved_models_agent'],
)

solution_design_agent = Agent(
  config=agents_config['solution_design_agent'],
)

concilation_and_testing_agent = Agent(
  config=agents_config['concilation_and_testing_agent'],
)

## Tasks
check_model_task = Task(
  config=tasks_config['check_model_task'],
  agent=check_model_agent
)

search_model_task = Task(
  config=tasks_config['search_model_task'],
  agent=search_model_agent
)

interpretation_task = Task(
  config=tasks_config['interpretation_task'],
  agent=interpretation_agent
)

generate_info_report_task = Task(
  config=tasks_config['generate_info_report_task'],
  agent=generate_info_report_agent
)

search_models_impacted_task = Task(
  config=tasks_config['search_models_impacted_task'],
  agent=generate_info_report_agent
)

search_models_needed_task = Task(
  config=tasks_config['search_models_needed_task'],
  agent=generate_info_report_agent
)

solution_design_task = Task(
  config=tasks_config['solution_design_task'],
  agent=solution_design_agent
)

solution_design_models_impacted_task = Task(
  config=tasks_config['solution_design_models_impacted_task'],
  agent=solution_design_agent
)

concilation_and_testing_task = Task(
  config=tasks_config['concilation_and_testing_task'],
  agent=concilation_and_testing_agent
)

# Crews
check_model_crew = Crew(agents = [check_model_agent], tasks = [check_model_task], verbose = True)
search_model_crew = Crew(agents = [search_model_agent], tasks = [search_model_task], verbose = True)
interpretation_crew = Crew(agents = [interpretation_agent], tasks = [interpretation_task], verbose = True)

generate_info_report_crew = Crew(agents = [generate_info_report_agent], tasks = [generate_info_report_task], verbose = True)

search_models_impacted_task_crew = Crew(agents = [search_involved_models_agent], tasks = [search_models_impacted_task], verbose = True)
search_models_needed_task_crew = Crew(agents = [search_involved_models_agent], tasks = [search_models_needed_task], verbose = True)

solution_design_crew = Crew(agents = [solution_design_agent], tasks = [solution_design_task], verbose = True)
solution_design_models_impacted_crew = Crew(agents = [solution_design_agent], tasks = [solution_design_models_impacted_task], verbose = True)

concilation_and_testing_crew = Crew(agents = [concilation_and_testing_agent], tasks = [concilation_and_testing_task], verbose = True)

import nest_asyncio
nest_asyncio.apply()

class dbtChatFlow(Flow):
    def __init__(self, custom_llm=None):
        super().__init__()
        
        if custom_llm:
            self.update_agents_with_llm(custom_llm)
            print('using custom llm')

    def update_agents_with_llm(self, custom_llm):
        global check_model_agent, search_model_agent, interpretation_agent, generate_info_report_agent
        global search_involved_models_agent, solution_design_agent, concilation_and_testing_agent
        
        check_model_agent = Agent(
            config=agents_config['check_model_agent'],
            llm=custom_llm
        )

        search_model_agent = Agent(
            config=agents_config['search_model_agent'],
            llm=custom_llm
        )

        interpretation_agent = Agent(
            config=agents_config['interpretation_agent'],
            llm=custom_llm
        )

        generate_info_report_agent = Agent(
            config=agents_config['generate_info_report_agent'],
            llm=custom_llm
        )

        search_involved_models_agent = Agent(
            config=agents_config['search_involved_models_agent'],
            llm=custom_llm
        )

        solution_design_agent = Agent(
            config=agents_config['solution_design_agent'],
            llm=custom_llm
        )

        concilation_and_testing_agent = Agent(
            config=agents_config['concilation_and_testing_agent'],
            llm=custom_llm
        )

    @start()
    def check_model(self):
        request = self.state["request"]
        dbt_repo_knowledge_df = self.state["dbt_repo_knowledge_df"]

        lineage_df = create_rag_db.calculate_dbt_lineage(dbt_repo_knowledge_df)
        check_model_ouput = check_model_crew.kickoff(inputs = {"request": request, "lineage": str(lineage_df)})
        check_model_ouput_json =  eval(check_model_ouput.raw.replace("```json", "").replace("```", "").strip())
        
        self.state["check_model_ouput"] =check_model_ouput_json
        return check_model_ouput_json

    @listen(check_model)
    def retrieve_search_models(self, check_model_ouput_json):
        dbt_repo_knowledge_df = self.state["dbt_repo_knowledge_df"]
        vectorstore = self.state["vectorstore"]

        documents = llm_chain_tools.extract_documents_from_vectorstore(vectorstore)

        if not isinstance(check_model_ouput_json['identified_model'], list):
            identified_models = [check_model_ouput_json['identified_model']]
        identified_model_names = list(set(f"{model}.sql" for model in identified_models))
        identified_model_documents = [
            doc for doc in documents
            if hasattr(doc, 'metadata') and doc.metadata.get("name") in identified_model_names
        ]

        lineage_df = create_rag_db.calculate_dbt_lineage(dbt_repo_knowledge_df)
        identified_model_lineage = llm_chain_tools.get_affected_models(lineage_df, check_model_ouput_json['identified_model'])

        self.state["identified_model_documents"] =identified_model_documents
        return identified_model_names, identified_model_lineage, identified_model_documents

    @listen(retrieve_search_models)
    def search_model(self, retrieved_search_models):
        identified_model_names, identified_model_lineage, identified_model_documents = retrieved_search_models
        request = self.state["request"]
        
        search_impacted_models_ouput = search_model_crew.kickoff(
            inputs={
                "request": request,
                "lineage": str(identified_model_lineage),
                "impacted_models": identified_model_names,
                "impacted_models_documents": str(identified_model_documents)
            }
        )
        
        self.state["search_impacted_models_ouput"] = search_impacted_models_ouput
        return search_impacted_models_ouput
    
    @listen(search_model)
    def interpret_prompt(self):
        request = self.state["request"]

        interpretation = interpretation_crew.kickoff(inputs = {'request': request})
        self.state["interpretation"] = interpretation
        return interpretation

    @router(interpret_prompt)
    def select_required_ouput(self, interpretation):
        if interpretation.raw == 'RETRIEVE_INFO':
            return 'info'
        else:
            return 'code'

    @listen('info')
    def generate_info_report(self, search_impacted_models_ouput):
        request = self.state["request"]
        identified_model_documents = self.state["identified_model_documents"]
        
        generate_info_report_ouput = generate_info_report_crew.kickoff(
            inputs={
                "request": request,
                "search_impacted_models_ouput": str(search_impacted_models_ouput),
                "impacted_models_documents": str(identified_model_documents)
            }
        )

        self.state["generate_info_report_ouput"] = generate_info_report_ouput
        return generate_info_report_ouput

    @listen('code')
    def search_needed_models_for_change(self, search_impacted_models_ouput):
        request = self.state["request"]
        dbt_repo_knowledge_df = self.state["dbt_repo_knowledge_df"]
        check_model_ouput_json = self.state["check_model_ouput"]

        lineage_df = create_rag_db.calculate_dbt_lineage(dbt_repo_knowledge_df)

        search_needed_models_for_change_ouput = search_models_needed_task_crew.kickoff(
            inputs={
                "request": request,
                "identified_model": str(check_model_ouput_json['identified_model']),
                "search_impacted_models_ouput": str(search_impacted_models_ouput),
                "lineage_df": str(lineage_df)
            }
        )
        search_needed_models_for_change_ouput_json =  eval(search_needed_models_for_change_ouput.raw.replace("```json", "").replace("```", "").strip())
        self.state["search_needed_models_for_change_ouput"] = search_needed_models_for_change_ouput_json
        return search_needed_models_for_change_ouput_json
    
    @listen('code')
    def search_models_impacted_by_change(self, search_impacted_models_ouput):
        request = self.state["request"]
        dbt_repo_knowledge_df = self.state["dbt_repo_knowledge_df"]
        check_model_ouput_json = self.state["check_model_ouput"]

        lineage_df = create_rag_db.calculate_dbt_lineage(dbt_repo_knowledge_df)

        search_models_impacted_by_change_ouput = search_models_impacted_task_crew.kickoff(
            inputs={
                "request": request,
                "identified_model": str(check_model_ouput_json['identified_model']),
                "search_impacted_models_ouput": str(search_impacted_models_ouput),
                "lineage_df": str(lineage_df)
            }
        )
        search_models_impacted_by_change_ouput_json =  eval(search_models_impacted_by_change_ouput.raw.replace("```json", "").replace("```", "").strip())
        self.state["search_models_impacted_by_change_ouput"] = search_models_impacted_by_change_ouput_json
        return search_models_impacted_by_change_ouput_json

    @listen(search_needed_models_for_change)
    def retrieve_context_for_solution_main_model(self):
        search_needed_models_for_change_ouput = self.state["search_needed_models_for_change_ouput"]

        vectorstore = self.state["vectorstore"]
        embedding_function = self.state["embedding_function"]
        retriever = EnhancedRetriever(vectorstore = vectorstore, embedding_function= embedding_function)

        retrieve_context_for_solution_main_model = ""
        for model in search_needed_models_for_change_ouput['upstream_models']:
            retriever_input = f"""\n
                RELATION: parent model
                MODEL NAME: {model['model_name']}
                CONTEXT NEEDED FOR: {model['requirement']}
            \n"""
            _, retrieved_documents = retriever.retrieve(retriever_input)
            retrieved_context = "\n".join([doc.page_content for doc in retrieved_documents if hasattr(doc, 'page_content')])
            retrieve_context_for_solution_main_model += retriever_input + retrieved_context

        return retrieve_context_for_solution_main_model    

    @listen(search_models_impacted_by_change)
    def retrieve_context_for_solution_impacted_models(self):
        search_models_impacted_by_change_ouput = self.state["search_models_impacted_by_change_ouput"]

        vectorstore = self.state["vectorstore"]
        embedding_function = self.state["embedding_function"]
        retriever = EnhancedRetriever(vectorstore = vectorstore, embedding_function= embedding_function)

        retrieve_context_for_solution_impacted_models = ""
        for model_group in ['upstream_models', 'downstream_models']:
            for model in search_models_impacted_by_change_ouput.get(model_group, []):
                retriever_input = f"""
                    RELATION: {model_group}
                    MODEL NAME: {model['model_name']}
                    CONTEXT NEEDED FOR: {model['requirement']}
                """
                _, retrieved_documents = retriever.retrieve(retriever_input)
                retrieved_context = "\n".join([doc.page_content for doc in retrieved_documents if hasattr(doc, 'page_content')])
                retrieve_context_for_solution_impacted_models += retriever_input + retrieved_context

        return retrieve_context_for_solution_impacted_models

    @listen(retrieve_context_for_solution_main_model)
    def design_solution_main_model(self, retrieve_context_for_solution_main_model):
        request = self.state["request"]
        search_impacted_models_ouput = self.state["search_impacted_models_ouput"] #info about the model in markdown format
        identified_model_documents = self.state["identified_model_documents"]
        dbt_repo_knowledge_df = self.state["dbt_repo_knowledge_df"]

        lineage_df = create_rag_db.calculate_dbt_lineage(dbt_repo_knowledge_df)
        design_solution_main_model_output = solution_design_crew.kickoff(
            inputs={
                "request": request,
                "identified_model_documents": str(identified_model_documents),
                "search_impacted_models_ouput": str(search_impacted_models_ouput),
                "retrieved_context_complete": str(retrieve_context_for_solution_main_model),
                "lineage_df": str(lineage_df)
            }
        )
        
        self.state["design_solution_main_model_ouput"] = design_solution_main_model_output
        return design_solution_main_model_output

    @listen(and_(design_solution_main_model, retrieve_context_for_solution_impacted_models))
    def design_solution_impacted_models(self, retrieve_context_for_solution_impacted_models):
        request = self.state["request"]
        design_solution_main_model_ouput = self.state["design_solution_main_model_ouput"]
        search_models_impacted_by_change_ouput = self.state["search_models_impacted_by_change_ouput"]
        dbt_repo_knowledge_df = self.state["dbt_repo_knowledge_df"]

        lineage_df = create_rag_db.calculate_dbt_lineage(dbt_repo_knowledge_df)
        design_solution_impacted_models_output = solution_design_models_impacted_crew.kickoff(
            inputs={
                "request": request,
                "design_solution_main_model_ouput": str(design_solution_main_model_ouput),
                "search_models_impacted_by_change_ouput": str(search_models_impacted_by_change_ouput),
                "retrieve_context_for_solution_impacted_models": str(retrieve_context_for_solution_impacted_models),
                "lineage_df": str(lineage_df)
            }
        )
        self.state["design_solution_impacted_models_output"] = design_solution_impacted_models_output
        return design_solution_impacted_models_output

    @listen(and_(design_solution_main_model, design_solution_impacted_models))
    def concilation_and_testing(self):
        request = self.state["request"]
        design_solution_main_model_ouput = self.state["design_solution_main_model_ouput"]
        design_solution_impacted_models_output = self.state["design_solution_impacted_models_output"]
        dbt_repo_knowledge_df = self.state["dbt_repo_knowledge_df"]
    
        lineage_df = create_rag_db.calculate_dbt_lineage(dbt_repo_knowledge_df)
        concilation_and_testing_output = concilation_and_testing_crew.kickoff(
            inputs={
                "request": request,
                "design_solution_main_model_ouput": str(design_solution_main_model_ouput),
                "design_solution_impacted_models_output": str(design_solution_impacted_models_output),
                "lineage_df": str(lineage_df)
            }
        )
        self.state["concilation_and_testing_output"] = concilation_and_testing_output
        return concilation_and_testing_output