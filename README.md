# llm-rag-dbt

## Overview
This project provides a Retrieval-Augmented Generation (RAG) system designed to enhance LLM agents' understanding of dbt project structures. It retrieves context on models, dependencies, and documentation, allowing agents to generate, refine, and modify code step-by-step with project-specific accuracy. This approach minimizes hallucinations and improves responses by leveraging the full context of the repository.

## Problem Statement
Managing and modifying dbt projects can be challenging due to their complexity and interdependencies. This system aims to bridge the gap by equipping LLM agents with comprehensive knowledge of dbt models, enabling more accurate suggestions and modifications.

## Technologies

- **Python**: Core language used for processing and orchestrating the system.
- **dbt**: Framework for managing and transforming data in warehouses. We will use a dbt repo as knowledge source.
- **ChromaDB**: Efficient vector database for storing and retrieving document embeddings.
- **CrewAI**: Framework for structuring multi-agent AI workflows.
- **LangChain**: Provides integration and chaining capabilities for LLM interactions.
- **LM Studio**: Used to run and support local LLM models.
- **LLM Models**: Configured to use the models available in the Open aI APi (mainly gpt-4o and gpt4o-mini) and local models (tested with DeepSeek R1 Q4_K_M, Qwen2.5 Coder 7B Instruct Q4_K_M and Llama 3.2 3B Instriuct 4bit).
- **streamlit**: Displays the interface of the tool and the interactions made with the Agents Flow.

## Project Phases

1. **Data Collection**: Extract information from a repository that can be either online or local.
2. **Data Cleaning**: Format and clean the repository content, its structure, and the code/documentation of all models, macros, project files, snapshots, etc.
3. **Data Analysis**: Analyze relationships between models in the dbt project and add them to the processed information. Generate textual descriptions of the code, macros, models, and project to provide an easily interpretable source for the RAG system. Format all data into a document containing all necessary information for each model for later storage in a vector database.
4. **Data Storage**: Process the formatted documents by dividing them into chunks and storing them in ChromaDB, which will serve as the database for the RAG system.
5. **LLM Agents Flow**: Configure an agent flow using CrewAI, where agents analyze and process requests step-by-step, querying the RAG for the necessary information at each stage.
6. **Tool and Visualization**: Implement an interface in Streamlit to enable easy interaction with all functionalities, formatting input and output from the agent flow as a chat interface.


## Folder Structure

```bash

llm-rag-dbt/
│
├── chromadb/                     # ChromaDB storage
├── config/                       # Configuration files
│   ├── agents.yml                 # Agent definitions and roles
│   └── tasks.yml                  # Task definitions for agent workflow
│
├── data/                          # Sample and processed data
│
├── notebooks/                     # Jupyter notebooks for development and testing
│   ├── create_rag_db.ipynb         # Notebook to generate the RAG database
│   ├── generate_knowledge.ipynb    # Notebook for extracting insights
│   ├── interface_design.ipynb      # UI/UX design for interaction
│   ├── llm_agents_tests.ipynb      # Agent behavior and response validation
│   └── llm_chain_tests.ipynb       # Testing the multi-agent workflow
│
├── src/                          # Source code
│   ├── create_rag_db.py           # Scripts to generate RAG database
│   ├── enhanced_retriever.py      # Improved retriever class
│   ├── generate_knowledge.py      # Knowledge extraction scripts
│   ├── interface_app.py           # Streamlit app interface
│   ├── llm_agents_flow.py         # Agent orchestration logic
│   ├── llm_chain.py               # Chaining of agents and context handling
│   └── llm_chain_tools.py         # Utility functions for LLM chain operations
│
├── test_repo/                     # Sample dbt project for testing
├── img/                           # Images and GIFs for README
│
├── requirements.txt             # Dependencies (pandas, scikit-learn, scrapy, etc.)
├── environment.yml              # Conda environment configuration
├── openai_setup.yml             # Keys for co (dont sync it in your repo!)
├── .gitignore                   # Ignore unnecessary files
└── README.md     

```

## Phases of the project
### Data Collection

We will select a repository from a dbt project, to process and clean up the structure and content of the repository, and use it as a knowledge base to give context to our LLM when making requests from the interface.

> **‼️👀🚨 IMPORTANT 🚨👀‼️**: The dbt project must be structure following the dbt best practices. If the models folder or project main files are not set as expected, the python script could not read them properly, so the context of this files wouldn't be added to the RAG context. Adapt the code if your repo structure differs.

#### Data Source Options

- **Local repository**: Select the path to the local repository you want to process. The test_repo folder contains an example using jaffle-shop, from dbt labs.
- **Online repository**: Enter the url of the repository to read, the scrit will read the structure and content automatically using the GitHub api and the requests library.
- **Already processed repository**: If you have already used a repository before, load it directly by selecting the corresponding dbt_models_knowledge file that will have been created in the data folder, in order to reuse the already processed data and speed up the process.

### Data Cleaning


### Data Analysis



#### Using the GPT API

The **GPT API** is integrated into this project to enable 


#### Purpose of the GPT API


#### Setup Instructions
1. **API Key**: To enable the GPT functionality, you'll need an API key from OpenAI. If you haven’t done so already, sign up for an API key at [OpenAI's website](https://platform.openai.com/signup).
   
2. **Configuration File**: In the project’s root directory, create or locate the `openai_setup.py` file. Replace the placeholders with your OpenAI credentials as shown below:

```python
   conf = {
       'organization': 'your_organization_key_here',
       'project': 'your_project_key_here',
       'key': 'your_openai_api_key_here'
   }
```

**Important**: Ensure that openai_setup.py is included in your .gitignore file to keep your API key secure and prevent accidental exposure in public repositories.

**Usage Notes**
- **Data Privacy:** Be mindful that enabling the GPT API may send review text to OpenAI’s servers. Consider reviewing OpenAI’s data usage policy to understand how your data is handled.
- **API Costs:** Since the GPT API is a paid service, usage may incur costs. Track your usage on the OpenAI dashboard to manage API expenses.


### Data Storage


### LLM Agents flow


### Tool use



2. **Running the App**: To start the application, run the following command in your terminal, navigating to your project’s root directory.

```bash
streamlit run app/app.py
```

3. **Data Loading**: 

4. **Interactivity**: 




![Tab1 of the dashboard](img/sar_tab1.gif)