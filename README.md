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

Once the dbt repository is obtained, whether from a local path or an online source, the next step is cleaning and structuring the content. The project begins by identifying all relevant files within the repository, such as models, macros, snapshots, and documentation. This process ensures that only meaningful elements are processed.

#### 1. Repository Structure Analysis
- The repository's file structure is listed recursively, distinguishing between different file types such as `.sql`, `.yml`, `.csv`, and excluding irrelevant files.
- If the repository is hosted online (e.g., GitHub), the GitHub API is used to dynamically extract the file list.

#### 2. Filtering dbt Elements
- Specific dbt-related files are filtered using predefined extensions.
- Files within the `models/` directory are categorized separately from other project-level files.

#### 3. Model Dataframe Creation
- A structured DataFrame is built to store metadata about each model, including file paths, extensions, and names.
- Snapshot models are moved to the model section to align with dbt's internal categorization.

#### 4. Content Extraction
- Each relevant file is read and processed based on its type:
  - **SQL files** are formatted and analyzed for key components such as Jinja templates, materialization strategies, and dependencies.
  - **YAML files** are parsed to extract metadata, descriptions, and tests.
  - **CSV files** are cleaned and standardized to ensure compatibility with dbt's expectations.

#### 5. Configuration Parsing
- SQL models are examined for `config()` blocks to identify materialization settings and snapshot strategies.
- Jinja macros within the models are identified and categorized.

#### 6. Metadata Enhancement
- Relationships between models are analyzed by identifying dependencies using `ref()` and `source()` functions.
- Models are categorized based on naming conventions (e.g., `base`, `stg`, `int`).
- Columns, tests, and other metadata extracted from YAML files are matched with corresponding models.

#### 7. Project-level Details
- The `dbt_project.yml` and other project configuration files are parsed to extract global settings, package dependencies, and custom configurations.
- Packages are documented and their purposes summarized.

### Data Analysis

Once the data has been cleaned and structured, the next step involves analyzing relationships, dependencies, and generating meaningful insights for the RAG system. 

#### 1. Relationship
- Extract relationships between models by analyzing `ref()` and `source()` functions within SQL code.
- Identify parent-child relationships between models to better understand project dependencies.

#### 2. Generating Descriptions with LLM
- A language model (GPT-4o) is used to generate concise, human-readable descriptions of each model, macro, and project configuration.
- The descriptions summarize key aspects such as:
  - **Purpose:** What the model/macro is designed to do.
  - **Dependencies:** Which tables or models it interacts with.
  - **Filters/Aggregations:** Any key transformations performed within the model.

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

#### 3. Jinja Code
- The LLM is leveraged to analyze Jinja code within SQL models and macros.
- The extracted Jinja blocks are explained with details on their functionality and impact on the transformation logic.

### Data Storage

The final processed data is formatted into structured documents containing enriched information, making it easier to query and retrieve insights efficiently. This data is divided into meaningful chunks, preparing it for storage in the vector database (ChromaDB) for the RAG system.

### LLM Agents flow


### Tool use



2. **Running the App**: To start the application, run the following command in your terminal, navigating to your project’s root directory.

```bash
streamlit run app/app.py
```

3. **Data Loading**: 

4. **Interactivity**: 




![Tab1 of the dashboard](img/sar_tab1.gif)