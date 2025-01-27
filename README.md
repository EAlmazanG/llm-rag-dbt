# llm-rag-dbt

## Overview
This project provides a Retrieval-Augmented Generation (RAG) system designed to enhance LLM agents' understanding of dbt project structures. It retrieves context on models, dependencies, and documentation, allowing agents to generate, refine, and modify code step-by-step with project-specific accuracy. This approach minimizes hallucinations and improves responses by leveraging the full context of the repository.

## Problem Statement
Managing and modifying dbt projects can be challenging due to their complexity and interdependencies. This system aims to bridge the gap by equipping LLM agents with comprehensive knowledge of dbt models, enabling more accurate suggestions and modifications.

## Technologies


- **streamlit**: Displays the results in an accessible and interactive dashboard.

## Project Phases


## Folder Structure

```bash


```

## Phases of the project
### Data Collection


#### Data Source Options

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