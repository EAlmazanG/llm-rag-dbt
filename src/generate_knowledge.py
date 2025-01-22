import pandas as pd
import re
import yaml
import sqlparse
import os
import pandas as pd
import requests
from tqdm import tqdm
from io import StringIO
tqdm.pandas()

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

def extract_owner_and_repo(github_url):
    try:
        # Remove the base URL and split the rest
        parts = github_url.replace("https://github.com/", "").split("/")
        # Validate structure
        if len(parts) >= 2:
            owner = parts[0]
            repo = parts[1]
            return owner, repo
        else:
            raise ValueError("Invalid GitHub URL structure.")
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def list_local_repo_structure(repo_path):
    paths = []
    for root, dirs, files in os.walk(repo_path):
        rel_dir = os.path.relpath(root, repo_path)
        if rel_dir == '.':
            rel_dir = ''
        if rel_dir:
            paths.append(rel_dir + '/')
        for f in files:
            file_path = f"{rel_dir}/{f}" if rel_dir else f
            paths.append(file_path)
    return paths

def list_online_repo_structure(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/"
    stack = [(url, '')]
    paths = []
    while stack:
        current_url, current_path = stack.pop()
        response = requests.get(current_url)
        if response.status_code == 200:
            items = response.json()
            for item in items:
                if item['type'] == 'dir':
                    paths.append(current_path + item['name'] + '/')
                    stack.append((item['url'], current_path + item['name'] + '/'))
                else:
                    paths.append(current_path + item['name'])
    return paths

def is_online_repo(path):
    return path.startswith("http://") or path.startswith("https://")

def select_dbt_elements_by_extension(repo_elements):
    dbt_extensions = ['.sql', '.yml', '.yaml', '.csv']
    # Filter elements with relevant extensions
    return [element for element in repo_elements if any(element.endswith(ext) for ext in dbt_extensions)]

def select_dbt_models(repo_dbt_elements):
    dbt_extensions = ['.sql', '.yml', '.yaml', '.csv']
    return [
        element for element in repo_dbt_elements
        if element.startswith('models/') and any(element.endswith(ext) for ext in dbt_extensions)
    ]

def select_dbt_project_files(repo_elements):
    valid_extensions = [".sql", ".yml", ".csv"]
    exclude_folders = ["models/"]

    # Filter repo elements
    filtered_elements = [
        element for element in repo_elements
        if any(element.endswith(ext) for ext in valid_extensions)
        and not any(folder in element for folder in exclude_folders)
        and not element.startswith(".")
    ]

    # Create DataFrame
    repo_df = pd.DataFrame(filtered_elements, columns=["path"])

    # Add columns for useful details
    repo_df["name"] = repo_df["path"].apply(lambda x: x.split("/")[-1])
    repo_df["extension"] = repo_df["path"].apply(lambda x: "." + x.split(".")[-1])
    return repo_df

def generate_dbt_models_df(repo_dbt_models):
    data = []
    for path in repo_dbt_models:
        name = os.path.basename(path)
        extension = os.path.splitext(name)[1]
        data.append({'path': path, 'name': name, 'extension': extension})
    return pd.DataFrame(data)

def get_base_url(repo_url):
    if repo_url.startswith("https://github.com"):
        parts = repo_url.replace("https://github.com/", "").split("/")
        owner, repo = parts[0], parts[1]
        return f"https://raw.githubusercontent.com/{owner}/{repo}/main"
    else:
        raise ValueError("URL not valid.")

def move_snapshots_to_models(dbt_project_df, dbt_models_df):
    snapshots_filter = dbt_project_df['path'].str.contains(r'(snapshots/|^snap)', case=False, regex=True)

    snapshots_rows = dbt_project_df[snapshots_filter]
    dbt_project_df = dbt_project_df[~snapshots_filter]

    dbt_models_df = pd.concat([dbt_models_df, snapshots_rows], ignore_index=True)

    return dbt_project_df, dbt_models_df

def extract_model_file_content(path, is_online=False, repo_base_url=None):
    try:
        if is_online:
            # Build complete URL
            file_url = f"{repo_base_url}/{path}" if repo_base_url else path
            response = requests.get(file_url)
            if response.status_code == 200:
                content = response.text
            else:
                return f"Error: {response.status_code} {response.reason}"
        else:
            # Read content locally
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()

        # Process content based on file type
        if path.endswith(('.yml', '.yaml')):
            try:
                return yaml.safe_load(content)  # Parse YAML and return as dictionary
            except yaml.YAMLError as e:
                return f"Error parsing YAML: {e}"
        elif path.endswith('.sql'):
            try:
                return sqlparse.format(content, reindent=True, keyword_case='upper')  # Format SQL
            except Exception as e:
                return f"Error parsing SQL: {e}"
        else:
            return content  # Return plain text for other types

    except Exception as e:
        return f"Error: {e}"

def add_model_code_column(df, is_online=False, repo_url=None):
    if is_online:
        repo_base_url = get_base_url(repo_url)
    else:
        repo_base_url = ''

    # Extract content for each file and process it based on type
    df['sql_code'] = df['path'].apply(lambda path: extract_model_file_content(path, is_online, repo_base_url))
    return df

def extract_config_block(sql_code):
    pattern = r"{{\s*config\((.*?)\)\s*}}"
    match = re.search(pattern, sql_code, re.DOTALL)
    return match.group(0) if match else None

def add_config_column(df):
    df['config'] = df.apply(
        lambda row: extract_config_block(row['sql_code']) if row['extension'] == '.sql' else None,
        axis=1
    )
    return df

def extract_materialized_value(config_text):
    if config_text:
        match = re.search(r"materialized\s*=\s*[\"']([^\"']+)[\"']", config_text)
        return match.group(1) if match else None
    return None

def check_is_snapshot(config_text):
    if config_text:
        return 'strategy' in config_text
    return False

def contains_jinja_code(code_text):
    if isinstance(code_text, str):
        return bool(re.search(r"{%|{#", code_text))
    return False

def categorize_model(name):
    if name.startswith("base"):
        return "base"
    elif name.startswith("stg"):
        return "stg"
    elif name.startswith("int"):
        return "int"
    elif name.startswith("test"):
        return "test"
    elif name.startswith("snap"):
        return "snap"
    elif name.startswith("__sources"):
        return "sources"
    else:
        return "other"
    
def get_vertical(name, model_category):
    base_name = re.sub(r'\.[^.]+$', '', name)
    
    if model_category == 'sources':
        return 'sources'
    
    known_categories = ['stg', 'int']
    if model_category not in known_categories:
        # Para model_category = other u otras no conocidas, devolver base_name sin extensión
        return base_name
    
    # Para stg o int, extraer vertical antes de "__" o "."
    pattern = rf'^{re.escape(model_category)}_([a-z0-9_]+?)(?:__|\.|$)'
    match = re.search(pattern, base_name)
    return match.group(1) if match else base_name

def assign_yml_rows_to_each_model(dbt_models_df):
    dbt_models_df['yml_code'] = None

    yml_df = dbt_models_df[dbt_models_df['extension'] == '.yml'].copy()
    yml_df['delete'] = False

    for idx, row in yml_df.iterrows():
        base_name = row['name'].rsplit('.', 1)[0]

        sql_match = dbt_models_df[(dbt_models_df['name'] == base_name + '.sql')]

        if not sql_match.empty:
            dbt_models_df.at[sql_match.index[0], 'yml_code'] = row['sql_code']
            yml_df.at[idx, 'delete'] = True
        else:
            yml_df.at[idx, 'yml_code'] = row['sql_code']
            yml_df.at[idx, 'sql_code'] = None

    yml_df = yml_df[~yml_df['delete']]

    dbt_models_df = dbt_models_df[dbt_models_df['extension'] != '.yml']

    yml_df = yml_df.drop(columns=['delete'])
    dbt_models_df = pd.concat([dbt_models_df, yml_df], ignore_index=True)

    return dbt_models_df

def extract_tests(yml_code):
    if not isinstance(yml_code, dict):
        return None

    tests_dict = {'columns': {}, 'unit_tests': []}

    # Extract tests from all models
    for model in yml_code.get('models', []):
        for column in model.get('columns', []):
            column_name = column.get('name')
            if column_name:
                # Combine 'tests' and 'data_tests' if present
                tests = column.get('tests', []) + column.get('data_tests', [])
                if tests:
                    tests_dict['columns'][column_name] = tests

    # Extract unit tests
    if 'unit_tests' in yml_code:
        unit_test_names = [test.get('name') for test in yml_code['unit_tests'] if test.get('name')]
        if unit_test_names:
            tests_dict['unit_tests'] = unit_test_names

    return tests_dict if tests_dict['columns'] or tests_dict['unit_tests'] else None

def extract_ids_from_query(code):
    if not isinstance(code, str):
        return None
    
    # Parse the SQL query
    parsed = sqlparse.parse(code)
    if not parsed:
        return None
    
    # Regular expression to find columns ending in '_id'
    id_pattern = re.compile(r'\b(\w+_id)\b')
    
    cte_ids = set()
    output_ids = set()
    
    for statement in parsed:
        # Flatten tokens to handle nested structures
        token_list = sqlparse.sql.TokenList(statement.tokens).flatten()
        inside_cte = False
        
        for token in token_list:
            # Detect CTE start (with keyword 'WITH')
            if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == 'WITH':
                inside_cte = True
            
            # Detect SELECT after a WITH block ends
            if inside_cte and token.ttype is sqlparse.tokens.Keyword and token.value.upper() == 'SELECT':
                inside_cte = False
            
            if token.ttype is sqlparse.tokens.Name or token.ttype is None:
                match = id_pattern.search(token.value)
                if match:
                    if inside_cte:
                        cte_ids.add(match.group(1))
                    else:
                        output_ids.add(match.group(1))
    ids = {
        'cte_ids': list(cte_ids),
        'output_ids': list(output_ids)
    }
    return ids['output_ids'] if ids['output_ids'] != [] else None

def has_select_all_in_last_select(code):
    if not isinstance(code, str):
        return False

    parsed = sqlparse.parse(code)
    if not parsed:
        return False

    select_statements = [stmt for stmt in parsed if stmt.get_type() == 'SELECT']
    if not select_statements:
        return False
    last_select = select_statements[-1]

    for token in last_select.tokens:
        if token.ttype is sqlparse.tokens.Wildcard and token.value == '*':
            return True

    return False

def has_group_by(code):
    if not isinstance(code, str):
        return False

    parsed = sqlparse.parse(code)
    if not parsed:
        return False
    return 'group by' in code.lower()

def find_primary_key(tests_dict):
    if not isinstance(tests_dict, dict) or 'columns' not in tests_dict:
        return None

    for column, tests in tests_dict.get('columns', {}).items():
        # Check if the column has the required tests for a primary key
        if tests == ['not_null', 'unique'] or 'dbt_constraints.primary_key' in tests:
            return column
    
    return None

def extract_sql_filters(sql_query):
    if not isinstance(sql_query, str) or not sql_query.strip():
        return None

    sql_query_clean = ' '.join(sql_query.split()).lower()

    filters_patterns = [
        (r'\bwhere\b\s+(.*?)(?=\bgroup\b|\border\b|\blimit\b|\bhaving\b|;|$)', 'where'),
        (r'\bon\b\s+(.*?)(?=\bleft\b|\bright\b|\binner\b|\bouter\b|\bjoin\b|\bselect\b|\bwhere\b|\bgroup\b|\border\b|\blimit\b|;|$)', 'join'),
        (r'\bhaving\b\s+(.*?)(?=\bgroup\b|\border\b|\blimit\b|;|$)', 'having')
    ]

    filters = []
    joins = []

    for pattern, clause_type in filters_patterns:
        matches = re.findall(pattern, sql_query_clean, re.DOTALL)
        for match in matches:
            sub_conditions = re.split(r'\band\b|\bor\b', match)
            for condition in sub_conditions:
                cleaned = condition.strip().strip('()')
                if cleaned:
                    if clause_type == 'join':
                        joins.append(cleaned)
                    else:
                        filters.append(cleaned)
    all_filters = filters + joins
    return all_filters if all_filters != [] else None

def extract_dbt_macros(sql_query):

    if not isinstance(sql_query, str) or not sql_query.strip():
        return None
    
    macro_pattern = r"\{\{\s*([\w\.]+)\s*\(.*?\)\s*\}\}"
    matches = re.findall(macro_pattern, sql_query)
    filtered_macros = sorted(set(m for m in matches if m not in ('ref', 'source')))
    
    return filtered_macros if filtered_macros != [] else None

def extract_source_details(code, source_pattern):
    if not isinstance(code, str):
        return False, None
    sources = re.findall(source_pattern, code)
    if sources:
        return True, [f"{source[0]}.{source[1]}" for source in sources]
    return False, None

def enrich_dbt_models(dbt_models_df):
    # Helper regex patterns
    source_pattern = r"\{\{\s*source\(['\"](.*?)['\"],\s*['\"](.*?)['\"]\)\s*\}\}"
    ref_pattern = r"\{\{\s*ref\(['\"](.*?)['\"]\)\s*\}\}"
    
    # Add 'parent_models' - extract all models referenced using 'ref'
    dbt_models_df['parent_models'] = dbt_models_df['sql_code'].apply(
        lambda code: re.findall(ref_pattern, code) if isinstance(code, str) else []
    )
    
    dbt_models_df[['is_source_model', 'source']] = dbt_models_df['sql_code'].apply(
        lambda code: pd.Series(extract_source_details(code, source_pattern))
    )
    
    # Build a dictionary to track children relationships
    model_children = {}
    for idx, row in dbt_models_df.iterrows():
        for parent in row['parent_models']:
            model_children.setdefault(parent, []).append(row['name'].replace('.sql', ''))

    # Add 'children_models' - list all models that depend on this model
    dbt_models_df['children_models'] = dbt_models_df['name'].apply(
        lambda name: model_children.get(name.replace('.sql', ''), [])
    )
    
    # Add 'is_end_model' - True if there are no children
    dbt_models_df['is_end_model'] = dbt_models_df['children_models'].apply(lambda children: len(children) == 0)
    
    return dbt_models_df

def add_repo_root_path():
    import os
    import sys
    repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    if repo_root not in sys.path:
        sys.path.append(repo_root)

def generate_query_description(llm, query, documentation = None):
    # Context and prompt
    prompt = f"""
        You are an expert data analyst, dbt analytics engineer and technical writer. 
        Your task is to generate a concise, clear, and standardized description of the following SQL query of a dbt model.

        SQL Query of the dbt model:
        {query}

        Additional Documentation in the model yaml file:
        {documentation}

        Guidelines:
        1. Describe the **main purpose** of the query in 2 to 3 sentences.
        2. Include the following details explicitly:
        - Tables or data sources referenced.
        - Filters, conditions, and joins applied.
        - Any aggregations (e.g., SUM, COUNT) or calculations performed.
        3. Avoid technical jargon and vague expressions.
        4. Limit the description to around **50 words** maximum.
        5. Maintain a similar level of detail and length for all responses to ensure consistency.
        6. Don't use "This query" or "This model", all the info must usefull and coherent.
        
        Format Example:
        "Retrieves all customer records from the 'customers' table where the country is 'US'. It joins the 'orders' table on 'customer_id' to calculate the total order amount per customer using SUM(). The result is grouped by 'customer_id'."
    """

    # Interact
    response = llm([HumanMessage(content=prompt)])
    return response.content

def generate_model_description(llm, row):
    if pd.notna(row['sql_code']) or pd.notna(row['yml_code']):
        sql_code = row['sql_code'] if pd.notna(row['sql_code']) else ""
        yml_code = row['yml_code'] if pd.notna(row['yml_code']) else ""
        return generate_query_description(llm, sql_code, yml_code)
    return None

def generate_jinja_code_description(llm, query, documentation = None):
    # Context and prompt
    prompt = f"""
        You are an expert data analyst, dbt analytics engineer, technical writer, and Jinja programmer. 
        Your task is to generate a concise, clear, and standardized description of the Jinja code within the dbt model.

        SQL Query of the dbt model with the Jinja code:
        {query}

        Additional Documentation in the model yaml file:
        {documentation}

        Guidelines:
        1. Focus only on what the Jinja code does, ignoring the logic or dependencies related to refs or source functions.
        2. Clearly explain the **main purpose** of the Jinja code in plain language.
        3. Avoid technical jargon and vague expressions.
        4. Limit the description to around **50 words** maximum.
        5. Ensure all responses are coherent, useful, and follow a consistent format.
        6. Avoid using phrases like "This ..." or "The code ...". Focus on describing the purpose directly.
        7. If has multiple sections, describe each section separetly.

        Examples Format:
        - Calculates the rolling average of sales over the last 30 days for each product.
        - Formats the date column to a standard YYYY-MM-DD format.
        - Dynamically generates filter conditions based on user inputs.

        Provide the description of the Jinja code:
    """

    # Interact
    response = llm([HumanMessage(content=prompt)])
    return response.content

def generate_jinja_description(llm, row):
    if row['has_jinja_code']:
        sql_code = row['sql_code'] if pd.notna(row['sql_code']) else ""
        yml_code = row['yml_code'] if pd.notna(row['yml_code']) else ""
        return generate_jinja_code_description(llm, sql_code, yml_code)
    return None

def extract_project_file_content(file_path, is_online=False, repo_base_url=None):
    try:
        # Read content from local or online
        if is_online:
            # Build complete URL
            file_url = f"{repo_base_url}/{file_path}" if repo_base_url else file_path
            response = requests.get(file_url)
            if response.status_code == 200:
                content = response.text
            else:
                return f"Error: {response.status_code} {response.reason}"
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

        # Process content based on file type
        if file_path.endswith(('.yml', '.yaml')):
            try:
                yaml_content = yaml.safe_load(content)
                return yaml.dump(yaml_content, sort_keys=False, default_flow_style=False)
            except yaml.YAMLError as e:
                return f"Error parsing YAML: {e}"
        elif file_path.endswith('.sql'):
            try:
                return sqlparse.format(content, reindent=True, keyword_case="lower")
            except Exception as e:
                return f"Error parsing SQL: {e}"
        elif file_path.endswith('.csv'):
            try:
                from io import StringIO
                df = pd.read_csv(StringIO(content))
                df_cleaned = df.dropna().reset_index(drop=True)  # Clean missing values and reset index
                df_cleaned.columns = [col.strip().lower() for col in df_cleaned.columns]  # Standardize column names
                return df_cleaned.head(3).to_string(index=False)
            except Exception as e:
                return f"Error reading CSV: {e}"
        else:
            return content  # Return plain text for unsupported types

    except Exception as e:
        return f"Error: {e}"
    
def add_project_code_column(df, is_online=False, repo_url=None):
    if is_online:
        repo_base_url = get_base_url(repo_url)
    else:
        repo_base_url = ''

    # Extract content for each file and process it based on type
    df['code'] = df['path'].apply(lambda path: extract_project_file_content(path, is_online, repo_base_url))
    return df

def is_seed(file_path):
    file_name = os.path.basename(file_path)
    return "seeds/" in file_path or file_name.lower().startswith("seed_")

def is_macro(file_path):
    return file_path.endswith(".sql") and "macros/" in file_path

def is_test(file_path):
    file_name = os.path.basename(file_path)
    return "tests/" in file_path or file_name.lower().startswith("test_")

def extract_packages(row):
    if 'name' in row and row['name'] == 'packages.yml':
        try:
            packages_content = yaml.safe_load(row['code'])
            packages = []
            for pkg in packages_content.get("packages", []):
                if "package" in pkg:
                    packages.append(f"{pkg['package']}=={pkg.get('version', 'unknown_version')}")
                elif "git" in pkg:
                    revision = pkg.get("revision", "unknown_revision")
                    packages.append(f"{pkg['git']}@{revision}")
            return packages
        except Exception as e:
            print(f"Error processing packages content: {e}")
            return []
    return None

def generate_packages_code_description(llm, packages):
    # Context and prompt
    prompt = f"""
        You are a dbt expert. Your task is to generate a concise and clear description of the installed dbt packages.

        Installed Packages:
        {packages}

        Guidelines:
        1. Provide an overview of the purpose and functionality of each package listed.
        2. Write in a professional tone, avoiding redundancy and technical jargon.
        3. Focus on the practical use and benefits of each package for dbt projects.
        4. Limit the description to **70 words** per package, ensuring clarity and relevance.
        5. If possible, group related packages and highlight their combined purpose.

        Examples:
        - `dbt_utils`: Provides helper macros to simplify common dbt tasks, such as column selection, testing, and transformations.
        - `dbt_audit_helper`: Facilitates data quality audits by generating SQL for data validation and testing.
        - `dbt_date`: Simplifies date-related transformations and calculations for better time series analysis.

        Provide a clear and concise description of the installed packages:
    """

    # Interact
    response = llm([HumanMessage(content=prompt)])
    return response.content

def generate_packages_description(llm, row):
    if row['packages'] is not None:
        return generate_packages_code_description(llm, row['packages'])
    return row['description']

def generate_macro_code_description(llm, query):
    # Context and prompt
    prompt = f"""
        You are a dbt expert and Jinja programmer. Your task is to generate a concise description of the macro code provided.

        Macro Jinja Code:
        {query}

        Guidelines:
        1. Focus on the main purpose of the macro in simple and clear language.
        2. Avoid explaining dbt-specific functions like `ref`, `source`, or `config`, unless they are central to the macro's purpose.
        3. Write in a professional tone, avoiding redundant phrases like "This macro" or "The code".
        4. Limit the description to a maximum of **50 words**.
        5. Describe multiple functional sections separately if applicable, using a clear and structured format.

        Examples:
        - Creates dynamic SQL for filtering data by date range and category.
        - Defines a reusable calculation for profit margin across models.
        - Dynamically formats column names to snake_case based on inputs.
        - Generates a pivot table structure for specified dimensions.

        Provide a clear and concise description of the macro:
    """

    # Interact
    response = llm([HumanMessage(content=prompt)])
    return response.content

def generate_macro_description(llm, row):
    if row['is_macro']:
        sql_code = row['code'] if pd.notna(row['code']) else ""
        return generate_macro_code_description(llm, sql_code)
    return row['description']

def generate_dbt_config_code_summary(llm, config_content):
    prompt = f"""
        You are an expert in dbt configurations. Your task is to extract and summarize the key configurations from the provided dbt project configuration file.

        dbt Project Configuration Content:
        {config_content}

        Guidelines:
        1. Extract and list the key configurations such as project name, version, schema, paths, and any custom settings.
        2. Highlight specific values or settings for important configurations (e.g., `name`, `version`, `target-path`, `source-paths`).
        3. Ignore general information or default values unless explicitly overridden.
        4. Format the output as a clear and concise bulleted list of configurations.
        5. Avoid explanations or context; only list the configurations and their values.

        Example Output:
        - Name: jaffle_shop
        - Version: 1.2
        - Target Path: target/
        - Source Paths: models/
        - Schema: analytics
        - Custom Setting: Materialized as incremental

        Provide a concise bulleted summary of the dbt project configuration:
    """
    response = llm([HumanMessage(content=prompt)])
    return response.content

def generate_dbt_config_summary(llm, row):
    if row['name'] == 'dbt_project.yml' and pd.notna(row['code']):
        return generate_dbt_config_code_summary(llm, row['code'])
    return row['description']

def generate_test_code_description(llm, tests_content):
    prompt = f"""
        You are a dbt expert. Your task is to generate a concise and clear description of the tests applied in the dbt project.

        Tests Content:
        {tests_content}

        Guidelines:
        1. Summarize the purpose and functionality of each test, focusing on what it validates or ensures.
        2. Use a professional tone, avoiding redundancy and overly technical details.
        3. Highlight key aspects such as data quality, constraints, and validation objectives.
        4. Limit the description to **50 words** per test, ensuring clarity and relevance.
        5. If applicable, group similar tests and describe their collective purpose.

        Examples:
        - Test 1: Validates that all primary keys in the `orders` table are unique.
        - Test 2: Ensures that no null values exist in critical columns like `customer_id` and `order_date`.
        - Grouped Tests: Validate referential integrity between `orders` and `customers`.

        Provide a clear and concise description of the tests:
    """
    response = llm([HumanMessage(content=prompt)])
    return response.content

def generate_tests_description(llm, row):
    if row['is_test']:
        return generate_test_code_description(llm, row['code'])
    return row['description']

def generate_knowledge_from_repo_elements(repo_elements, is_online, repo_path):
    add_repo_root_path()
    import openai_setup
    OPENAI_API_KEY = openai_setup.conf['key']
    OPENAI_PROJECT = openai_setup.conf['project']
    OPENAI_ORGANIZATION = openai_setup.conf['organization']
    DEFAULT_LLM_MODEL = "gpt-4o-mini"

    repo_dbt_elements = select_dbt_elements_by_extension(repo_elements)
    repo_dbt_models = select_dbt_models(repo_dbt_elements)
    dbt_project_df = select_dbt_project_files(repo_dbt_elements)
    dbt_models_df = generate_dbt_models_df(repo_dbt_models)
    dbt_project_df, dbt_models_df = move_snapshots_to_models(dbt_project_df, dbt_models_df)
    dbt_models_df = add_model_code_column(dbt_models_df, is_online = True, online_dbt_repo = repo_path)
    dbt_models_df = add_config_column(dbt_models_df)
    dbt_models_df['materialized'] = dbt_models_df['config'].apply(extract_materialized_value)
    dbt_models_df['is_snapshot'] = dbt_models_df['config'].apply(check_is_snapshot)
    dbt_models_df['materialized'] = dbt_models_df.apply(lambda row: 'snapshot' if row['is_snapshot'] else row['materialized'] ,1)
    dbt_models_df['has_jinja_code'] = dbt_models_df['sql_code'].apply(contains_jinja_code)
    dbt_models_df['model_category'] = dbt_models_df['name'].apply(categorize_model)
    dbt_models_df['vertical'] = dbt_models_df.apply(lambda row: get_vertical(row['name'], row['model_category']), axis=1)
    dbt_models_df = assign_yml_rows_to_each_model(dbt_models_df)
    dbt_models_df['tests'] = dbt_models_df['yml_code'].apply(extract_tests)
    dbt_models_df['has_tests'] = dbt_models_df['tests'].apply(lambda x: x is not None)
    dbt_models_df['sql_ids'] = dbt_models_df['sql_code'].apply(extract_ids_from_query)
    dbt_models_df['has_select_all_in_last_select'] = dbt_models_df['sql_code'].apply(has_select_all_in_last_select)
    dbt_models_df['has_group_by'] = dbt_models_df['sql_code'].apply(has_group_by)
    dbt_models_df['primary_key'] = dbt_models_df['tests'].apply(find_primary_key)
    dbt_models_df['filters'] = dbt_models_df['sql_code'].apply(extract_sql_filters)
    dbt_models_df['is_filtered'] = dbt_models_df['filters'].apply(lambda x: x is not None)
    dbt_models_df['macros'] = dbt_models_df['sql_code'].apply(extract_dbt_macros)
    dbt_models_df['has_macros'] = dbt_models_df['macros'].apply(lambda x: x is not None)
    dbt_models_enriched_df = enrich_dbt_models(dbt_models_df)

    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage

    llm = ChatOpenAI(model=DEFAULT_LLM_MODEL, temperature=0.1, openai_api_key=OPENAI_API_KEY, openai_organization = OPENAI_ORGANIZATION)
    dbt_models_enriched_df['model_description'] = dbt_models_enriched_df.progress_apply(
        lambda row: generate_model_description(llm, row),
        axis=1
    )
    dbt_models_enriched_df['jinja_description'] = dbt_models_enriched_df.progress_apply(
        lambda row: generate_jinja_description(llm, row),
        axis=1
    )
    dbt_project_df = add_project_code_column(dbt_project_df, is_online, online_dbt_repo = repo_path)
    dbt_project_df['is_seed'] = dbt_project_df['path'].apply(is_seed)
    dbt_project_df['is_macro'] = dbt_project_df['path'].apply(is_macro)
    dbt_project_df['is_test'] = dbt_project_df['path'].apply(is_test)
    dbt_project_df['packages'] = dbt_project_df.apply(extract_packages, 1)
    dbt_project_df['description'] = None
    dbt_project_df['description'] = dbt_project_df.progress_apply(
        lambda row: generate_packages_description(llm, row),
        axis=1
    )
    dbt_project_df['description'] = dbt_project_df.progress_apply(
        lambda row: generate_macro_description(llm, row),
        axis=1
    )
    dbt_project_df['description'] = dbt_project_df.progress_apply(
        lambda row: generate_dbt_config_summary(llm, row),
        axis=1
    )
    dbt_project_df['description'] = dbt_project_df.progress_apply(
        lambda row: generate_tests_description(llm, row),
        axis=1
    )
    _, repo_name = extract_owner_and_repo(repo_path)
    print(repo_name)

    dbt_models_enriched_df.to_csv('../data/dbt_models_' + repo_name + '.csv', index=False)
    dbt_project_df.to_csv('../data/dbt_project_' + repo_name + '.csv', index=False)

    return dbt_models_enriched_df, dbt_project_df
        