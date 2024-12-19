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
    repo_df["file_name"] = repo_df["path"].apply(lambda x: x.split("/")[-1])
    repo_df["extension"] = repo_df["path"].apply(lambda x: "." + x.split(".")[-1])
    return repo_df