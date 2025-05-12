import git
from pathlib import Path
from app.utils import read_file  # Assuming you have a utility to read generic files

def clone_and_extract_all_files(repo_url: str, branch: str = "main") -> list[str]:
    """
    Clones the given GitHub repository, extracts all files (any type), 
    and returns the list of documents as raw text.
    """
    repo_path = Path("repo_clone")
    if repo_path.exists():
        import shutil
        shutil.rmtree(repo_path)
    
    # Clone the repository
    git.Repo.clone_from(repo_url, repo_path, branch=branch)

    docs = []
    # Iterate through all files in the repo and read them
    for file in repo_path.rglob("*"):  # `rglob("*")` gets all files recursively
        if file.is_file():  # Only process actual files (not directories)
            docs.append(read_file(file))  # Assuming a generic file reader in utils.py
    return docs
