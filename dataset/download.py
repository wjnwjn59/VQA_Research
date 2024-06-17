import os
import subprocess
from dotenv import load_dotenv
load_dotenv()


def clone_hf_repo(repo_url, save_dir, token=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory {save_dir}")
    else:
        print(f"Repo {save_dir} already exists")
        return

    try:
        if token:
            repo_url = repo_url.replace("https://", f"https://{token}@")
        
        result = subprocess.run(
            ["git", "clone", repo_url, save_dir],
            check=True,
            text=True,
            capture_output=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr}")


if __name__ == "__main__":
    repo_url = [
        {
            "url": "https://huggingface.co/datasets/uitnlp/OpenViVQA-dataset",
            "save_dir": "OpenViVQA"
        },
        {
            "url": "https://github.com/kh4nh12/ViVQA.git",
            "save_dir": "ViVQA"
        },
    ]

    for repo in repo_url:
        clone_hf_repo(repo["url"], repo["save_dir"])

