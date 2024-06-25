import os
import subprocess
import gdown
from dotenv import load_dotenv
load_dotenv()

def download_img_from_gdrive(gdrive_id, save_dir):
    output = f"{save_dir}/images.zip"
    print(f"Downloading images from Google Drive with id {gdrive_id} to {output}")
    gdown.download(id=gdrive_id, output=output)

    unzip_result = subprocess.run(
        ["unzip", output, "-d", save_dir],
        check=True,
        text=True,
        capture_output=True
    )
    print(unzip_result.stdout)

    delete_result = subprocess.run(
        ["rm", output],
        check=True,
        text=True,
        capture_output=True
    )
    print(delete_result.stdout)


def clone_hf_repo(repo_url, save_dir, download_img, label_encoder, token=None):
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
        print (f"Cloned {repo_url} to {save_dir}")

        if download_img is not None:
            if download_img["type"] == "gdrive":
                download_img_from_gdrive(download_img["id"], save_dir)
                print(f"Downloaded images from Google Drive to {save_dir}")
            else:
                print("Invalid download_img type")
        if label_encoder is not None:
            if label_encoder["idx2label"] is not None:
                download_img_from_gdrive(label_encoder["idx2label"], save_dir)
                print(f"Downloaded idx2label from Google Drive to {save_dir}")
            if label_encoder["label2idx"] is not None:
                download_img_from_gdrive(label_encoder["label2idx"], save_dir)
                print(f"Downloaded label2idx from Google Drive to {save_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr}")
    print("Done!")


if __name__ == "__main__":
    repo_url = [
        {
            "url": "https://huggingface.co/datasets/uitnlp/OpenViVQA-dataset",
            "save_dir": "OpenViVQA",
            "download_img": None
        },
        {
            "url": "https://github.com/kh4nh12/ViVQA.git",
            "save_dir": "ViVQA",
            "download_img": {
                "type": "gdrive",
                "id": "18AwVsUOdrEXahiPIE5fAQD7z_8ttDtNv",
            },
            "label_encoder": {
                "idx2label": "16r4d6KToIyYxkmyUPtLO-Vt_ywL8BBOU",
                "label2idx": "13I3TN03EgegC9f8s8At9ml_HTKVx88a0"
            }

        },
    ]

    for repo in repo_url:
        clone_hf_repo(repo["url"], 
                      repo["save_dir"], 
                      download_img=repo["download_img"], 
                      label_encoder=repo["label_encoder"]
                      )

