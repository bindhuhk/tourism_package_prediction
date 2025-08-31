from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

# Initialize Hugging Face API with token
api = HfApi(token=os.getenv("HF_TOKEN"))

# Set repository details
repo_id = "hkbindhu/Tourism-Package-Prediction"  # Update with your Hugging Face username/repo name
repo_type = "dataset"
folder_path = "tourism_project/data"

# Check authentication
try:
    user = api.whoami()
    print("✅ Successfully authenticated as:", user["name"])
except Exception as e:
    print("❌ Authentication failed:", e)

# Step 1: Check if the space exists or create it
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

# Upload the data folder to Hugging Face dataset repo
api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type=repo_type,
    path_in_repo="data"
)
print("✅ File uploaded!")
