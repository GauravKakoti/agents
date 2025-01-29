from huggingface_hub import snapshot_download
import os

weights_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "src/"))
weights_dir += "/weights/omniparser/"
if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)
commit_hash = "f5a160e5c325bf4fbf26723794f577587b170435"
repo_id = "microsoft/OmniParser"
snapshot_download(repo_id=repo_id, revision=commit_hash, local_dir=weights_dir)