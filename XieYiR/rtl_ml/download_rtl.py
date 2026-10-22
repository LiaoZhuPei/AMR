from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="TrevTron/rtl-ml-dataset",
    repo_type="dataset",
    local_dir="datasets_validated",
)