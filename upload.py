from huggingface_hub import HfApi

api = HfApi()

api.upload_file(
    path_or_fileobj="/usr1/home/s125mdg54_09/tongshen/DELT/data/opencode_zigzag_l5.jsonl",
    path_in_repo="opencode_zigzag_l5.jsonl",  # <--- 重点：这里直接写原文件名！
    repo_id="stack0x0/opencodeInstruct",
    repo_type="dataset",
)
