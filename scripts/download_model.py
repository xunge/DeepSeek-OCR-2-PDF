#!/usr/bin/env python3
import os
from pathlib import Path

from huggingface_hub import snapshot_download

model_name = 'Qwen3-VL-2B-Instruct'
author_name = 'Qwen'

REPO_ID = os.getenv("MODEL_REPO_ID", f"{author_name}/{model_name}")
TARGET_DIR = Path(
    os.getenv("MODEL_DIR", f"models/{model_name}")
).expanduser().resolve()


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=TARGET_DIR,
    )
    print(f"Downloaded {REPO_ID} to {TARGET_DIR}")


if __name__ == "__main__":
    main()
