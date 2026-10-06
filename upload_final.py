#!/usr/bin/env python3
"""Upload data/final/ folder to HuggingFace"""

from pathlib import Path
from huggingface_hub import HfApi
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Config
REPO_ID = "raj270898/youtube-video-dataset"
TOKEN = os.getenv("HF_TOKEN")
FINAL_DIR = Path("data/final")

if not TOKEN:
    print("❌ HF_TOKEN not found in environment variables!")
    print("Please set HF_TOKEN in your .env file")
    sys.exit(1)

def main():
    if not FINAL_DIR.exists():
        print(f"❌ {FINAL_DIR} does not exist!")
        sys.exit(1)
    
    print(f"📦 Uploading {FINAL_DIR} to {REPO_ID}...")
    
    api = HfApi()
    
    # Upload entire folder
    api.upload_folder(
        folder_path=str(FINAL_DIR),
        repo_id=REPO_ID,
        repo_type="dataset",
        token=TOKEN,
        path_in_repo="data/final",
        commit_message="Upload final dataset (177 videos)",
    )
    
    print("✅ Upload complete!")

if __name__ == "__main__":
    main()
