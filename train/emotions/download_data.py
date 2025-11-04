"""
Download script for Emotions training data and tokenizer.

This script downloads:
1. tokenizer.json - Required for text tokenization
2. ru-goemotions dataset - Russian GoEmotions dataset for training
"""

import os
import subprocess
import sys
from pathlib import Path


# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "emotions"
DATA_DIR = PROJECT_ROOT / "train" / "emotions" / "data"

TOKENIZER_PATH = MODELS_DIR / "tokenizer.json"
TOKENIZER_URL = "https://drive.google.com/uc?id=1Atnpckju2lf31LU1li6HbGwIjp_pQRTK"


def download_tokenizer():
    """Download tokenizer.json from Google Drive."""
    print("=" * 60)
    print("Downloading tokenizer.json...")
    print("=" * 60)
    
    if TOKENIZER_PATH.exists():
        print(f"✅ Tokenizer already exists at {TOKENIZER_PATH}")
        return True
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        import gdown
        gdown.download(TOKENIZER_URL, str(TOKENIZER_PATH), quiet=False)
        print(f"✅ Tokenizer downloaded to {TOKENIZER_PATH}")
        return True
    except ImportError:
        print("❌ gdown not installed. Install it with: pip install gdown")
        print("\nAlternatively, download manually:")
        print(f"  URL: {TOKENIZER_URL}")
        print(f"  Save to: {TOKENIZER_PATH}")
        return False
    except Exception as e:
        print(f"❌ Error downloading tokenizer: {e}")
        print("\nTry manual download:")
        print(f"  URL: {TOKENIZER_URL}")
        print(f"  Save to: {TOKENIZER_PATH}")
        return False


def clone_dataset():
    """Clone ru-goemotions dataset from GitHub."""
    print("\n" + "=" * 60)
    print("Cloning ru-goemotions dataset...")
    print("=" * 60)
    
    dataset_path = DATA_DIR / "ru-goemotions"
    
    if dataset_path.exists():
        print(f"✅ Dataset already exists at {dataset_path}")
        return True
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        subprocess.run([
            "git", "clone",
            "https://github.com/searayeah/ru-goemotions.git",
            str(dataset_path)
        ], check=True)
        print(f"✅ Dataset cloned to {dataset_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error cloning dataset: {e}")
        print("\nTry manual clone:")
        print("  git clone https://github.com/searayeah/ru-goemotions.git")
        return False
    except FileNotFoundError:
        print("❌ git not found. Please install git first.")
        return False


def verify_setup():
    """Verify that all required files are present."""
    print("\n" + "=" * 60)
    print("Verifying setup...")
    print("=" * 60)
    
    checks = {
        "Tokenizer": TOKENIZER_PATH.exists(),
        "Dataset": (DATA_DIR / "ru-goemotions").exists(),
        "Emotion labels": (MODELS_DIR / "emotion_labels.json").exists(),
        "Model checkpoint": (MODELS_DIR / "ambient_director.pt").exists(),
    }
    
    all_good = True
    for name, exists in checks.items():
        status = "✅" if exists else "❌"
        print(f"{status} {name}")
        if not exists:
            all_good = False
    
    return all_good


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("Emotions Training Data Download Script")
    print("=" * 60)
    
    # Download tokenizer
    tokenizer_ok = download_tokenizer()
    
    # Clone dataset
    dataset_ok = clone_dataset()
    
    # Verify
    all_ok = verify_setup()
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ Setup complete! You can now train the model.")
        print("=" * 60)
        print("\nTo train:")
        print("  cd train/emotions")
        print("  python train.py")
    else:
        print("⚠️  Setup incomplete. Please resolve the issues above.")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()

