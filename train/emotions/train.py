"""
Emotions Classifier Training Script

This script trains the AmbientDirector LSTM-based model for emotion
classification from text.
"""

import os
import random
import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, Adam
from tokenizers import Tokenizer
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

# Import model architecture from the main module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from modules.emotions.model import AmbientDirector

from config import TrainConfig, DEFAULT_TRAIN_CONFIG


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class EmotionDataset(Dataset):
    """Dataset for emotion classification."""
    
    def __init__(self, texts, labels, tokenizer, max_len=64, padding_idx=1):
        """
        Initialize dataset.
        
        Args:
            texts: List of text strings
            labels: List of emotion labels (0-27)
            tokenizer: Tokenizer instance
            max_len: Maximum sequence length
            padding_idx: Index for padding tokens
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.padding_idx = padding_idx
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer.encode(text)
        
        # Truncate and pad
        ids = encoding.ids[:self.max_len]
        ids += [self.padding_idx] * (self.max_len - len(ids))
        
        return torch.tensor(ids), torch.tensor(self.labels[idx], dtype=torch.long)


def load_goemotions(csv_path: str) -> pd.DataFrame:
    """
    Load and prepare the ru-goemotions dataset.
    
    Args:
        csv_path: Path to the CSV file
    
    Returns:
        DataFrame with 'text' and 'label' columns
    """
    df = pd.read_csv(csv_path)
    
    # Determine text column name
    text_col = "ru_text" if "ru_text" in df.columns else "text"
    
    def parse_label(val):
        """Parse label from various formats."""
        try:
            parsed = ast.literal_eval(val) if isinstance(val, str) else val
            if isinstance(parsed, list) and parsed:
                # If multiple labels, take the first one
                # Note: This is single-label classification
                return int(parsed[0])
            elif isinstance(parsed, int):
                return parsed
        except Exception:
            pass
        return 0  # Default to admiration
    
    df["label"] = df["labels"].apply(parse_label)
    df = df[[text_col, "label"]].rename(columns={text_col: "text"}).dropna()
    
    return df


def train_epoch(model, dataloader, optimizer, criterion, device, grad_clip_norm=None):
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        grad_clip_norm: Gradient clipping norm (optional)
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        
        if grad_clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model on validation/test set.
    
    Args:
        model: Model to evaluate
        dataloader: Evaluation dataloader
        criterion: Loss function
        device: Device to evaluate on
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            logits = model(inputs)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            predictions = logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def train(
    cfg: TrainConfig,
    train_csv: str,
    tokenizer_path: str,
    output_dir: Path,
    val_csv: str = None,
):
    """
    Main training function.
    
    Args:
        cfg: Training configuration
        train_csv: Path to training CSV
        tokenizer_path: Path to tokenizer.json
        output_dir: Output directory for checkpoints
        val_csv: Path to validation CSV (optional)
    """
    # Setup
    set_seed(cfg.seed)
    device = get_device()
    print(f"Using device: {device}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    cfg.vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulary size: {cfg.vocab_size}")
    
    # Load data
    print(f"Loading dataset from {train_csv}...")
    df = load_goemotions(train_csv)
    print(f"Loaded {len(df)} samples")
    print(f"Number of classes: {df['label'].nunique()}")
    
    # Split into train/val if val_csv not provided
    if val_csv is None:
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(
            df, test_size=cfg.val_size, random_state=cfg.seed, stratify=df['label']
        )
        print(f"Split: {len(train_df)} train, {len(val_df)} val")
    else:
        train_df = df
        val_df = load_goemotions(val_csv)
        print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Create datasets
    train_dataset = EmotionDataset(
        train_df['text'].tolist(),
        train_df['label'].tolist(),
        tokenizer,
        max_len=cfg.max_length,
        padding_idx=cfg.padding_idx
    )
    
    val_dataset = EmotionDataset(
        val_df['text'].tolist(),
        val_df['label'].tolist(),
        tokenizer,
        max_len=cfg.max_length,
        padding_idx=cfg.padding_idx
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    print("Creating model...")
    model = AmbientDirector(
        vocab_size=cfg.vocab_size,
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        num_classes=cfg.num_classes,
        dropout_p=cfg.dropout_p,
        padding_idx=cfg.padding_idx
    )
    model.to(device)
    print(f"Model parameters: {model.get_num_parameters():,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if cfg.optimizer.lower() == "adamw":
        optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    else:
        optimizer = Adam(model.parameters(), lr=cfg.learning_rate)
    
    # Scheduler (optional)
    scheduler = None
    if cfg.use_scheduler:
        num_training_steps = len(train_loader) * cfg.num_epochs
        num_warmup_steps = int(cfg.warmup_ratio * num_training_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    
    # Training loop
    print(f"\nStarting training for {cfg.num_epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(1, cfg.num_epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.num_epochs}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, cfg.grad_clip_norm
        )
        
        if scheduler:
            scheduler.step()
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_dir / "ambient_director.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Saved best model to {checkpoint_path}")
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train Emotions Classifier")
    parser.add_argument(
        '--train-csv', type=str, required=True,
        help='Path to training CSV file'
    )
    parser.add_argument(
        '--val-csv', type=str, default=None,
        help='Path to validation CSV file (optional)'
    )
    parser.add_argument(
        '--tokenizer', type=str, required=True,
        help='Path to tokenizer.json file'
    )
    parser.add_argument(
        '--output-dir', type=str, default='./output',
        help='Output directory for checkpoints'
    )
    
    # Allow overriding config parameters
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    
    args = parser.parse_args()
    
    # Create config
    cfg = TrainConfig()
    if args.epochs:
        cfg.num_epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.learning_rate = args.lr
    
    output_dir = Path(args.output_dir)
    
    # Train
    train(cfg, args.train_csv, args.tokenizer, output_dir, args.val_csv)


if __name__ == '__main__':
    main()

