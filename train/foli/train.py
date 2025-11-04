"""
Foli Classifier Training Script

This script trains a multi-channel classifier for foley sound classification
based on text descriptions.
"""

import os
import json
import math
import random
import argparse
from pathlib import Path
from contextlib import nullcontext
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
)

# Import model architecture from the main module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from modules.foli.model import MultiHeadClassifier

from config import TrainConfig, DEFAULT_TRAIN_CONFIG


# Set environment variables
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("WANDB_MODE", "offline")


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """Auto-detect the best available device."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


class FoliDataset(Dataset):
    """Dataset for Foli classification."""
    
    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int):
        """
        Initialize dataset.
        
        Args:
            frame: DataFrame with columns: text, ch1_id, ch2_id, ch3_id
            tokenizer: Tokenizer for text encoding
            max_length: Maximum sequence length
        """
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.frame)
    
    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        enc = self.tokenizer(
            str(row['text']).lower(),
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )
        item = {
            'input_ids': enc['input_ids'],
            'attention_mask': enc['attention_mask'],
            'labels_ch1': int(row['ch1_id']),
            'labels_ch2': int(row['ch2_id']),
            'labels_ch3': int(row['ch3_id']),
        }
        return item


class CollateWithLabels:
    """Custom collator that handles labels along with text."""
    
    def __init__(self, tokenizer, padding=True):
        self.base = DataCollatorWithPadding(tokenizer, padding=padding)
    
    def __call__(self, features):
        labels_ch1 = torch.tensor([f['labels_ch1'] for f in features], dtype=torch.long)
        labels_ch2 = torch.tensor([f['labels_ch2'] for f in features], dtype=torch.long)
        labels_ch3 = torch.tensor([f['labels_ch3'] for f in features], dtype=torch.long)
        
        batch = self.base([
            {k: v for k, v in f.items() if k in ['input_ids', 'attention_mask']}
            for f in features
        ])
        
        batch.update({
            'labels_ch1': labels_ch1,
            'labels_ch2': labels_ch2,
            'labels_ch3': labels_ch3,
        })
        return batch


def load_label_spaces(classes_map_csv: str) -> dict:
    """
    Load and prepare label spaces from classes map CSV.
    
    Args:
        classes_map_csv: Path to classes_map.csv file
    
    Returns:
        Dictionary with label spaces for each channel
    """
    cm_df = pd.read_csv(classes_map_csv, sep=';')
    assert {'class', 'ch1', 'ch2', 'ch3'}.issubset(cm_df.columns), \
        "classes_map.csv must have columns: class;ch1;ch2;ch3"
    
    allowed_by_ch = {}
    for ch in ['ch1', 'ch2', 'ch3']:
        allowed_by_ch[ch] = cm_df.loc[cm_df[ch].str.upper() == 'Y', 'class'].tolist()
        if 'Silence' not in allowed_by_ch[ch]:
            allowed_by_ch[ch].append('Silence')
    
    label_spaces = {}
    for ch in ['ch1', 'ch2', 'ch3']:
        classes = sorted(allowed_by_ch[ch])
        label2id = {c: i for i, c in enumerate(classes)}
        id2label = {i: c for c, i in label2id.items()}
        label_spaces[ch] = {
            'classes': classes,
            'label2id': label2id,
            'id2label': id2label,
        }
    
    return label_spaces


def load_and_prepare_data(
    data_csvs: List[str],
    label_spaces: dict,
    seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and split dataset into train/val/test.
    
    Args:
        data_csvs: List of paths to dataset CSV files
        label_spaces: Label spaces dictionary
        seed: Random seed
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    def read_any_csv(path: str) -> pd.DataFrame:
        return pd.read_csv(path, sep=None, engine='python')
    
    raw_frames = [read_any_csv(p) for p in data_csvs]
    df = pd.concat(raw_frames, ignore_index=True)
    
    required_cols = ['text', 'ch1', 'ch2', 'ch3']
    missing = [c for c in required_cols if c not in df.columns]
    assert not missing, f"Dataset is missing columns: {missing}"
    
    df = df.dropna(subset=['text']).copy()
    
    # Map labels to IDs
    for ch in ['ch1', 'ch2', 'ch3']:
        allowed = set(label_spaces[ch]['classes'])
        df[ch] = df[ch].apply(lambda x: x if x in allowed else 'Silence')
        df[f'{ch}_id'] = df[ch].map(label_spaces[ch]['label2id'])
    
    # Split data
    stratify_col = df['ch1_id'] if df['ch1_id'].nunique() > 1 else None
    
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=stratify_col
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=seed,
        stratify=(temp_df['ch1_id'] if temp_df['ch1_id'].nunique() > 1 else None)
    )
    
    return train_df, val_df, test_df


def effective_weights_all(
    frame: pd.DataFrame,
    ch: str,
    label_spaces: dict,
    beta: float = 0.999,
    min_w: float = 0.5,
    max_w: float = 3.0
) -> np.ndarray:
    """
    Calculate effective class weights using effective number of samples.
    
    Args:
        frame: Training dataframe
        ch: Channel name (ch1, ch2, or ch3)
        label_spaces: Label spaces dictionary
        beta: Beta parameter for effective number
        min_w: Minimum weight
        max_w: Maximum weight
    
    Returns:
        Array of class weights
    """
    num_classes = len(label_spaces[ch]['classes'])
    counts = frame[f'{ch}_id'].value_counts().reindex(
        range(num_classes), fill_value=0
    ).astype(float).to_numpy()
    counts = np.clip(counts, 1.0, None)
    
    eff = (1.0 - beta) / (1.0 - np.power(beta, counts))
    eff = eff / eff.mean()
    eff = np.clip(eff, min_w, max_w).astype(np.float32)
    
    return eff


def class_priors(frame: pd.DataFrame, ch: str, label_spaces: dict) -> np.ndarray:
    """
    Calculate log prior probabilities for each class.
    
    Args:
        frame: Training dataframe
        ch: Channel name
        label_spaces: Label spaces dictionary
    
    Returns:
        Array of log prior probabilities
    """
    num_classes = len(label_spaces[ch]['classes'])
    counts = frame[f'{ch}_id'].value_counts().reindex(
        range(num_classes), fill_value=0
    ).astype(float).to_numpy()
    counts = counts + 1.0
    p = counts / counts.sum()
    return np.log(p).astype(np.float32)


class MultiHeadClassifierWithLoss(MultiHeadClassifier):
    """Extended model with loss computation for training."""
    
    def __init__(
        self,
        base_name: str,
        label_spaces: dict,
        dropout_p: float = 0.1,
        class_weights: dict = None,
        label_smoothing: float = 0.0,
        head_weights: dict = None,
        prior_log: dict = None,
        tau: dict = None,
        cls_loss_weight: float = 1.0,
    ):
        super().__init__(base_name, label_spaces, dropout_p)
        
        self.cls_loss_weight = cls_loss_weight
        self.head_weights = head_weights or {'ch1': 1.0, 'ch2': 1.0, 'ch3': 1.0}
        self.prior_log = prior_log
        self.tau = tau
        
        # Create loss functions with class weights
        self.loss_ce = nn.ModuleDict()
        for ch in ['ch1', 'ch2', 'ch3']:
            w = None
            if class_weights is not None and ch in class_weights:
                w = torch.tensor(class_weights[ch], dtype=torch.float32)
            self.loss_ce[ch] = nn.CrossEntropyLoss(weight=w, label_smoothing=label_smoothing)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels_ch1=None,
        labels_ch2=None,
        labels_ch3=None,
    ):
        # Get logits from parent class
        outputs = super().forward(input_ids, attention_mask, token_type_ids)
        logits = outputs['logits']
        
        loss = None
        if labels_ch1 is not None:
            # Apply prior adjustment if specified
            if self.prior_log is not None and self.tau is not None:
                logits_for_loss = {
                    'ch1': logits['ch1'] + float(self.tau.get('ch1', 0.0)) * self.prior_log['ch1'].to(logits['ch1'].device),
                    'ch2': logits['ch2'] + float(self.tau.get('ch2', 0.0)) * self.prior_log['ch2'].to(logits['ch2'].device),
                    'ch3': logits['ch3'] + float(self.tau.get('ch3', 0.0)) * self.prior_log['ch3'].to(logits['ch3'].device),
                }
            else:
                logits_for_loss = logits
            
            # Compute losses
            l1 = self.loss_ce['ch1'](logits_for_loss['ch1'], labels_ch1)
            l2 = self.loss_ce['ch2'](logits_for_loss['ch2'], labels_ch2)
            l3 = self.loss_ce['ch3'](logits_for_loss['ch3'], labels_ch3)
            
            # Weighted sum
            hw = self.head_weights
            denom = float(hw['ch1'] + hw['ch2'] + hw['ch3'])
            loss_cls = (hw['ch1'] * l1 + hw['ch2'] * l2 + hw['ch3'] * l3) / denom
            loss = self.cls_loss_weight * loss_cls
        
        return {'loss': loss, 'logits': logits}


def evaluate(model, loader, device, use_amp=False):
    """
    Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        loader: DataLoader
        device: Device to run on
        use_amp: Whether to use mixed precision
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    losses = []
    all_true = {'ch1': [], 'ch2': [], 'ch3': []}
    all_pred = {'ch1': [], 'ch2': [], 'ch3': []}
    
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            
            with (
                torch.autocast(device_type='cuda', dtype=torch.float16)
                if use_amp else nullcontext()
            ):
                out = model(**batch)
            
            if out['loss'] is not None:
                losses.append(out['loss'].detach().float().item())
            
            for ch in ['ch1', 'ch2', 'ch3']:
                probs = torch.softmax(out['logits'][ch], dim=-1)
                preds = probs.argmax(dim=-1).detach().cpu().numpy()
                all_pred[ch].extend(preds.tolist())
                all_true[ch].extend(batch[f'labels_{ch}'].detach().cpu().numpy().tolist())
    
    metrics = {}
    metrics['loss'] = float(np.mean(losses)) if losses else 0.0
    
    for ch in ['ch1', 'ch2', 'ch3']:
        acc = accuracy_score(all_true[ch], all_pred[ch]) if len(all_true[ch]) else 0.0
        f1m = f1_score(all_true[ch], all_pred[ch], average='macro') if len(set(all_true[ch])) > 1 else 0.0
        metrics[f'acc_{ch}'] = acc
        metrics[f'f1_{ch}'] = f1m
    
    return metrics


def train(
    cfg: TrainConfig,
    data_csvs: List[str],
    classes_map_csv: str,
    output_dir: Path,
):
    """
    Main training function.
    
    Args:
        cfg: Training configuration
        data_csvs: List of dataset CSV paths
        classes_map_csv: Path to classes_map.csv
        output_dir: Output directory for artifacts
    """
    # Setup
    set_seed(cfg.seed)
    device = get_device()
    print(f"Using device: {device}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load label spaces
    label_spaces = load_label_spaces(classes_map_csv)
    print(f"Label spaces: {{{', '.join([f'{ch}: {len(label_spaces[ch]['classes'])}' for ch in label_spaces])}}}")
    
    # Save label spaces
    with open(output_dir / 'label_spaces.json', 'w', encoding='utf-8') as f:
        json.dump(label_spaces, f, ensure_ascii=False, indent=2)
    
    # Load and prepare data
    train_df, val_df, test_df = load_and_prepare_data(data_csvs, label_spaces, cfg.seed)
    print(f"Dataset sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # Create datasets
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    train_ds = FoliDataset(train_df, tokenizer, cfg.max_length)
    val_ds = FoliDataset(val_df, tokenizer, cfg.max_length)
    test_ds = FoliDataset(test_df, tokenizer, cfg.max_length)
    
    collate_fn = CollateWithLabels(tokenizer)
    
    pin_mem = (device == 'cuda')
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train_batch_size, shuffle=True,
        collate_fn=collate_fn, pin_memory=pin_mem
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.eval_batch_size, shuffle=False,
        collate_fn=collate_fn, pin_memory=pin_mem
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.eval_batch_size, shuffle=False,
        collate_fn=collate_fn, pin_memory=pin_mem
    )
    
    # Calculate class weights if enabled
    class_weights = None
    if cfg.use_class_weights:
        class_weights = {
            ch: effective_weights_all(
                train_df, ch, label_spaces,
                beta=cfg.class_weight_beta,
                min_w=cfg.class_weight_min,
                max_w=cfg.class_weight_max
            ).tolist()
            for ch in ['ch1', 'ch2', 'ch3']
        }
        print("Class weights calculated")
    
    # Calculate priors if enabled
    prior_log = None
    if cfg.use_prior_adjustment:
        prior_log = {
            ch: torch.tensor(class_priors(train_df, ch, label_spaces))
            for ch in ['ch1', 'ch2', 'ch3']
        }
        print("Prior distributions calculated")
    
    # Create model
    model = MultiHeadClassifierWithLoss(
        cfg.model_name,
        label_spaces,
        dropout_p=cfg.dropout_p,
        class_weights=class_weights,
        label_smoothing=cfg.label_smoothing,
        head_weights=cfg.head_weights,
        prior_log=prior_log,
        tau=cfg.tau,
        cls_loss_weight=cfg.cls_loss_weight,
    )
    model.to(device)
    
    if cfg.compile_model and device != 'mps':
        try:
            model = torch.compile(model, mode='default', fullgraph=False, dynamic=True)
            print("Model compiled with torch.compile")
        except Exception as e:
            print(f"torch.compile disabled: {e}")
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer and scheduler
    num_training_steps = math.ceil(
        len(train_loader) / max(1, cfg.gradient_accumulation_steps)
    ) * cfg.num_epochs
    warmup_steps = int(cfg.warmup_ratio * num_training_steps)
    
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
    )
    
    # Setup mixed precision
    use_amp = cfg.use_fp16 and (device == 'cuda')
    if device == 'cuda':
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    else:
        class _NullScaler:
            def scale(self, x): return x
            def step(self, opt): opt.step()
            def update(self): pass
        scaler = _NullScaler()
    
    # Setup logging
    try:
        import wandb
        run = wandb.init(
            project="foli-multichannel",
            config=cfg.to_dict(),
            mode=os.environ.get("WANDB_MODE", "offline")
        )
    except ImportError:
        print("wandb not available, skipping W&B logging")
        run = None
    
    writer = SummaryWriter(log_dir=str(output_dir / 'tb'))
    
    # Training loop
    best_val_loss = float('inf')
    patience_left = cfg.patience
    step = 0
    
    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        
        for batch in train_loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            
            with (
                torch.autocast(device_type='cuda', dtype=torch.float16)
                if use_amp else nullcontext()
            ):
                out = model(**batch)
                loss = out['loss'] / cfg.gradient_accumulation_steps
            
            if device == 'cuda':
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                if cfg.grad_clip_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                
                if device == 'cuda':
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            
            epoch_loss += loss.detach().float().item()
            step += 1
            
            if step % cfg.log_every_n_steps == 0:
                log_dict = {
                    'train/loss': loss.item(),
                    'lr': scheduler.get_last_lr()[0],
                    'epoch': epoch,
                    'step': step
                }
                if run:
                    wandb.log(log_dict)
                writer.add_scalar('train/loss', loss.item(), step)
        
        # Validation
        val_metrics = evaluate(model, val_loader, device, use_amp)
        
        log_dict = {f'val/{k}': v for k, v in val_metrics.items()}
        log_dict['epoch'] = epoch
        if run:
            wandb.log(log_dict)
        writer.add_scalar('val/loss', val_metrics['loss'], epoch)
        
        print(
            f"Epoch {epoch}: "
            f"train_loss={epoch_loss/len(train_loader):.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"acc_ch1={val_metrics['acc_ch1']:.3f} | f1_ch1={val_metrics['f1_ch1']:.3f} | "
            f"acc_ch2={val_metrics['acc_ch2']:.3f} | f1_ch2={val_metrics['f1_ch2']:.3f} | "
            f"acc_ch3={val_metrics['acc_ch3']:.3f} | f1_ch3={val_metrics['f1_ch3']:.3f}"
        )
        
        # Save best checkpoint
        if val_metrics['loss'] + 1e-6 < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_left = cfg.patience
            
            ckpt_dir = output_dir / 'best'
            ckpt_dir.mkdir(exist_ok=True)
            
            # Save only model weights (not loss modules)
            state = model.state_dict()
            state = {k: v for k, v in state.items() if not k.startswith('loss_ce.')}
            
            torch.save({
                'model_state_dict': state,
                'cfg': cfg.to_dict(),
                'label_spaces': label_spaces,
            }, ckpt_dir / 'checkpoint.pt')
            
            print("✓ Saved new best checkpoint")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered")
                break
    
    # Test evaluation
    print("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader, device, use_amp)
    
    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    print(f"Test metrics: {test_metrics}")
    
    # Cleanup
    if run:
        run.finish()
    writer.flush()
    writer.close()
    
    print(f"\nTraining complete! Artifacts saved to {output_dir}")


def export_to_onnx(checkpoint_path: Path, output_path: Path, config: TrainConfig):
    """
    Export model encoder to ONNX format.
    
    Args:
        checkpoint_path: Path to model checkpoint
        output_path: Path to save ONNX model
        config: Training configuration
    """
    print("\nExporting to ONNX...")
    
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("onnx or onnxruntime not available, skipping ONNX export")
        return
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    label_spaces = checkpoint['label_spaces']
    
    # Create model
    model = MultiHeadClassifier(
        config.model_name,
        label_spaces,
        dropout_p=config.dropout_p,
    )
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    # Prepare dummy input
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    dummy = tokenizer("пример текста", return_tensors='pt', max_length=config.max_length)
    
    # Export encoder only
    base = model.base.cpu()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        base,
        (dummy['input_ids'], dummy['attention_mask']),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "last_hidden_state": {0: "batch", 1: "seq"},
        },
        opset_version=17,
    )
    
    print(f"✓ Exported ONNX to {output_path}")
    
    # Verify ONNX model
    sess = ort.InferenceSession(str(output_path), providers=['CPUExecutionProvider'])
    ort_inputs = {
        "input_ids": dummy['input_ids'].numpy(),
        "attention_mask": dummy['attention_mask'].numpy()
    }
    ort_outs = sess.run(None, ort_inputs)
    print(f"✓ ONNX runtime output shape: {ort_outs[0].shape}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train Foli Classifier")
    parser.add_argument(
        '--data', type=str, nargs='+', required=True,
        help='Path(s) to dataset CSV file(s)'
    )
    parser.add_argument(
        '--classes-map', type=str, required=True,
        help='Path to classes_map.csv file'
    )
    parser.add_argument(
        '--output-dir', type=str, default='./artifacts',
        help='Output directory for artifacts'
    )
    parser.add_argument(
        '--export-onnx', action='store_true',
        help='Export model to ONNX format after training'
    )
    
    # Allow overriding config parameters
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Training batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    
    args = parser.parse_args()
    
    # Create config
    cfg = TrainConfig()
    if args.epochs:
        cfg.num_epochs = args.epochs
    if args.batch_size:
        cfg.train_batch_size = args.batch_size
    if args.lr:
        cfg.learning_rate = args.lr
    
    output_dir = Path(args.output_dir)
    
    # Train
    train(cfg, args.data, args.classes_map, output_dir)
    
    # Export to ONNX if requested
    if args.export_onnx:
        checkpoint_path = output_dir / 'best' / 'checkpoint.pt'
        onnx_path = output_dir / 'onnx' / 'encoder.onnx'
        export_to_onnx(checkpoint_path, onnx_path, cfg)


if __name__ == '__main__':
    main()

