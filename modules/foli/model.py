"""
Foli Multi-Channel Classifier Model Architecture

This module contains the neural network architecture for classifying
background sounds (foley) into three channels (ch1, ch2, ch3).
"""

import torch
from torch import nn
from transformers import AutoModel


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Apply mean pooling to transformer outputs.
    
    Args:
        last_hidden_state: Transformer output, shape (batch, seq_len, hidden_dim)
        attention_mask: Attention mask, shape (batch, seq_len)
    
    Returns:
        Pooled embeddings, shape (batch, hidden_dim)
    """
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


class MultiHeadClassifier(nn.Module):
    """
    Multi-head classifier for foley sound classification.
    
    Uses a pre-trained transformer as encoder and three separate
    classification heads for three channels (ch1, ch2, ch3).
    """
    
    def __init__(
        self,
        base_name: str,
        label_spaces: dict,
        dropout_p: float = 0.1,
    ):
        """
        Initialize the multi-head classifier.
        
        Args:
            base_name: Name of the pre-trained transformer model
            label_spaces: Dictionary with label information for each channel
                         Format: {'ch1': {'classes': [...], 'label2id': {...}, 'id2label': {...}}, ...}
            dropout_p: Dropout probability
        """
        super().__init__()
        
        # Load base transformer model
        self.base = AutoModel.from_pretrained(base_name)
        
        # Get hidden size from model config
        hidden = getattr(self.base.config, 'hidden_size', None) or getattr(
            self.base.config, 'dim', 256
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_p)
        
        # Create classification heads for each channel
        self.classifiers = nn.ModuleDict()
        for ch in ['ch1', 'ch2', 'ch3']:
            num_classes = len(label_spaces[ch]['classes'])
            self.classifiers[ch] = nn.Linear(hidden, num_classes)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
    ):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs, shape (batch, seq_len)
            attention_mask: Attention mask, shape (batch, seq_len)
            token_type_ids: Token type IDs (optional), shape (batch, seq_len)
        
        Returns:
            Dictionary with logits for each channel:
            {
                'logits': {
                    'ch1': tensor of shape (batch, num_classes_ch1),
                    'ch2': tensor of shape (batch, num_classes_ch2),
                    'ch3': tensor of shape (batch, num_classes_ch3),
                }
            }
        """
        # Get transformer outputs
        base_outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Pool the hidden states
        last_hidden = base_outputs.last_hidden_state
        pooled = mean_pooling(last_hidden, attention_mask)
        pooled = self.dropout(pooled)
        
        # Get logits from each classification head
        logits = {
            'ch1': self.classifiers['ch1'](pooled),
            'ch2': self.classifiers['ch2'](pooled),
            'ch3': self.classifiers['ch3'](pooled),
        }
        
        return {'logits': logits}

