"""
Emotions Classifier Model Architecture

This module contains the AmbientDirector LSTM-based architecture for
emotion classification from text.
"""

import torch
import torch.nn as nn


class AmbientDirector(nn.Module):
    """
    LSTM-based emotion classifier.
    
    Architecture:
    - Embedding layer
    - Bidirectional LSTM encoder
    - Dropout for regularization
    - Linear classification head
    
    The model classifies text into 28 emotion categories from the GoEmotions dataset.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        num_classes: int = 28,
        dropout_p: float = 0.3,
        padding_idx: int = 1,
    ):
        """
        Initialize the AmbientDirector model.
        
        Args:
            vocab_size: Size of the vocabulary
            embed_dim: Dimension of embedding layer
            hidden_dim: Hidden dimension of LSTM
            num_classes: Number of emotion classes (default: 28)
            dropout_p: Dropout probability
            padding_idx: Index used for padding tokens
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Embedding layer with padding
        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=padding_idx
        )
        
        # Bidirectional LSTM encoder
        self.encoder = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_p)
        
        # Emotion classification head
        # BiLSTM outputs hidden_dim * 2 (forward + backward)
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input token IDs, shape (batch, seq_len)
        
        Returns:
            Logits for each emotion class, shape (batch, num_classes)
        """
        # Embedding: (batch, seq_len) -> (batch, seq_len, embed_dim)
        x = self.embedding(x)
        
        # LSTM encoding: (batch, seq_len, embed_dim) -> (batch, seq_len, hidden_dim * 2)
        x, _ = self.encoder(x)
        
        # Take the last hidden state (final output of sequence)
        x = x[:, -1, :]  # (batch, hidden_dim * 2)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Classification head: (batch, hidden_dim * 2) -> (batch, num_classes)
        logits = self.emotion_head(x)
        
        return logits
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

