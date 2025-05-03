"""
NLP models for TAuto.

This module provides implementations and wrappers for common NLP models.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Union, Callable

from tauto.models.registry import register_model
from tauto.utils import get_logger

logger = get_logger(__name__)

# Try to import transformers, but provide fallbacks if not available
try:
    import transformers
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available. Limited NLP models will be provided.")


class SimpleRNN(nn.Module):
    """
    Simple RNN model for text classification.
    
    This model uses an embedding layer followed by an RNN (GRU by default)
    and a linear output layer.
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 100,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.2,
        rnn_type: str = "gru",
        bidirectional: bool = True,
    ):
        """
        Initialize the model.
        
        Args:
            vocab_size: Size of the vocabulary
            embed_dim: Dimension of the embeddings
            hidden_dim: Dimension of the hidden states
            num_layers: Number of RNN layers
            num_classes: Number of output classes
            dropout: Dropout probability
            rnn_type: Type of RNN ('lstm', 'gru', or 'rnn')
            bidirectional: Whether to use bidirectional RNN
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Choose RNN type
        rnn_cls = nn.LSTM if rnn_type.lower() == "lstm" else nn.GRU if rnn_type.lower() == "gru" else nn.RNN
        
        self.rnn = rnn_cls(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # Factor of 2 for bidirectional
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.fc = nn.Linear(fc_input_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, lengths=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, L) containing token indices
            lengths: Optional tensor of shape (B,) containing sequence lengths
            
        Returns:
            Tensor: Output tensor of shape (B, num_classes)
        """
        # Embed the input
        embedded = self.dropout(self.embedding(x))  # (B, L, embed_dim)
        
        # Pack padded sequence if lengths are provided
        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # Pass through RNN
        output, hidden = self.rnn(embedded)
        
        # Unpack if packed
        if lengths is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        # Use different strategies to get the final representation
        if isinstance(hidden, tuple):  # LSTM returns (h_n, c_n)
            hidden = hidden[0]
        
        # Take the last non-padded entry from each sequence
        if lengths is not None:
            # Extract the last hidden state for each sequence
            batch_size = x.size(0)
            
            # For bidirectional RNN, we need to concatenate the forward and backward states
            if self.rnn.bidirectional:
                hidden = hidden.view(self.rnn.num_layers, 2, batch_size, -1)
                last_hidden = hidden[-1].transpose(0, 1).contiguous().view(batch_size, -1)
            else:
                last_hidden = hidden[-1]
            
            final_output = last_hidden
        else:
            # Use the last token if no lengths are provided (all sequences same length)
            if self.rnn.bidirectional:
                # For bidirectional, concatenate forward and backward last hidden states
                forward, backward = output[:, -1, :self.rnn.hidden_size], output[:, 0, self.rnn.hidden_size:]
                final_output = torch.cat([forward, backward], dim=1)
            else:
                final_output = output[:, -1, :]
        
        # Apply dropout and pass through linear layer
        final_output = self.dropout(final_output)
        logits = self.fc(final_output)
        
        return logits


# Register the simple RNN model
register_model(
    name="simple_rnn",
    architecture="RNN",
    description="Simple RNN model for text classification",
    task="text_classification",
    model_cls=SimpleRNN,
    default_args={
        "vocab_size": 10000,
        "embed_dim": 100,
        "hidden_dim": 256,
        "num_layers": 2,
        "num_classes": 2,
        "dropout": 0.2,
        "rnn_type": "gru",
        "bidirectional": True,
    },
    pretrained_available=False,
)


class TextCNN(nn.Module):
    """
    CNN model for text classification.
    
    This model uses an embedding layer followed by 1D convolutional layers
    with different kernel sizes for n-gram feature extraction.
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 100,
        num_classes: int = 2,
        kernel_sizes: List[int] = [3, 4, 5],
        num_filters: int = 100,
        dropout: float = 0.5,
    ):
        """
        Initialize the model.
        
        Args:
            vocab_size: Size of the vocabulary
            embed_dim: Dimension of the embeddings
            num_classes: Number of output classes
            kernel_sizes: List of kernel sizes for the convolutional layers
            num_filters: Number of filters for each kernel size
            dropout: Dropout probability
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Convolutional layers for different n-gram sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size)
            for kernel_size in kernel_sizes
        ])
        
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, L) containing token indices
            
        Returns:
            Tensor: Output tensor of shape (B, num_classes)
        """
        # Embed the input
        embedded = self.embedding(x)  # (B, L, embed_dim)
        
        # Prepare for Conv1d by transposing to (B, embed_dim, L)
        embedded = embedded.transpose(1, 2)
        
        # Apply convolutions and max pooling
        conv_outputs = []
        for conv in self.convs:
            # Apply convolution and ReLU
            conv_out = torch.relu(conv(embedded))
            
            # Apply max pooling over the entire sequence
            pooled = torch.max(conv_out, dim=2)[0]
            
            conv_outputs.append(pooled)
        
        # Concatenate outputs from different kernel sizes
        concatenated = torch.cat(conv_outputs, dim=1)
        
        # Apply dropout and pass through linear layer
        concatenated = self.dropout(concatenated)
        logits = self.fc(concatenated)
        
        return logits


# Register the TextCNN model
register_model(
    name="text_cnn",
    architecture="CNN",
    description="CNN model for text classification",
    task="text_classification",
    model_cls=TextCNN,
    default_args={
        "vocab_size": 10000,
        "embed_dim": 100,
        "num_classes": 2,
        "kernel_sizes": [3, 4, 5],
        "num_filters": 100,
        "dropout": 0.5,
    },
    pretrained_available=False,
)


# Register transformers models if available
if _TRANSFORMERS_AVAILABLE:
    # Helper function to create HuggingFace models
    def _create_transformers_model(model_class, model_name, **kwargs):
        # Load pre-trained model and tokenizer
        try:
            config = transformers.AutoConfig.from_pretrained(model_name, **kwargs)
            model = model_class.from_pretrained(model_name, config=config)
            return model
        except Exception as e:
            logger.error(f"Error loading {model_name}: {e}")
            # Fallback to creating with config only
            config = transformers.AutoConfig.from_pretrained(model_name, **kwargs)
            model = model_class.from_config(config)
            return model
    
    # Register BERT model
    try:
        register_model(
            name="bert_base",
            architecture="BERT",
            description="BERT base model for text classification and other NLP tasks",
            task="text_classification",
            factory_fn=lambda **kwargs: _create_transformers_model(
                transformers.BertForSequenceClassification, "bert-base-uncased", **kwargs
            ),
            default_args={"num_labels": 2},
            pretrained_available=True,
            reference_speed={"inference_fp32": 45.0, "inference_fp16": 25.0},  # Examples in ms/batch
            reference_memory={"inference_fp32": 433.0, "inference_fp16": 217.0},  # Examples in MB
        )
    except Exception as e:
        logger.warning(f"Failed to register BERT model: {e}")
    
    # Register RoBERTa model
    try:
        register_model(
            name="roberta_base",
            architecture="RoBERTa",
            description="RoBERTa base model for text classification and other NLP tasks",
            task="text_classification",
            factory_fn=lambda **kwargs: _create_transformers_model(
                transformers.RobertaForSequenceClassification, "roberta-base", **kwargs
            ),
            default_args={"num_labels": 2},
            pretrained_available=True,
        )
    except Exception as e:
        logger.warning(f"Failed to register RoBERTa model: {e}")
    
    # Register DistilBERT model
    try:
        register_model(
            name="distilbert_base",
            architecture="DistilBERT",
            description="DistilBERT base model, a distilled version of BERT",
            task="text_classification",
            factory_fn=lambda **kwargs: _create_transformers_model(
                transformers.DistilBertForSequenceClassification, "distilbert-base-uncased", **kwargs
            ),
            default_args={"num_labels": 2},
            pretrained_available=True,
            reference_speed={"inference_fp32": 25.0, "inference_fp16": 14.0},  # Examples in ms/batch
            reference_memory={"inference_fp32": 260.0, "inference_fp16": 130.0},  # Examples in MB
        )
    except Exception as e:
        logger.warning(f"Failed to register DistilBERT model: {e}")