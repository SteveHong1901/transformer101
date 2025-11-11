import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    ''' Embedding of size (vocab_size, d_model) '''
    def __init__(self, d_model: int, vocab_size: int): 
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) # Map each numbers to 1x512 vectors. Vectors learnt by the model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    ''' Positional Encoding: Vector of size (1 x d_model) to add on each of the input embedding. One for each position. '''
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1), can also use .reshape(-1, 1) or .view(-1, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000)/d_model))

        # Apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        # register_buffer: Registers 'pe' as a buffer (non-trainable parameter)
        # - Automatically moved to GPU/CPU when module.to(device) is called
        # - Saved/loaded with model state_dict
        # - Not updated during backprop (positional encodings are fixed)
        # Equivalent to self.pe = pe, but properly integrated with PyTorch's module system
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encodings to input embeddings
        # self.pe shape: (1, seq_len, d_model)
        # x shape: (batch_size, seq_len, d_model)
        # We slice self.pe to match x's sequence length (in case x is shorter than max seq_len)
        # .requires_grad_(False) ensures gradients don't flow through positional encodings
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        
        # Apply dropout for regularization during training
        # This helps prevent overfitting by randomly zeroing some values
        return self.dropout(x)

def LayerNormalization(nn.Module):
    ''' 
    output = alpha * (x - mean) / sqrt(variance + eps) + bias 
    nn.Parameter is a special wrapper that tells PyTorch: "This tensor should be learned during training."
    '''
    def __init__(self, eps: float = 10e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Paramter(torch.zeros(1)) # Added

