import math
import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    """Combines token embeddings with sinusoidal positional encoding."""
    
    def __init__(
        self,
        vocab_size,
        embed_dim,
        max_seq_len,
        pad_token_id,
        dropout_prob,
        norm_eps,
        **kwargs
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        
        # Sinusoidal positional encoding (fixed, not learned)
        self.register_buffer(
            "positional_encoding", 
            self._create_positional_encoding(max_seq_len, embed_dim)
        )
        
        self.layer_norm = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.dropout = nn.Dropout(dropout_prob)
    
    def _create_positional_encoding(self, max_seq_len, embed_dim):
        """
        Create sinusoidal positional encoding from 'Attention is All You Need'.
        PE(pos, 2i) = sin(pos / 10000^(2i/embed_dim))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/embed_dim))
        """
        position = torch.arange(max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        
        pos_encoding = torch.zeros(max_seq_len, embed_dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding.unsqueeze(0)  # Shape: (1, max_seq_len, embed_dim)

    def forward(self, input_ids, **kwargs):
        seq_len = input_ids.size(1)
        
        # Token embeddings + positional encoding
        embeddings = self.token_embedding(input_ids)
        embeddings = embeddings + self.positional_encoding[:, :seq_len, :]
        
        # Layer norm and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, embed_dim, num_heads, dropout_prob):
        super().__init__()
        
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"Embedding dimension ({embed_dim}) must be divisible by "
                f"number of attention heads ({num_heads})"
            )
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Q, K, V projections
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
            mask: (batch_size, 1, 1, seq_len)
        Returns:
            (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.size()
        
        # Project to Q, K, V and split into multiple heads
        # Shape: (batch_size, seq_len, embed_dim) -> (batch_size, num_heads, seq_len, head_dim)
        query = self.query_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores + mask
        
        # Softmax and dropout
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_probs, value)
        
        # Merge heads: (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, embed_dim)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Final output projection
        output = self.out_proj(attention_output)
        
        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, embed_dim, ffn_dim, dropout_prob, activation='gelu'):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_prob)
        
        # Activation function
        activation_map = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'silu': nn.SiLU(),
        }
        self.activation = activation_map.get(activation.lower(), nn.GELU())
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
        Returns:
            (batch_size, seq_len, embed_dim)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    """
    Single Transformer encoder layer.
    Structure: 
        1. Multi-head attention -> Dropout -> Add & Norm
        2. Feed-forward -> Dropout -> Add & Norm
    """
    
    def __init__(
        self,
        embed_dim,
        num_heads,
        ffn_dim,
        dropout_prob,
        attention_dropout,
        norm_eps,
        activation='gelu'
    ):
        super().__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention(embed_dim, num_heads, attention_dropout)
        self.attention_dropout = nn.Dropout(dropout_prob)
        self.attention_norm = nn.LayerNorm(embed_dim, eps=norm_eps)
        
        # Feed-forward network
        self.feed_forward = FeedForward(embed_dim, ffn_dim, dropout_prob, activation)
        self.ffn_dropout = nn.Dropout(dropout_prob)
        self.ffn_norm = nn.LayerNorm(embed_dim, eps=norm_eps)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
            mask: (batch_size, 1, 1, seq_len)
        Returns:
            (batch_size, seq_len, embed_dim)
        """
        # Multi-head attention with Add & Norm
        attention_output = self.attention(x, mask)
        attention_output = self.attention_dropout(attention_output)
        x = self.attention_norm(x + attention_output)  # Add & Norm
        
        # Feed-forward with Add & Norm
        ffn_output = self.feed_forward(x)
        ffn_output = self.ffn_dropout(ffn_output)
        x = self.ffn_norm(x + ffn_output)  # Add & Norm
        
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer encoder: stack of encoder layers.
    Input: token embeddings with positional encoding
    Output: contextualized representations
    """
    
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_layers,
        num_heads,
        ffn_dim,
        max_seq_len,
        pad_token_id,
        dropout_prob=0.1,
        attention_dropout=0.1,
        norm_eps=1e-12,
        activation='gelu'
    ):
        super().__init__()
        
        # Input embeddings
        self.embeddings = InputEmbedding(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
            pad_token_id=pad_token_id,
            dropout_prob=dropout_prob,
            norm_eps=norm_eps
        )
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout_prob=dropout_prob,
                attention_dropout=attention_dropout,
                norm_eps=norm_eps,
                activation=activation
            )
            for _ in range(num_layers)
        ])
    
    def create_attention_mask(self, attention_mask):
        """
        Convert 2D attention mask to 4D for multi-head attention.
        Args:
            attention_mask: (batch_size, seq_len) with 1=attend, 0=ignore
        Returns:
            (batch_size, 1, 1, seq_len) with 0=attend, -10000=ignore
        """
        extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_mask = (1.0 - extended_mask) * -10000.0
        return extended_mask
    
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len) with 1=attend, 0=ignore
        Returns:
            (batch_size, seq_len, embed_dim)
        """
        # Create default attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Convert to 4D mask
        mask = self.create_attention_mask(attention_mask)
        
        # Embed input tokens
        x = self.embeddings(input_ids)
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        return x


# Backward compatibility wrapper
class TransformerModel(nn.Module):
    """
    Wrapper for backward compatibility with the original API.
    Use TransformerEncoder directly for cleaner code.
    """
    
    def __init__(
        self,
        vocab_size,
        hidden_size,
        pad_token_id,
        max_position_embeddings,
        type_vocab_size,
        layer_norm_eps,
        hidden_dropout_prob,
        num_hidden_layers,
        num_attention_heads,
        attention_probs_dropout_prob,
        intermediate_size,
        hidden_act,
    ):
        super().__init__()
        
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            embed_dim=hidden_size,
            num_layers=num_hidden_layers,
            num_heads=num_attention_heads,
            ffn_dim=intermediate_size,
            max_seq_len=max_position_embeddings,
            pad_token_id=pad_token_id,
            dropout_prob=hidden_dropout_prob,
            attention_dropout=attention_probs_dropout_prob,
            norm_eps=layer_norm_eps,
            activation=hidden_act
        )
        
        # Pooler: extract CLS token representation
        self.pooler_dense = nn.Linear(hidden_size, hidden_size)
        self.pooler_activation = nn.Tanh()
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            token_type_ids: Ignored (kept for compatibility)
        Returns:
            sequence_output: (batch_size, seq_len, embed_dim)
            pooled_output: (batch_size, embed_dim)
        """
        sequence_output = self.encoder(input_ids, attention_mask)
        
        # Pool by taking the first token (CLS)
        cls_token = sequence_output[:, 0]
        pooled_output = self.pooler_dense(cls_token)
        pooled_output = self.pooler_activation(pooled_output)
        
        return sequence_output, pooled_output


# Backward compatibility wrapper
class TransformerClassifier(nn.Module):
    """
    Wrapper for backward compatibility.
    For new code, use TransformerEncoder + your own classification head.
    """
    
    def __init__(
        self,
        vocab_size,
        hidden_size,
        pad_token_id,
        max_position_embeddings,
        type_vocab_size,
        layer_norm_eps,
        hidden_dropout_prob,
        num_hidden_layers,
        num_attention_heads,
        attention_probs_dropout_prob,
        intermediate_size,
        hidden_act,
        num_labels,
    ):
        super().__init__()
        self.num_labels = num_labels
        
        self.transformer = TransformerModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            pad_token_id=pad_token_id,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=hidden_dropout_prob,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
        )
        
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        sequence_output, pooled_output = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        
        return logits
