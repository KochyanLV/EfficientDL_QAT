import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sasrec.train_utils.save_checkpoint import record_init


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (B, T, D)
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)


@record_init
class BaseSASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation (SASRec) base model.
    Uses transformer encoder with self-attention for sequential recommendation.
    
    Quantization hooks:
        - quant_embed_out(x): quantize after embedding
        - quant_attn_out(x): quantize after each attention block
        - quant_ffn_out(x): quantize after each FFN block
        - quant_head_weight(w): quantize prediction head weights
    """
    def __init__(
        self,
        num_items: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_blocks: int = 2,
        max_len: int = 50,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.max_len = max_len
        
        # Item embedding (including padding token 0)
        self.item_emb = nn.Embedding(num_items, embed_dim, padding_idx=0)
        
        # Positional encoding
        self.pos_enc = PositionalEncoding(embed_dim, max_len)
        
        # Transformer encoder blocks
        self.num_blocks = num_blocks
        self.attn_blocks = nn.ModuleList()
        self.ffn_blocks = nn.ModuleList()
        self.attn_layer_norms = nn.ModuleList()
        self.ffn_layer_norms = nn.ModuleList()
        
        for _ in range(num_blocks):
            # Multi-head self-attention
            self.attn_blocks.append(
                nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            )
            self.attn_layer_norms.append(nn.LayerNorm(embed_dim))
            
            # Feed-forward network
            self.ffn_blocks.append(nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout)
            ))
            self.ffn_layer_norms.append(nn.LayerNorm(embed_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        # Prediction head - predict over all items
        self.head = nn.Linear(embed_dim, num_items)
        
    def create_attention_mask(self, seq_len: int, device: torch.device):
        """Create causal attention mask"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask
    
    # Quantization hooks (no-op by default)
    def quant_embed_out(self, x):
        return x
    
    def quant_attn_out(self, x, block_idx: int = 0):
        return x
    
    def quant_ffn_out(self, x, block_idx: int = 0):
        return x
    
    def quant_head_weight(self, w):
        return w
    
    def forward(self, sequences):
        """
        Args:
            sequences: (B, T) - padded sequences of item IDs
            
        Returns:
            logits: (B, num_items) - prediction scores for all items
        """
        # Embedding
        x = self.item_emb(sequences)  # (B, T, D)
        x = self.quant_embed_out(x)
        
        # Positional encoding
        x = self.pos_enc(x)
        x = self.dropout(x)
        
        # Create attention mask (causal)
        seq_len = sequences.size(1)
        attn_mask = self.create_attention_mask(seq_len, sequences.device)
        
        # Transformer blocks
        for i in range(self.num_blocks):
            # Self-attention with residual
            attn_out, _ = self.attn_blocks[i](x, x, x, attn_mask=attn_mask, need_weights=False)
            attn_out = self.quant_attn_out(attn_out, block_idx=i)
            x = self.attn_layer_norms[i](x + attn_out)
            
            # FFN with residual
            ffn_out = self.ffn_blocks[i](x)
            ffn_out = self.quant_ffn_out(ffn_out, block_idx=i)
            x = self.ffn_layer_norms[i](x + ffn_out)
        
        # Take last position (most recent item)
        # For sequences with padding, we want the last non-padding position
        # Simple approach: take last position (assumes at least one non-padding item)
        x_last = x[:, -1, :]  # (B, D)
        
        # Prediction
        w_q = self.quant_head_weight(self.head.weight)
        logits = F.linear(x_last, w_q, self.head.bias)  # (B, num_items)
        
        return logits
