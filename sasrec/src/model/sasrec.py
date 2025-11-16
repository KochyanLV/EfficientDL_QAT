import torch
from torch import nn

from model import EmbeddingLayer, SelfAttnBlock


class SASRec(nn.Module):
    """
    Base SASRec model with quantization hooks.
    
    Quantization hooks (no-op by default, override in subclasses):
        - quant_embed_out(x): quantize after embedding layer
        - quant_attn_out(x, block_idx): quantize after each attention block
        - quant_final_out(x): quantize after final layer norm
    """
    def __init__(
        self,
        num_items: int,
        num_blocks: int,
        hidden_dim: int,
        max_seq_len: int,
        dropout_p: float,
        share_item_emb: bool,
        device: str,
    ) -> None:
        super().__init__()

        self.device = device
        self.num_blocks = num_blocks

        self.embedding_layer = EmbeddingLayer(
            num_items=num_items,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
        )
        self_attn_blocks = [
            SelfAttnBlock(
                max_seq_len=max_seq_len,
                hidden_dim=hidden_dim,
                dropout_p=dropout_p,
                device=device,
            )
            for _ in range(num_blocks)
        ]
        self.self_attn_blocks = nn.ModuleList(self_attn_blocks)

        self.dropout = nn.Dropout(p=dropout_p)
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_dim)

    def quant_embed_out(self, x: torch.Tensor) -> torch.Tensor:
        """Quantization hook after embedding layer"""
        return x
    
    def quant_attn_out(self, x: torch.Tensor, block_idx: int = 0) -> torch.Tensor:
        """Quantization hook after each attention block"""
        return x
    
    def quant_final_out(self, x: torch.Tensor) -> torch.Tensor:
        """Quantization hook after final layer norm"""
        return x

    def get_padding_mask(self, seqs: torch.Tensor) -> torch.Tensor:
        is_padding = torch.tensor(seqs == 0, dtype=torch.bool)
        padding_mask = ~is_padding

        return padding_mask

    def forward(
        self,
        input_seqs: torch.Tensor,
        item_idxs: torch.Tensor = None,
        positive_seqs: torch.Tensor = None,
        negative_seqs: torch.Tensor = None,
    ) -> torch.Tensor:
        padding_mask = self.get_padding_mask(seqs=input_seqs).to(self.device)

        input_embs = self.dropout(self.embedding_layer(input_seqs))
        input_embs = self.quant_embed_out(input_embs)
        input_embs *= padding_mask.unsqueeze(-1)

        # For loop because we need to apply quantization after each block
        attn_output = input_embs
        for i, block in enumerate(self.self_attn_blocks):
            attn_output = block(x=attn_output, padding_mask=padding_mask)
            attn_output = self.quant_attn_out(attn_output, block_idx=i)
        
        attn_output = self.layer_norm(attn_output)
        attn_output = self.quant_final_out(attn_output)

        if item_idxs is not None:  # Inference.
            item_embs = self.embedding_layer.item_emb_matrix(item_idxs)
            logits = attn_output @ item_embs.transpose(2, 1)
            logits = logits[:, -1, :]
            outputs = (logits,)
        elif (positive_seqs is not None) and (negative_seqs is not None):  # Training.
            positive_embs = self.dropout(self.embedding_layer(positive_seqs))
            positive_embs = self.quant_embed_out(positive_embs)
            
            negative_embs = self.dropout(self.embedding_layer(negative_seqs))
            negative_embs = self.quant_embed_out(negative_embs)

            positive_logits = (attn_output * positive_embs).sum(dim=-1)
            negative_logits = (attn_output * negative_embs).sum(dim=-1)

            outputs = (positive_logits,)
            outputs += (negative_logits,)

        return outputs
