import torch
from torch import nn
import sys
from pathlib import Path

# Add parent directory to path for qats import
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from qats.AdaRound import AdaRoundQuant
from model import SASRec


class QuantSASRecAdaRound(SASRec):
    """
    SASRec with AdaRound (Adaptive Rounding).
    
    AdaRound is primarily for weight quantization, so we apply it to
    the embedding weights conceptually (though embeddings are looked up).
    For activations, we use simple quantization.
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
        bits_w: int = 8,
    ) -> None:
        super().__init__(
            num_items=num_items,
            num_blocks=num_blocks,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
            dropout_p=dropout_p,
            share_item_emb=share_item_emb,
            device=device,
        )
        
        # AdaRound for activations (treating them as "weights" to quantize)
        self.adaround_embed = AdaRoundQuant(bits=bits_w)
        
        # One AdaRound quantizer per attention block
        self.adaround_attn_blocks = nn.ModuleList([
            AdaRoundQuant(bits=bits_w)
            for _ in range(num_blocks)
        ])
        
        self.adaround_final = AdaRoundQuant(bits=bits_w)
    
    def quant_embed_out(self, x: torch.Tensor) -> torch.Tensor:
        return self.adaround_embed(x)
    
    def quant_attn_out(self, x: torch.Tensor, block_idx: int = 0) -> torch.Tensor:
        return self.adaround_attn_blocks[block_idx](x)
    
    def quant_final_out(self, x: torch.Tensor) -> torch.Tensor:
        return self.adaround_final(x)
    
    def adaround_regularization(self, lam: float = 1e-3, beta: float = 2.0) -> torch.Tensor:
        """Regularization for AdaRound rounding parameters"""
        reg = self.adaround_embed.regularization(lam, beta)
        for ada_block in self.adaround_attn_blocks:
            reg += ada_block.regularization(lam, beta)
        reg += self.adaround_final.regularization(lam, beta)
        return reg

