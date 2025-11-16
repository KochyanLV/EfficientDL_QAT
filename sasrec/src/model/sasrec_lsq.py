import torch
from torch import nn
import sys
from pathlib import Path

# Add parent directory to path for qats import
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from qats.LSQ import LSQQuant

try:
    from model import SASRec
except ImportError:
    from src.model import SASRec


class QuantSASRecLSQ(SASRec):
    """
    SASRec with LSQ (Learned Step Size Quantization).
    
    Applies LSQ quantization to:
    - Activations after embedding layer (per-tensor)
    - Activations after each attention block (per-tensor)
    - Activations after final layer norm (per-tensor)
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
        bits: int = 8,
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
        
        # LSQ quantizers for activations (per-tensor)
        self.lsq_embed = LSQQuant(bits=bits, per_channel=False)
        
        # One LSQ quantizer per attention block
        self.lsq_attn_blocks = nn.ModuleList([
            LSQQuant(bits=bits, per_channel=False)
            for _ in range(num_blocks)
        ])
        
        self.lsq_final = LSQQuant(bits=bits, per_channel=False)
    
    def quant_embed_out(self, x: torch.Tensor) -> torch.Tensor:
        return self.lsq_embed(x)
    
    def quant_attn_out(self, x: torch.Tensor, block_idx: int = 0) -> torch.Tensor:
        return self.lsq_attn_blocks[block_idx](x)
    
    def quant_final_out(self, x: torch.Tensor) -> torch.Tensor:
        return self.lsq_final(x)

