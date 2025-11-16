import torch
from torch import nn
import sys
from pathlib import Path

# Add parent directory to path for qats import
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from qats.Apot import APoTQuantRCF

try:
    from model import SASRec
except ImportError:
    from src.model import SASRec


class QuantSASRecAPoT(SASRec):
    """
    SASRec with APoT (Additive Powers-of-Two Quantization).
    
    Applies APoT quantization to:
    - Activations after embedding layer
    - Activations after each attention block
    - Activations after final layer norm
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
        k: int = 1,
        init_alpha_act: float = 6.0,
        use_weight_norm_w: bool = False,
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
        
        # APoT quantizers for activations
        self.apot_embed = APoTQuantRCF(
            bits=bits,
            k=k,
            init_alpha=init_alpha_act,
            use_weight_norm=use_weight_norm_w,
        )
        
        # One APoT quantizer per attention block
        self.apot_attn_blocks = nn.ModuleList([
            APoTQuantRCF(
                bits=bits,
                k=k,
                init_alpha=init_alpha_act,
                use_weight_norm=use_weight_norm_w,
            )
            for _ in range(num_blocks)
        ])
        
        self.apot_final = APoTQuantRCF(
            bits=bits,
            k=k,
            init_alpha=init_alpha_act,
            use_weight_norm=use_weight_norm_w,
        )
    
    def quant_embed_out(self, x: torch.Tensor) -> torch.Tensor:
        return self.apot_embed(x)
    
    def quant_attn_out(self, x: torch.Tensor, block_idx: int = 0) -> torch.Tensor:
        return self.apot_attn_blocks[block_idx](x)
    
    def quant_final_out(self, x: torch.Tensor) -> torch.Tensor:
        return self.apot_final(x)

