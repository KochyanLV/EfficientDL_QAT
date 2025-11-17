import torch
from torch import nn
import sys
from pathlib import Path

# Add parent directory to path for qats import
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from qats.DoReFa import DoReFaActQuant

try:
    from model import SASRec
except ImportError:
    from src.model import SASRec


class QuantSASRecDoReFa(SASRec):
    """
    SASRec with DoReFa-Net quantization.
    
    Applies DoReFa quantization to:
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
        bits_a: int = 8,
        act_signed: bool = True,
        act_preproc: str = "tanh",
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
        
        # DoReFa quantizers for activations
        self.dorefa_embed = DoReFaActQuant(
            bits_a=bits_a,
            signed=act_signed,
            preproc=act_preproc,
        )
        
        # One DoReFa quantizer per attention block
        self.dorefa_attn_blocks = nn.ModuleList([
            DoReFaActQuant(
                bits_a=bits_a,
                signed=act_signed,
                preproc=act_preproc,
            )
            for _ in range(num_blocks)
        ])
        
        self.dorefa_final = DoReFaActQuant(
            bits_a=bits_a,
            signed=act_signed,
            preproc=act_preproc,
        )
    
    def quant_embed_out(self, x: torch.Tensor) -> torch.Tensor:
        return self.dorefa_embed(x)
    
    def quant_attn_out(self, x: torch.Tensor, block_idx: int = 0) -> torch.Tensor:
        return self.dorefa_attn_blocks[block_idx](x)
    
    def quant_final_out(self, x: torch.Tensor) -> torch.Tensor:
        return self.dorefa_final(x)

