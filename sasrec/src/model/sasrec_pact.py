import torch
from torch import nn
import sys
from pathlib import Path

# Add parent directory to path for qats import
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from qats.Pact import PACTAct
from model import SASRec


class QuantSASRecPACT(SASRec):
    """
    SASRec with PACT (Parameterized Clipping Activation).
    
    Applies PACT quantization to:
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
        bits_act: int = 8,
        pact_init_alpha: float = 6.0,
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
        
        # PACT quantizers for activations
        self.pact_embed = PACTAct(bits=bits_act, init_alpha=pact_init_alpha)
        
        # One PACT quantizer per attention block
        self.pact_attn_blocks = nn.ModuleList([
            PACTAct(bits=bits_act, init_alpha=pact_init_alpha)
            for _ in range(num_blocks)
        ])
        
        self.pact_final = PACTAct(bits=bits_act, init_alpha=pact_init_alpha)
    
    def quant_embed_out(self, x: torch.Tensor) -> torch.Tensor:
        return self.pact_embed(x)
    
    def quant_attn_out(self, x: torch.Tensor, block_idx: int = 0) -> torch.Tensor:
        return self.pact_attn_blocks[block_idx](x)
    
    def quant_final_out(self, x: torch.Tensor) -> torch.Tensor:
        return self.pact_final(x)
    
    def pact_regularization(self, lam: float = 1e-4) -> torch.Tensor:
        """L2 regularization on PACT alpha parameters"""
        reg = self.pact_embed.l2_alpha(lam)
        for pact_block in self.pact_attn_blocks:
            reg += pact_block.l2_alpha(lam)
        reg += self.pact_final.l2_alpha(lam)
        return reg

