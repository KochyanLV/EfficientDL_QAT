import torch
from torch import nn
import sys
from pathlib import Path

# Add parent directory to path for qats import
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from qats.FakeSte import FakeQuantSTE

try:
    from model import SASRec
except ImportError:
    from src.model import SASRec


class QuantSASRecSTE(SASRec):
    """
    SASRec with Fake STE (Straight-Through Estimator) quantization.
    
    Applies simple STE quantization to:
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
        
        # STE quantizers for activations
        self.ste_embed = FakeQuantSTE(bits=bits)
        
        # One STE quantizer per attention block
        self.ste_attn_blocks = nn.ModuleList([
            FakeQuantSTE(bits=bits)
            for _ in range(num_blocks)
        ])
        
        self.ste_final = FakeQuantSTE(bits=bits)
    
        # Learnable scale parameters for each quantization point
        self.s_act_embed = nn.Parameter(torch.tensor(1.0))
        self.s_act_attn_blocks = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0)) for _ in range(num_blocks)
        ])
        self.s_act_final = nn.Parameter(torch.tensor(1.0))
    
    def quant_embed_out(self, x: torch.Tensor) -> torch.Tensor:
        return self.ste_embed(x, self.s_act_embed)
    
    def quant_attn_out(self, x: torch.Tensor, block_idx: int = 0) -> torch.Tensor:
        return self.ste_attn_blocks[block_idx](x, self.s_act_attn_blocks[block_idx])
    
    def quant_final_out(self, x: torch.Tensor) -> torch.Tensor:
        return self.ste_final(x, self.s_act_final)

