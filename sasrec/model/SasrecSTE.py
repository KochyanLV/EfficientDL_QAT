import torch
import torch.nn as nn
from qats.FakeSte import FakeQuantSTE
from sasrec.model.Base import BaseSASRec
from sasrec.train_utils.save_checkpoint import record_init


@record_init
class QuantSASRecSTE(BaseSASRec):
    """
    SASRec with simple fake quantization via STE:
      - Fake quant on activations after Embedding and transformer blocks (per-tensor scales)
      - Fake quant on head weights (per-tensor scale)
    """
    def __init__(
        self,
        num_items: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_blocks: int = 2,
        max_len: int = 50,
        dropout: float = 0.2,
        bits: int = 8,
    ):
        super().__init__(num_items, embed_dim, num_heads, num_blocks, max_len, dropout)
        
        self.fq = FakeQuantSTE(bits)
        
        self.s_act_emb = nn.Parameter(torch.tensor(1.0))
        
        self.s_act_attn = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0)) for _ in range(num_blocks)
        ])
        self.s_act_ffn = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0)) for _ in range(num_blocks)
        ])
        
        self.s_w_head = nn.Parameter(torch.tensor(1.0))
    
    def quant_embed_out(self, x: torch.Tensor) -> torch.Tensor:
        return self.fq(x, self.s_act_emb)
    
    def quant_attn_out(self, x: torch.Tensor, block_idx: int = 0) -> torch.Tensor:
        return self.fq(x, self.s_act_attn[block_idx])
    
    def quant_ffn_out(self, x: torch.Tensor, block_idx: int = 0) -> torch.Tensor:
        return self.fq(x, self.s_act_ffn[block_idx])
    
    def quant_head_weight(self, w: torch.Tensor) -> torch.Tensor:
        return self.fq(w, self.s_w_head)

