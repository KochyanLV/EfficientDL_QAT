from qats.Pact import PACTAct, UniformWeightQuant
from sasrec.model.Base import BaseSASRec
import torch
from sasrec.train_utils.save_checkpoint import record_init


@record_init
class QuantSASRecPACT(BaseSASRec):
    """
    SASRec + PACT on activations + symmetric weight fake-quant on the head.
      - PACT after Embedding
      - PACT after each attention block
      - PACT after each FFN block
      - Uniform symmetric per-channel fake-quant for Linear head weights
    """
    def __init__(
        self,
        num_items: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_blocks: int = 2,
        max_len: int = 50,
        dropout: float = 0.2,
        bits_act: int = 8,
        bits_w: int = 8,
        pact_init_alpha: float = 6.0,
    ):
        super().__init__(num_items, embed_dim, num_heads, num_blocks, max_len, dropout)
        
        # PACT activation quantizers
        self.pact_emb = PACTAct(bits=bits_act, init_alpha=pact_init_alpha)
        
        self.pact_attn = torch.nn.ModuleList([
            PACTAct(bits=bits_act, init_alpha=pact_init_alpha) for _ in range(num_blocks)
        ])
        self.pact_ffn = torch.nn.ModuleList([
            PACTAct(bits=bits_act, init_alpha=pact_init_alpha) for _ in range(num_blocks)
        ])
        
        # Weight quantizer for head
        self.wq_head = UniformWeightQuant(bits=bits_w, per_channel=True, ch_axis=0)
    
    def quant_embed_out(self, x: torch.Tensor) -> torch.Tensor:
        return self.pact_emb(x)
    
    def quant_attn_out(self, x: torch.Tensor, block_idx: int = 0) -> torch.Tensor:
        return self.pact_attn[block_idx](x)
    
    def quant_ffn_out(self, x: torch.Tensor, block_idx: int = 0) -> torch.Tensor:
        return self.pact_ffn[block_idx](x)
    
    def quant_head_weight(self, w: torch.Tensor) -> torch.Tensor:
        return self.wq_head(w)
    
    def pact_regularization(self, lam: float = 1e-4) -> torch.Tensor:
        """L2 regularization on all PACT alpha parameters"""
        reg = self.pact_emb.l2_alpha(lam)
        for pact_block in self.pact_attn:
            reg = reg + pact_block.l2_alpha(lam)
        for pact_block in self.pact_ffn:
            reg = reg + pact_block.l2_alpha(lam)
        return reg

