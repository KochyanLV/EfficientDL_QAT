from qats.Apot import APoTQuantRCF
from sasrec.model.Base import BaseSASRec
import torch
from sasrec.train_utils.save_checkpoint import record_init


@record_init
class QuantSASRecAPoT(BaseSASRec):
    """
    SASRec + APoTQuantRCF:
      - APoT on activations (per-tensor) after embedding and transformer blocks
      - APoT on head weights (per-channel by out_features), with optional WeightNorm
    Default parameters: bits=8, k=1 (for bits=8, k must divide (bits-1), so k=1 or k=7).
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
        k: int = 1,
        init_alpha_act: float = 6.0,
        init_alpha_w: float = 2.0,
        use_weight_norm_w: bool = True,
    ):
        super().__init__(num_items, embed_dim, num_heads, num_blocks, max_len, dropout)
        
        self.apot_act_emb = APoTQuantRCF(bits=bits, k=k, per_channel=False, init_alpha=init_alpha_act)
        
        self.apot_act_attn = torch.nn.ModuleList([
            APoTQuantRCF(bits=bits, k=k, per_channel=False, init_alpha=init_alpha_act)
            for _ in range(num_blocks)
        ])
        self.apot_act_ffn = torch.nn.ModuleList([
            APoTQuantRCF(bits=bits, k=k, per_channel=False, init_alpha=init_alpha_act)
            for _ in range(num_blocks)
        ])
        
        self.apot_w_head = APoTQuantRCF(
            bits=bits, k=k, per_channel=True, ch_axis=0,
            init_alpha=init_alpha_w, use_weight_norm=use_weight_norm_w
        )
    
    def quant_embed_out(self, x: torch.Tensor) -> torch.Tensor:
        return self.apot_act_emb(x, is_weight=False)
    
    def quant_attn_out(self, x: torch.Tensor, block_idx: int = 0) -> torch.Tensor:
        return self.apot_act_attn[block_idx](x, is_weight=False)
    
    def quant_ffn_out(self, x: torch.Tensor, block_idx: int = 0) -> torch.Tensor:
        return self.apot_act_ffn[block_idx](x, is_weight=False)
    
    def quant_head_weight(self, w: torch.Tensor) -> torch.Tensor:
        return self.apot_w_head(w, is_weight=True)

