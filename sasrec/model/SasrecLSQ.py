from qats.LSQ import LSQQuant
from sasrec.model.Base import BaseSASRec
import torch
from sasrec.train_utils.save_checkpoint import record_init


@record_init
class QuantSASRecLSQ(BaseSASRec):
    """
    SASRec + LSQ fake quantization:
      - LSQ on activations after Embedding (per-tensor)
      - LSQ on activations after each attention block (per-tensor)
      - LSQ on activations after each FFN block (per-tensor)
      - LSQ on head weights (per-channel along out_features)
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
        
        self.lsq_act_emb = LSQQuant(bits=bits, per_channel=False)
        
        self.lsq_act_attn = torch.nn.ModuleList([
            LSQQuant(bits=bits, per_channel=False) for _ in range(num_blocks)
        ])
        self.lsq_act_ffn = torch.nn.ModuleList([
            LSQQuant(bits=bits, per_channel=False) for _ in range(num_blocks)
        ])
        
        self.lsq_w_head = LSQQuant(bits=bits, per_channel=True, ch_axis=0)
    
    def quant_embed_out(self, x: torch.Tensor) -> torch.Tensor:
        return self.lsq_act_emb(x)
    
    def quant_attn_out(self, x: torch.Tensor, block_idx: int = 0) -> torch.Tensor:
        return self.lsq_act_attn[block_idx](x)
    
    def quant_ffn_out(self, x: torch.Tensor, block_idx: int = 0) -> torch.Tensor:
        return self.lsq_act_ffn[block_idx](x)
    
    def quant_head_weight(self, w: torch.Tensor) -> torch.Tensor:
        return self.lsq_w_head(w)

