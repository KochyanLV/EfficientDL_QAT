from qats.DoReFa import DoReFaActQuant, DoReFaWeightQuant
from sasrec.model.Base import BaseSASRec
import torch
from sasrec.train_utils.save_checkpoint import record_init


@record_init
class QuantSASRecDoReFa(BaseSASRec):
    """
    SASRec + DoReFa:
      - DoReFa activations after Embedding and transformer blocks
      - DoReFa weights for prediction head (Linear)
    By default:
      * Activations: signed=True (transformer outputs can be negative)
      * preproc='tanh' for activation normalization
    """
    def __init__(
        self,
        num_items: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_blocks: int = 2,
        max_len: int = 50,
        dropout: float = 0.2,
        bits_a: int = 8,
        bits_w: int = 8,
        act_signed: bool = True,
        act_preproc: str = "tanh",
    ):
        super().__init__(num_items, embed_dim, num_heads, num_blocks, max_len, dropout)
        
        # DoReFa activation quantizers
        self.dorefa_act_emb = DoReFaActQuant(bits_a=bits_a, signed=act_signed, preproc=act_preproc)
        
        self.dorefa_act_attn = torch.nn.ModuleList([
            DoReFaActQuant(bits_a=bits_a, signed=act_signed, preproc=act_preproc)
            for _ in range(num_blocks)
        ])
        self.dorefa_act_ffn = torch.nn.ModuleList([
            DoReFaActQuant(bits_a=bits_a, signed=act_signed, preproc=act_preproc)
            for _ in range(num_blocks)
        ])
        
        # DoReFa weight quantizer for head
        self.dorefa_w_head = DoReFaWeightQuant(bits_w=bits_w)
    
    def quant_embed_out(self, x: torch.Tensor) -> torch.Tensor:
        return self.dorefa_act_emb(x)
    
    def quant_attn_out(self, x: torch.Tensor, block_idx: int = 0) -> torch.Tensor:
        return self.dorefa_act_attn[block_idx](x)
    
    def quant_ffn_out(self, x: torch.Tensor, block_idx: int = 0) -> torch.Tensor:
        return self.dorefa_act_ffn[block_idx](x)
    
    def quant_head_weight(self, w: torch.Tensor) -> torch.Tensor:
        return self.dorefa_w_head(w)

