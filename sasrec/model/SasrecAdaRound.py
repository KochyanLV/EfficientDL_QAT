from qats.AdaRound import AdaRoundWeightQuant
from sasrec.model.Base import BaseSASRec
import torch
from sasrec.train_utils.save_checkpoint import record_init


@record_init
class QuantSASRecAdaRound(BaseSASRec):
    """
    SASRec + AdaRound for prediction head weights.
    Quantizes only the weights of the prediction head (PTQ-baseline for comparison with QAT).
    """
    def __init__(
        self,
        num_items: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_blocks: int = 2,
        max_len: int = 50,
        dropout: float = 0.2,
        bits_w: int = 8,
    ):
        super().__init__(num_items, embed_dim, num_heads, num_blocks, max_len, dropout)
        
        self.wq_head = AdaRoundWeightQuant(bits=bits_w, per_channel=True, ch_axis=0)
    
    def quant_head_weight(self, w: torch.Tensor) -> torch.Tensor:
        return self.wq_head(w)
    
    def adaround_regularization(self, lam: float = 1e-4) -> torch.Tensor:
        return self.wq_head.regularization(lam)
    
    def set_eval_hard_round(self, use_hard: bool = True):
        self.wq_head.set_hard_round(use_hard)

