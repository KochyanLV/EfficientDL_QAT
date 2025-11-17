from qats.AdaRound import AdaRoundWeightQuant
from espcn.model.Base import BaseESPCN
import torch
from espcn.train_utils.save_checkpoint import record_init


@record_init
class QuantESPCNAdaRound(BaseESPCN):
    """
    BaseESPCN + AdaRound for weights.
    Applies learnable rounding to conv layer weights.
    """
    def __init__(self, scale_factor: int = 3, num_channels: int = 3, 
                 feature_dim: int = 64, bits_w: int = 8):
        super().__init__(scale_factor, num_channels, feature_dim)
        
        self.quant_enabled = True  # Can disable during warmup
        
        # AdaRound weight quantizers (per-channel)
        self.wq1 = AdaRoundWeightQuant(bits=bits_w, per_channel=True, ch_axis=0)
        self.wq2 = AdaRoundWeightQuant(bits=bits_w, per_channel=True, ch_axis=0)
        self.wq3 = AdaRoundWeightQuant(bits=bits_w, per_channel=True, ch_axis=0)
    
    def quant_conv1_weight(self, w: torch.Tensor) -> torch.Tensor:
        if not self.quant_enabled:
            return w
        return self.wq1(w)
    
    def quant_conv2_weight(self, w: torch.Tensor) -> torch.Tensor:
        if not self.quant_enabled:
            return w
        return self.wq2(w)
    
    def quant_conv3_weight(self, w: torch.Tensor) -> torch.Tensor:
        if not self.quant_enabled:
            return w
        return self.wq3(w)
    
    def adaround_regularization(self, lam: float = 1e-4) -> torch.Tensor:
        """Regularization to push rounding parameters to {0,1}"""
        return (self.wq1.regularization(lam) + 
                self.wq2.regularization(lam) + 
                self.wq3.regularization(lam))
    
    def set_eval_hard_round(self, use_hard: bool = True):
        """Set hard rounding mode for evaluation"""
        self.wq1.set_hard_round(use_hard)
        self.wq2.set_hard_round(use_hard)
        self.wq3.set_hard_round(use_hard)

