from qats.Pact import PACTAct, UniformWeightQuant
from espcn.model.Base import BaseESPCN
import torch
from espcn.train_utils.save_checkpoint import record_init


@record_init
class QuantESPCNPACT(BaseESPCN):
    """
    BaseESPCN + PACT on activations + symmetric weight fake-quant.
      - PACT after each conv layer
      - Uniform symmetric per-channel fake-quant for conv weights
    """
    def __init__(self, scale_factor: int = 3, num_channels: int = 3, 
                 feature_dim: int = 64, bits_act: int = 8, bits_w: int = 8, 
                 pact_init_alpha: float = 6.0):
        super().__init__(scale_factor, num_channels, feature_dim)
        
        # PACT activation quantizers
        self.pact_act1 = PACTAct(bits=bits_act, init_alpha=pact_init_alpha)
        self.pact_act2 = PACTAct(bits=bits_act, init_alpha=pact_init_alpha)
        self.pact_act3 = PACTAct(bits=bits_act, init_alpha=pact_init_alpha)
        
        # Weight quantizers (symmetric per-channel)
        self.wq1 = UniformWeightQuant(bits=bits_w, per_channel=True, ch_axis=0)
        self.wq2 = UniformWeightQuant(bits=bits_w, per_channel=True, ch_axis=0)
        self.wq3 = UniformWeightQuant(bits=bits_w, per_channel=True, ch_axis=0)
    
    def quant_conv1_out(self, x: torch.Tensor) -> torch.Tensor:
        return self.pact_act1(x)
    
    def quant_conv2_out(self, x: torch.Tensor) -> torch.Tensor:
        return self.pact_act2(x)
    
    def quant_conv3_out(self, x: torch.Tensor) -> torch.Tensor:
        return self.pact_act3(x)
    
    def quant_conv1_weight(self, w: torch.Tensor) -> torch.Tensor:
        return self.wq1(w)
    
    def quant_conv2_weight(self, w: torch.Tensor) -> torch.Tensor:
        return self.wq2(w)
    
    def quant_conv3_weight(self, w: torch.Tensor) -> torch.Tensor:
        return self.wq3(w)
    
    def pact_regularization(self, lam: float = 1e-4) -> torch.Tensor:
        """L2 regularization on PACT alpha parameters"""
        return (self.pact_act1.l2_alpha(lam) + 
                self.pact_act2.l2_alpha(lam) + 
                self.pact_act3.l2_alpha(lam))

