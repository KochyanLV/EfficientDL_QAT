from qats.DoReFa import DoReFaActQuant, DoReFaWeightQuant
from espcn.model.Base import BaseESPCN
import torch
from espcn.train_utils.save_checkpoint import record_init


@record_init
class QuantESPCNDoReFa(BaseESPCN):
    """
    BaseESPCN + DoReFa quantization:
      - DoReFa activation quantization (with optional preprocessing)
      - DoReFa weight quantization (symmetric per-channel)
    """
    def __init__(self, scale_factor: int = 3, num_channels: int = 3, 
                 feature_dim: int = 64, bits_a: int = 8, bits_w: int = 8,
                 act_signed: bool = True, act_preproc: str = "tanh"):
        super().__init__(scale_factor, num_channels, feature_dim)
        
        # Activation quantizers
        self.aq1 = DoReFaActQuant(bits_a=bits_a, signed=act_signed, preproc=act_preproc)
        self.aq2 = DoReFaActQuant(bits_a=bits_a, signed=act_signed, preproc=act_preproc)
        self.aq3 = DoReFaActQuant(bits_a=bits_a, signed=act_signed, preproc=act_preproc)
        
        # Weight quantizers
        self.wq1 = DoReFaWeightQuant(bits_w=bits_w)
        self.wq2 = DoReFaWeightQuant(bits_w=bits_w)
        self.wq3 = DoReFaWeightQuant(bits_w=bits_w)
    
    def quant_conv1_out(self, x: torch.Tensor) -> torch.Tensor:
        return self.aq1(x)
    
    def quant_conv2_out(self, x: torch.Tensor) -> torch.Tensor:
        return self.aq2(x)
    
    def quant_conv3_out(self, x: torch.Tensor) -> torch.Tensor:
        return self.aq3(x)
    
    def quant_conv1_weight(self, w: torch.Tensor) -> torch.Tensor:
        return self.wq1(w)
    
    def quant_conv2_weight(self, w: torch.Tensor) -> torch.Tensor:
        return self.wq2(w)
    
    def quant_conv3_weight(self, w: torch.Tensor) -> torch.Tensor:
        return self.wq3(w)

