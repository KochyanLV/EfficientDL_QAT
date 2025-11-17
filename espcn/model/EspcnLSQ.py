from qats.LSQ import LSQQuant
from espcn.model.Base import BaseESPCN
import torch
from espcn.train_utils.save_checkpoint import record_init


@record_init
class QuantESPCNLSQ(BaseESPCN):
    """
    BaseESPCN + LSQ fake quantization:
      - LSQ on activations after each conv layer (per-tensor)
      - LSQ on weights for each conv layer (per-channel along out_channels)
    """
    def __init__(self, scale_factor: int = 3, num_channels: int = 3, 
                 feature_dim: int = 64, bits: int = 8):
        super().__init__(scale_factor, num_channels, feature_dim)
        
        self.quant_enabled = True  # Can disable during warmup
        
        # Activation quantizers (per-tensor)
        self.lsq_act1 = LSQQuant(bits=bits, per_channel=False)
        self.lsq_act2 = LSQQuant(bits=bits, per_channel=False)
        self.lsq_act3 = LSQQuant(bits=bits, per_channel=False)
        
        # Weight quantizers (per-channel)
        self.lsq_w1 = LSQQuant(bits=bits, per_channel=True, ch_axis=0)
        self.lsq_w2 = LSQQuant(bits=bits, per_channel=True, ch_axis=0)
        self.lsq_w3 = LSQQuant(bits=bits, per_channel=True, ch_axis=0)
    
    def quant_conv1_out(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quant_enabled:
            return x
        return self.lsq_act1(x)
    
    def quant_conv2_out(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quant_enabled:
            return x
        return self.lsq_act2(x)
    
    def quant_conv3_out(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quant_enabled:
            return x
        return self.lsq_act3(x)
    
    def quant_conv1_weight(self, w: torch.Tensor) -> torch.Tensor:
        if not self.quant_enabled:
            return w
        return self.lsq_w1(w)
    
    def quant_conv2_weight(self, w: torch.Tensor) -> torch.Tensor:
        if not self.quant_enabled:
            return w
        return self.lsq_w2(w)
    
    def quant_conv3_weight(self, w: torch.Tensor) -> torch.Tensor:
        if not self.quant_enabled:
            return w
        return self.lsq_w3(w)

