from qats.Apot import APoTQuantRCF
from espcn.model.Base import BaseESPCN
import torch
from espcn.train_utils.save_checkpoint import record_init


@record_init
class QuantESPCNAPoT(BaseESPCN):
    """
    BaseESPCN + APoT (Additive Powers-of-Two) quantization:
      - APoT on activations (per-tensor)
      - APoT on weights (per-channel), with optional WeightNorm
    """
    def __init__(self, scale_factor: int = 3, num_channels: int = 3, 
                 feature_dim: int = 64, bits: int = 8, k: int = 2,
                 init_alpha_act: float = 6.0, init_alpha_w: float = 2.0,
                 use_weight_norm_w: bool = True):
        super().__init__(scale_factor, num_channels, feature_dim)
        
        self.quant_enabled = True  # Can disable during warmup
        
        # Activation quantizers (per-tensor)
        self.apot_act1 = APoTQuantRCF(bits=bits, k=k, per_channel=False, 
                                      init_alpha=init_alpha_act)
        self.apot_act2 = APoTQuantRCF(bits=bits, k=k, per_channel=False, 
                                      init_alpha=init_alpha_act)
        self.apot_act3 = APoTQuantRCF(bits=bits, k=k, per_channel=False, 
                                      init_alpha=init_alpha_act)
        
        # Weight quantizers (per-channel)
        self.apot_w1 = APoTQuantRCF(bits=bits, k=k, per_channel=True, ch_axis=0,
                                    init_alpha=init_alpha_w, 
                                    use_weight_norm=use_weight_norm_w)
        self.apot_w2 = APoTQuantRCF(bits=bits, k=k, per_channel=True, ch_axis=0,
                                    init_alpha=init_alpha_w, 
                                    use_weight_norm=use_weight_norm_w)
        self.apot_w3 = APoTQuantRCF(bits=bits, k=k, per_channel=True, ch_axis=0,
                                    init_alpha=init_alpha_w, 
                                    use_weight_norm=use_weight_norm_w)
    
    def quant_conv1_out(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quant_enabled:
            return x
        return self.apot_act1(x, is_weight=False)
    
    def quant_conv2_out(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quant_enabled:
            return x
        return self.apot_act2(x, is_weight=False)
    
    def quant_conv3_out(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quant_enabled:
            return x
        return self.apot_act3(x, is_weight=False)
    
    def quant_conv1_weight(self, w: torch.Tensor) -> torch.Tensor:
        if not self.quant_enabled:
            return w
        return self.apot_w1(w, is_weight=True)
    
    def quant_conv2_weight(self, w: torch.Tensor) -> torch.Tensor:
        if not self.quant_enabled:
            return w
        return self.apot_w2(w, is_weight=True)
    
    def quant_conv3_weight(self, w: torch.Tensor) -> torch.Tensor:
        if not self.quant_enabled:
            return w
        return self.apot_w3(w, is_weight=True)

