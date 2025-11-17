from qats.FakeSte import FakeQuantSTE
from espcn.model.Base import BaseESPCN
import torch
import torch.nn as nn
from espcn.train_utils.save_checkpoint import record_init


@record_init
class QuantESPCNSTE(BaseESPCN):
    """
    BaseESPCN + Straight-Through Estimator (STE) fake quantization:
      - Simple STE quantization on activations (per-tensor)
      - Simple STE quantization on weights (per-tensor for simplicity)
    """
    def __init__(self, scale_factor: int = 3, num_channels: int = 3, 
                 feature_dim: int = 64, bits: int = 8):
        super().__init__(scale_factor, num_channels, feature_dim)
        
        self.bits = bits
        self.Q = 2 ** (bits - 1) - 1
        self.ste_quant = FakeQuantSTE(bits=bits)
    
    def _compute_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-tensor scale for quantization."""
        return x.abs().max() / self.Q
    
    def quant_conv1_out(self, x: torch.Tensor) -> torch.Tensor:
        s = self._compute_scale(x)
        return self.ste_quant(x, s)
    
    def quant_conv2_out(self, x: torch.Tensor) -> torch.Tensor:
        s = self._compute_scale(x)
        return self.ste_quant(x, s)
    
    def quant_conv3_out(self, x: torch.Tensor) -> torch.Tensor:
        s = self._compute_scale(x)
        return self.ste_quant(x, s)
    
    def quant_conv1_weight(self, w: torch.Tensor) -> torch.Tensor:
        s = self._compute_scale(w)
        return self.ste_quant(w, s)
    
    def quant_conv2_weight(self, w: torch.Tensor) -> torch.Tensor:
        s = self._compute_scale(w)
        return self.ste_quant(w, s)
    
    def quant_conv3_weight(self, w: torch.Tensor) -> torch.Tensor:
        s = self._compute_scale(w)
        return self.ste_quant(w, s)

