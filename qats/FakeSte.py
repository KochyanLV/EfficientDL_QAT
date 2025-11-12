import torch
import torch.nn as nn


class FakeQuantSTE(nn.Module):
    def __init__(self, bits: int = 8):
        super().__init__()
        self.Q = 2 ** (bits - 1) - 1
        self.register_buffer('eps', torch.tensor(1e-8))

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        s = s.clamp(min=self.eps)
        x_div = x / s
        x_clamped = x_div.clamp(-self.Q, self.Q)
        x_rounded = (x_clamped - x_clamped.detach()) + x_clamped.round().detach()
        return x_rounded * s