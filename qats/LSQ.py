import math
import torch
import torch.nn as nn


class LSQQuant(nn.Module):
    """
    LSQ-style quantization with learnable step size `s` and grad scaling.
    - Symmetric signed INT8 by default (levels in [-Q, Q]).
    - Per-tensor for activations; per-channel for weights recommended.
    Minimal, self-contained, no external deps.
    """
    def __init__(self, bits: int = 8, per_channel: bool = False, ch_axis: int = 0):
        super().__init__()
        assert bits >= 2
        self.bits = bits
        self.Q = 2 ** (bits - 1) - 1
        self.per_channel = per_channel
        self.ch_axis = ch_axis
        self.s = nn.Parameter(torch.tensor(1.0))
        self.register_buffer('initialized', torch.tensor(False))
        self.register_buffer('eps', torch.tensor(1e-8))

    def _view_like(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # reshape per-channel parameter to broadcast over x
        if not self.per_channel:
            return t
        view = [1] * x.dim()
        view[self.ch_axis] = -1
        return t.view(view)

    @torch.no_grad()
    def _init_scale(self, x: torch.Tensor):
        if self.per_channel:
            x_perm = x.transpose(0, self.ch_axis).contiguous().flatten(1)
            s = (2 * x_perm.abs().mean(dim=1) / max(self.Q, 1)).clamp(min=1e-4)
            self.s = nn.Parameter(s)
        else:
            s = (2 * x.abs().mean() / max(self.Q, 1)).clamp(min=1e-4)
            self.s = nn.Parameter(s)
        self.initialized.fill_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.initialized:
            self._init_scale(x.detach())

        # Grad scale factor g = 1/sqrt(N * Q)
        if self.per_channel:
            N = x.numel() // x.size(self.ch_axis)
        else:
            N = x.numel()
        g = 1.0 / math.sqrt(N * max(self.Q, 1))

        # apply grad scaling on s via hook (no custom backward needed)
        def _hook(grad: torch.Tensor):
            return grad * g
        h = self.s.register_hook(_hook)

        s_b = self._view_like(x, self.s).clamp(min=self.eps)
        x_div = x / s_b
        x_clamped = x_div.clamp(-self.Q, self.Q)
        x_rounded = (x_clamped - x_clamped.detach()) + x_clamped.round().detach()
        y = x_rounded * s_b

        h.remove()
        return y