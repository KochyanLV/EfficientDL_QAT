import torch
import torch.nn as nn


class PACTAct(nn.Module):
    """
    PACT: learnable activation clipping with uniform fake-quantization.
    - Range: [0, alpha], where alpha is learnable.
    - Quant levels: [0, Q], where Q = 2^bits - 1.
    - Uses STE for rounding; alpha is trained via autograd.
    """
    def __init__(self, bits: int = 8, init_alpha: float = 6.0):
        super().__init__()
        assert bits >= 2
        self.bits = bits
        self.Q = 2 ** bits - 1  # unsigned levels for ReLU-like activations
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        self.register_buffer('eps', torch.tensor(1e-8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.alpha.to(dtype=x.dtype, device=x.device)
        zero = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        y = torch.clamp(x, min=zero, max=a)
        # scale s = alpha / Q
        s = (a.clamp(min=self.eps)) / self.Q
        y_div = y / s
        # clamp to [0, Q] just in case
        y_clamped = y_div.clamp(0, self.Q)
        # STE: grad(identity) through rounding
        y_rounded = (y_clamped - y_clamped.detach()) + y_clamped.round().detach()
        return y_rounded * s

    def l2_alpha(self, lam: float = 1e-4) -> torch.Tensor:
        """Optional L2 regularization term on alpha (add to loss externally)."""
        return lam * self.alpha.pow(2)


class UniformWeightQuant(nn.Module):
    """
    Symmetric per-channel (out_features) uniform fake-quant for weights.
    - Scale computed on-the-fly as s = max(|w_channel|) / Q (no grad for s).
    - Uses STE for rounding.
    """
    def __init__(self, bits: int = 8, per_channel: bool = True, ch_axis: int = 0):
        super().__init__()
        assert bits >= 2
        self.bits = bits
        self.Q = 2 ** (bits - 1) - 1  # signed symmetric INT8
        self.per_channel = per_channel
        self.ch_axis = ch_axis
        self.register_buffer('eps', torch.tensor(1e-8))

    def _broadcast(self, w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if not self.per_channel:
            return s
        view = [1] * w.dim()
        view[self.ch_axis] = -1
        return s.view(view)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        if self.per_channel:
            # move channel axis to dim0 and flatten the rest
            w_perm = w.transpose(0, self.ch_axis).contiguous().flatten(1)
            s = (w_perm.abs().max(dim=1).values / max(self.Q, 1)).clamp(min=self.eps)
        else:
            s = (w.abs().max() / max(self.Q, 1)).clamp(min=self.eps)
        s_b = self._broadcast(w, s)

        w_div = w / s_b
        w_clamped = w_div.clamp(-self.Q, self.Q)
        w_rounded = (w_clamped - w_clamped.detach()) + w_clamped.round().detach()
        return w_rounded * s_b