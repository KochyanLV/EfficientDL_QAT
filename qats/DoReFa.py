import torch
import torch.nn as nn


def ste_round(x: torch.Tensor) -> torch.Tensor:
    return (x - x.detach()) + x.round().detach()

def quantize_kbits_unit(x01: torch.Tensor, k: int) -> torch.Tensor:
    """
    Квантование x∈[0,1] в k бит (равномерная сетка из 2^k уровней).
    """
    n = (1 << k) - 1  # 2^k - 1
    y = ste_round(x01 * n) / n
    return y.clamp(0.0, 1.0)


class DoReFaWeightQuant(nn.Module):
    """
    DoReFa для весов (signed):
      1) w_t = tanh(w)
      2) w_n = w_t / (2*max|w_t|) + 0.5 ∈ [0,1]
      3) k-битное квантование в [0,1]
      4) обратно в [-1,1]: w_q = 2*w_k - 1
    Примечание: scale берётся пер-тензорно, как в оригинале DoReFa.
    """
    def __init__(self, bits_w: int = 8):
        super().__init__()
        assert bits_w >= 1
        self.bits_w = bits_w
        self.register_buffer("eps", torch.tensor(1e-8))

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        w_t = torch.tanh(w)
        amax = w_t.abs().max().clamp(min=self.eps)
        w_n = w_t / (2 * amax) + 0.5                # -> [0,1]
        w_k = quantize_kbits_unit(w_n, self.bits_w) # k-бит
        w_q = 2.0 * w_k - 1.0                       # -> [-1,1]
        # STE уже реализован внутри quantize_kbits_unit
        return w_q


class DoReFaActQuant(nn.Module):
    """
    DoReFa для активаций.
    - Если signed=False: квантуем в [0,1] (классический вариант DoReFa).
        preproc='clip'  : x01 = clip(x, 0, 1)
        preproc='tanh'  : x01 = (tanh(x)/(2*max|tanh(x)|) + 0.5)
    - Если signed=True : приводим к [-1,1] через tanh-нормализацию, квантуем, возвращаем [-1,1].
    """
    def __init__(self, bits_a: int = 8, signed: bool = False, preproc: str = "tanh"):
        super().__init__()
        assert bits_a >= 1
        assert preproc in ("clip", "tanh")
        self.bits_a = bits_a
        self.signed = signed
        self.preproc = preproc
        self.register_buffer("eps", torch.tensor(1e-8))

    def _to_unit_from_clip(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(0.0, 1.0)

    def _to_unit_from_tanh(self, x: torch.Tensor) -> torch.Tensor:
        xt = torch.tanh(x)
        amax = xt.abs().max().clamp(min=self.eps)
        return xt / (2 * amax) + 0.5  # [0,1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.signed:
            # unsigned [0,1]
            if self.preproc == "clip":
                x01 = self._to_unit_from_clip(x)
            else:
                x01 = self._to_unit_from_tanh(x)
            xk = quantize_kbits_unit(x01, self.bits_a)     # ∈[0,1]
            return xk
        else:
            # signed [-1,1]: через tanh-нормализацию
            xt = torch.tanh(x)
            amax = xt.abs().max().clamp(min=self.eps)
            xsn = xt / amax                               # ∈[-1,1]
            # маппинг в [0,1] -> квант -> обратно
            x01 = (xsn + 1.0) * 0.5
            xk01 = quantize_kbits_unit(x01, self.bits_a)
            xsq = 2.0 * xk01 - 1.0
            return xsq
