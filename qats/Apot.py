import torch
import torch.nn as nn


class APoTQuantRCF(nn.Module):
    """
    APoT (Additive Powers-of-Two) с RCF:
      - bits: общее число бит (включая знак)
      - k: бит на один "банк" степеней двойки (обычно k=2)
      - n = (bits-1)//k банков; уровни строятся как сумма по банкам {0, 2^{-i}, 2^{-(i+n)}, ...}
      - alpha — обучаемый клиппинг (RCF-стиль): квантование идёт над x/alpha ∈ [-1, 1]
      - per_channel для весов по оси ch_axis (обычно 0 = out_features)
      - use_weight_norm: применять WN к весам перед квантованием (рекомендуется для weight)
    """
    def __init__(
        self,
        bits: int = 8,
        k: int = 2,
        per_channel: bool = False,
        ch_axis: int = 0,
        init_alpha: float = 6.0,
        use_weight_norm: bool = False,
    ):
        super().__init__()
        assert bits >= 2 and (bits - 1) % k == 0, "bits-1 должно делиться на k"
        self.bits = bits
        self.k = k
        self.n = (bits - 1) // k
        self.per_channel = per_channel
        self.ch_axis = ch_axis
        self.use_weight_norm = use_weight_norm

        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        self.register_buffer("eps", torch.tensor(1e-8))

        # Предрасчёт банков уровней для каждого терма i
        # bank_i = {0, 2^{-i}, 2^{-(i+n)}, ..., 2^{-(i+(2^k-2)n)}}
        banks = []
        levels_per_bank = 1 << self.k  # 2^k
        for i in range(self.n):
            exps = [i + t * self.n for t in range(levels_per_bank - 1)]  # без 0, его добавим явно
            bank = torch.tensor([0.0] + [2.0 ** (-e) for e in exps], dtype=torch.float32)
            banks.append(bank)
        self.register_buffer("banks", nn.utils.rnn.pad_sequence(banks, batch_first=True))  # [n, 2^k]

        # Максимально достижимая сумма (в нормализованном пространстве [0,1])
        with torch.no_grad():
            Lmax = 0.0
            for i in range(self.n):
                Lmax += max(banks[i]).item()
        self.register_buffer("Lmax", torch.tensor(Lmax, dtype=torch.float32))

    @staticmethod
    def _weight_norm(w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        mu = w.mean()
        sigma = w.std(unbiased=False)
        return (w - mu) / (sigma + eps)

    def _broadcast(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if not self.per_channel:
            return t
        view = [1] * x.dim()
        view[self.ch_axis] = -1
        return t.view(view)

    def _apot_project_unit(self, a: torch.Tensor) -> torch.Tensor:
        """
        Проекция |x_norm|∈[0,1] в APoT-решётку:
        greedy: для каждого банка выбираем ближайший уровень из bank_i, накапливая сумму.
        """
        # a: |x_norm| ∈ [0, 1]
        res = a
        acc = torch.zeros_like(a)
        for i in range(self.n):
            bank = self.banks[i].to(dtype=a.dtype, device=a.device)  # shape [2^k]
            # broadcast bank -> (..., 2^k)
            choices = bank.view(*([1] * a.dim()), -1).expand(*a.shape, bank.numel())
            diff = (res.unsqueeze(-1) - choices).abs()
            idx = diff.argmin(dim=-1)                           # (B, T, E)
            chosen = torch.gather(choices, -1, idx.unsqueeze(-1)).squeeze(-1)

            acc = acc + chosen
            res = (res - chosen).clamp(min=0.0)

        return (acc / (self.Lmax + self.eps)).clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor, is_weight: bool = False) -> torch.Tensor:
        """
        x — активации/веса. Если is_weight=True и use_weight_norm=True, применим WN.
        Квантование: x/alpha -> APoT на [-1,1], затем умножаем обратно на alpha, градиент — STE.
        """
        alpha = self.alpha.to(dtype=x.dtype, device=x.device)

        z = x
        if is_weight and self.use_weight_norm:
            z = self._weight_norm(z)

        x_norm = (z / (alpha + self.eps)).clamp(-1.0, 1.0)
        sign = x_norm.sign()
        a = x_norm.abs()  # [0,1]

        a_hat01 = self._apot_project_unit(a)         # в [0,1]
        xq_norm = sign * a_hat01                     # в [-1,1]

        y = xq_norm * alpha
        y = (z - z.detach()) + y.detach()            # STE через RCF-переменную z

        return y

