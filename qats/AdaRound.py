import torch
import torch.nn as nn


class AdaRoundWeightQuant(nn.Module):
    """
    AdaRound для весов:
      w_q = s * clamp( floor(w/s) + r , [-Q, Q] ), где r = sigmoid(alpha) (soft train) или {0,1} (hard eval).
    - bits: INT8 по умолчанию, симметрично
    - per_channel по оси out_features (ch_axis=0)
    - s фиксируется по max(|w|)/Q (PTQ-стиль)
    - alpha обучается; инициализируется так, чтобы r≈фракции (y - floor(y)) при первом проходе
    """
    def __init__(self, bits: int = 8, per_channel: bool = True, ch_axis: int = 0):
        super().__init__()
        assert bits >= 2
        self.bits = bits
        self.Q = 2 ** (bits - 1) - 1  # signed
        self.per_channel = per_channel
        self.ch_axis = ch_axis

        self.register_buffer("eps", torch.tensor(1e-8))
        self.register_buffer("initialized", torch.tensor(False))
        self.register_buffer("alpha_init", torch.tensor(False))

        # s и alpha создадим/уточним при первом forward (зависит от формы весов)
        self.s = None
        self.alpha = None

        # переключатель мягкого/жёсткого округления (eval -> hard по умолчанию)
        self.hard_round_in_eval = True

    def _broadcast(self, w: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if not self.per_channel:
            return t
        view = [1] * w.dim()
        view[self.ch_axis] = -1
        return t.view(view)

    @torch.no_grad()
    def _init_scale(self, w: torch.Tensor):
        if self.per_channel:
            w_perm = w.transpose(0, self.ch_axis).contiguous().flatten(1)
            s = (w_perm.abs().max(dim=1).values / max(self.Q, 1)).clamp(min=1e-8)
        else:
            s = (w.abs().max() / max(self.Q, 1)).clamp(min=1e-8)
        # хранить в nn.Parameter не нужно (PTQ-стиль, без градиента)
        self.s = s.detach()
        self.initialized.fill_(True)

    @torch.no_grad()
    def _init_alpha(self, w: torch.Tensor):
        """
        Инициализация alpha так, чтобы sigmoid(alpha) ≈ дробная часть (y - floor(y)),
        где y = w/s. Это старт с “обычного” округления.
        """
        s_b = self._broadcast(w, self.s)
        y = (w / s_b).detach()
        k = torch.floor(y)
        f = (y - k).clamp(0, 1 - 1e-6)  # дробная часть в [0,1)

        # alpha = logit(f) = log(f/(1-f))
        alpha = torch.log(f / (1 - f + 1e-6) + 1e-12)

        # перетаскиваем на тот же девайс/тип, что и w
        self.alpha = nn.Parameter(alpha.to(dtype=w.dtype, device=w.device))
        self.alpha_init.fill_(True)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        # ленивые инициализации
        if not self.initialized:
            self._init_scale(w)
        if (self.alpha is None) or (not self.alpha_init):
            self._init_alpha(w)

        s_b = self._broadcast(w, self.s.to(w.device, w.dtype))
        y = w / s_b
        k = torch.floor(y)

        r_soft = torch.sigmoid(self.alpha)  # ∈ (0,1)
        if (not self.training) and self.hard_round_in_eval:
            r = (r_soft >= 0.5).to(w.dtype)
        else:
            r = r_soft

        z = k + r
        z_clamped = z.clamp(-self.Q, self.Q)
        w_q = s_b * ((z_clamped - z_clamped.detach()) + z_clamped.detach().round())  # небольшая стабилизация


        return w_q

    def regularization(self, lam: float = 1e-4) -> torch.Tensor:
        """
        Регуляризатор, подталкивающий r к {0,1}.
        r = sigmoid(alpha). Тогда 1 - |2r-1| == 0, когда r∈{0,1}, и максимум при r=0.5.
        """
        r = torch.sigmoid(self.alpha)
        reg_term = (1.0 - (2.0 * r - 1.0).abs()).mean()
        return lam * reg_term

    def set_hard_round(self, hard_in_eval: bool = True):
        self.hard_round_in_eval = hard_in_eval

