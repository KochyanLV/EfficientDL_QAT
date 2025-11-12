from qats.AdaRound import AdaRoundWeightQuant
from lstm.model.Base import BaseLSTM
import torch


class QuantLSTMAdaRound(BaseLSTM):
    """
    BaseLSTM + AdaRound для весов Linear-головы.
    Активируем только веса головы (классический PTQ-бейзлайн для сравнения с QAT).
    """
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes=1, bits_w=8):
        super().__init__(vocab_size, emb_dim, hidden_dim, num_classes)
        self.wq_head = AdaRoundWeightQuant(bits=bits_w, per_channel=True, ch_axis=0)

    def quant_head_weight(self, w: torch.Tensor) -> torch.Tensor:
        return self.wq_head(w)

    def adaround_regularization(self, lam: float = 1e-4) -> torch.Tensor:
        return self.wq_head.regularization(lam)

    def set_eval_hard_round(self, use_hard: bool = True):
        self.wq_head.set_hard_round(use_hard)