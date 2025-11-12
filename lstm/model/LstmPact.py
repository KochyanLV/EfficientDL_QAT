from qats.Pact import PACTAct, UniformWeightQuant
from lstm.model.Base import BaseLSTM
import torch


class QuantLSTMPACT(BaseLSTM):
    """
    BaseLSTM + PACT on activations + symmetric weight fake-quant on the head.
      - PACT after Embedding
      - PACT after LSTM output
      - Uniform symmetric per-channel fake-quant for Linear head weights
    """
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes=1, bits_act=8, bits_w=8, pact_init_alpha=6.0):
        super().__init__(vocab_size, emb_dim, hidden_dim, num_classes)
        self.pact_in = PACTAct(bits=bits_act, init_alpha=pact_init_alpha)
        self.pact_lstm = PACTAct(bits=bits_act, init_alpha=pact_init_alpha)
        self.wq_head = UniformWeightQuant(bits=bits_w, per_channel=True, ch_axis=0)

    def quant_embed_out(self, x: torch.Tensor) -> torch.Tensor:
        return self.pact_in(x)

    def quant_lstm_out(self, h: torch.Tensor) -> torch.Tensor:
        return self.pact_lstm(h)

    def quant_head_weight(self, w: torch.Tensor) -> torch.Tensor:
        return self.wq_head(w)

    def pact_regularization(self, lam: float = 1e-4) -> torch.Tensor:
        return self.pact_in.l2_alpha(lam) + self.pact_lstm.l2_alpha(lam)