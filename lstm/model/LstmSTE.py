import torch
import torch.nn as nn

from qats.FakeSte import FakeQuantSTE
from lstm.model.Base import BaseLSTM

class QuantLSTMSTE(BaseLSTM):
    """
    Наследник BaseLSTM с простейшей fake-квантовкой на активациях и весах головы.
      - AQ после Embedding и после LSTM выхода (пер-тензорные scale)
      - WQ для веса Linear-головы (пер-тензорный scale)
    """
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes=1, bits=8):
        super().__init__(vocab_size, emb_dim, hidden_dim, num_classes)
        self.fq = FakeQuantSTE(bits)
        self.s_act_in = nn.Parameter(torch.tensor(1.0))
        self.s_act_lstm = nn.Parameter(torch.tensor(1.0))
        self.s_w_head = nn.Parameter(torch.tensor(1.0))

    def quant_embed_out(self, x: torch.Tensor) -> torch.Tensor:
        return self.fq(x, self.s_act_in)

    def quant_lstm_out(self, h: torch.Tensor) -> torch.Tensor:
        return self.fq(h, self.s_act_lstm)

    def quant_head_weight(self, w: torch.Tensor) -> torch.Tensor:
        return self.fq(w, self.s_w_head)