from qats.LSQ import LSQQuant
from lstm.model.Base import BaseLSTM
import torch

from lstm.train_utils.save_checkpoint import record_init


@record_init
class QuantLSTMLSQ(BaseLSTM):
    """
    BaseLSTM + LSQ fake quant:
      - LSQ on activations after Embedding (per-tensor)
      - LSQ on activations after LSTM output (per-tensor)
      - LSQ on head weights (per-channel along out_features)
    """
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes=1, bits=8):
        super().__init__(vocab_size, emb_dim, hidden_dim, num_classes)
        # activations: per-tensor LSQ
        self.lsq_act_in = LSQQuant(bits=bits, per_channel=False)
        self.lsq_act_lstm = LSQQuant(bits=bits, per_channel=False)
        self.lsq_w_head = LSQQuant(bits=bits, per_channel=True, ch_axis=0)

    def quant_embed_out(self, x: torch.Tensor) -> torch.Tensor:
        return self.lsq_act_in(x)

    def quant_lstm_out(self, h: torch.Tensor) -> torch.Tensor:
        return self.lsq_act_lstm(h)

    def quant_head_weight(self, w: torch.Tensor) -> torch.Tensor:
        return self.lsq_w_head(w)