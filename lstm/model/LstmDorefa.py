from qats.DoReFa import DoReFaActQuant, DoReFaWeightQuant
from lstm.model.Base import BaseLSTM
import torch
from lstm.train_utils.save_checkpoint import record_init


@record_init
class QuantLSTMDoReFa(BaseLSTM):
    """
    BaseLSTM + DoReFa:
      - активации после Embedding и после LSTM
      - веса головы (Linear) через DoReFaWeightQuant
    По умолчанию:
      * активации делаем signed=True (т.к. выходы Embedding/LSTM могут быть отрицательными)
      * если хочешь строго следовать DoReFa для ReLU-активаций — поставь signed=False и preproc='clip'
    """
    def __init__(
        self,
        vocab_size,
        emb_dim,
        hidden_dim,
        num_classes=1,
        bits_a=8,
        bits_w=8,
        act_signed=True,
        act_preproc="tanh",
    ):
        super().__init__(vocab_size, emb_dim, hidden_dim, num_classes)
        self.dorefa_act_in = DoReFaActQuant(bits_a=bits_a, signed=act_signed, preproc=act_preproc)
        self.dorefa_act_lstm = DoReFaActQuant(bits_a=bits_a, signed=act_signed, preproc=act_preproc)
        self.dorefa_w_head = DoReFaWeightQuant(bits_w=bits_w)

    # хуки BaseLSTM
    def quant_embed_out(self, x: torch.Tensor) -> torch.Tensor:
        return self.dorefa_act_in(x)

    def quant_lstm_out(self, h: torch.Tensor) -> torch.Tensor:
        return self.dorefa_act_lstm(h)

    def quant_head_weight(self, w: torch.Tensor) -> torch.Tensor:
        return self.dorefa_w_head(w)
