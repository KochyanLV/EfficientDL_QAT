from qats.Apot import APoTQuantRCF
from lstm.model.Base import BaseLSTM
import torch


class QuantLSTMAPoT(BaseLSTM):
    """
    BaseLSTM + APoTQuantRCF:
      - APoT на активациях (per-tensor)
      - APoT на весах головы (per-channel по out_features), с опцией WeightNorm
    Параметры по умолчанию: bits=8, k=2.
    """
    def __init__(
        self,
        vocab_size,
        emb_dim,
        hidden_dim,
        num_classes=1,
        bits=8,
        k=2,
        init_alpha_act=6.0,
        init_alpha_w=6.0,
        use_weight_norm_w=True,
    ):
        super().__init__(vocab_size, emb_dim, hidden_dim, num_classes)

        # активации: per-tensor
        self.apot_act_in = APoTQuantRCF(bits=bits, k=k, per_channel=False, init_alpha=init_alpha_act)
        self.apot_act_lstm = APoTQuantRCF(bits=bits, k=k, per_channel=False, init_alpha=init_alpha_act)

        self.apot_w_head = APoTQuantRCF(
            bits=bits, k=k, per_channel=True, ch_axis=0, init_alpha=init_alpha_w, use_weight_norm=use_weight_norm_w
        )

    def quant_embed_out(self, x: torch.Tensor) -> torch.Tensor:
        return self.apot_act_in(x, is_weight=False)

    def quant_lstm_out(self, h: torch.Tensor) -> torch.Tensor:
        return self.apot_act_lstm(h, is_weight=False)

    def quant_head_weight(self, w: torch.Tensor) -> torch.Tensor:
        return self.apot_w_head(w, is_weight=True)
