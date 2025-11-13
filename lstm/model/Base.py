import torch.nn as nn
import torch.nn.functional as F
from lstm.train_utils.save_checkpoint import record_init


@record_init
class BaseLSTM(nn.Module):
    """
    База для простых LSTM-классификаторов.
    В неё можно легко добавить квантизацию, переопределив hook-методы:
      - quant_embed_out(x)
      - quant_lstm_out(h)
      - quant_head_weight(w)
    По умолчанию это no-op.
    """
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes=1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, num_classes)

    def quant_embed_out(self, x):
        return x

    def quant_lstm_out(self, h):
        return h

    def quant_head_weight(self, w):
        return w

    def forward(self, x):
        x = self.emb(x)           # (B, T, E)
        x = self.quant_embed_out(x)

        out, _ = self.lstm(x)     # (B, T, H)
        out = self.quant_lstm_out(out)

        pooled = out.mean(dim=1)  # (B, H)

        w_q = self.quant_head_weight(self.head.weight)
        logits = F.linear(pooled, w_q, self.head.bias)
        return logits