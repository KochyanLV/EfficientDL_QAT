from lstm.model.LstmDorefa import QuantLSTMDoReFa
from lstm.model.Base import BaseLSTM

from lstm.train_utils.cycle import evaluate
from lstm.train_utils.loaders import make_loaders
from lstm.train_utils.cycle import compute_metrics, to_device

import torch
import torch.nn as nn
from torch.ao.quantization import quantize_dynamic
from typing import List, Tuple, Dict
import copy

from itertools import islice
import time

import logging
logger = logging.getLogger(__name__)


def to_int8_dynamic(model_fp32: torch.nn.Module) -> torch.nn.Module:
    m = copy.deepcopy(model_fp32).cpu().eval()
    qmodel = quantize_dynamic(m, {torch.nn.LSTM}, dtype=torch.qint8)
    return qmodel


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    num_batches: int = None,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    n_samples = 0

    logits_all = []
    labels_all = []
    
    iterator = islice(loader, num_batches) if num_batches else loader
    
    start_time = time.time()
    for batch in iterator:
        batch = to_device(batch, device)
        input_ids = batch["input_ids"]
        labels = batch["labels"].float()

        logits = model(input_ids).squeeze(-1)

        loss = loss_fn(logits, labels)
        bs = input_ids.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

        logits_all.append(logits)
        labels_all.append(labels)
    end_time = time.time()
    
    logger.info(f'It takes {end_time - start_time} seconds')

    logits_all = torch.cat(logits_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    metrics = compute_metrics(logits_all, labels_all)
    avg_loss = total_loss / max(n_samples, 1)
    logger.info(f'Metrics: {metrics}')
    logger.info(f'Loss: {avg_loss}')
    return avg_loss, metrics


if __name__ == '__main__':
    
    #### BEST Model
    model_best = QuantLSTMDoReFa(
        vocab_size=5000,
        emb_dim=128,
        hidden_dim=256,
        num_classes=1,
        bits_a=8,
        bits_w=8,
        act_signed=True,
        act_preproc='tanh'
    )

    weights_best = torch.load("lstm/checkpoints/quantlstmdorefa_vocab_size5000_emb_dim128_hidden_dim256_num_classes1_bits_a8_bits_w8_act_signedTrue_act_preproctanh_epochs5_bs128_lr0p001.pt")
    model_best.load_state_dict(weights_best)

    model_best.eval().cpu()
    int8_model_best = to_int8_dynamic(model_best)

    #### Base Model
    model_base = BaseLSTM(
        vocab_size=5000,
        emb_dim=256,
        hidden_dim=512,
        num_classes=1,
    )
    weights_base = torch.load("lstm/checkpoints/baselstm_vocab_size5000_emb_dim256_hidden_dim512_num_classes1_epochs10_bs64_lr0p0003.pt")
    model_base.load_state_dict(weights_base)

    model_base.eval().cpu()
    int8_model_base = to_int8_dynamic(model_base)

    # loaders
    train_loader, test_loader, _ = make_loaders(tokenizer_path="lstm/own_tokenizer", batch_size=8, max_len=512)
    
    
    evaluate(model_base, loader=test_loader, device='cpu')
    evaluate(int8_model_base, loader=test_loader, device='cpu')
    evaluate(model_best, loader=test_loader, device='cpu')
    evaluate(int8_model_best, loader=test_loader, device='cpu')