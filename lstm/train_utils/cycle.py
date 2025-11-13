import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import math
from typing import Dict, Tuple
from tqdm import tqdm
import json
from pathlib import Path

import torch
import torch.nn as nn

from lstm.train_utils.save_checkpoint import save_ckpt_from_init


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    logits: (N,) raw scores (до сигмоиды)
    labels: (N,) 0/1
    """
    probs = sigmoid(logits).detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy().astype(int)
    y_pred = (probs >= 0.5).astype(int)

    try:
        auc = roc_auc_score(y_true, probs)
    except Exception:
        auc = float("nan")

    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    return {"roc_auc": auc, "f1": f1, "precision": prec, "recall": rec}


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n_samples = 0

    for batch in tqdm(loader, total=len(loader)):
        batch = to_device(batch, device)
        input_ids = batch["input_ids"]        # (B, T)
        labels = batch["labels"].float()      # (B,)

        logits = model(input_ids).squeeze(-1) # (B,)

        loss = loss_fn(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = input_ids.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

    return total_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    n_samples = 0

    logits_all = []
    labels_all = []

    for batch in tqdm(loader, total=len(loader)):
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

    logits_all = torch.cat(logits_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    metrics = compute_metrics(logits_all, labels_all)
    avg_loss = total_loss / max(n_samples, 1)
    return avg_loss, metrics


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = epochs * max(len(train_loader), 1)
    def lr_lambda(step):
        return 0.5 * (1 + math.cos(math.pi * step / max(total_steps, 1)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_auc = -1.0
    best_state = None
    history = []
    global_step = 0
    for epoch in tqdm(range(1, epochs + 1)):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_metrics = evaluate(model, val_loader, device)
        scheduler.step()
        global_step += len(train_loader)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_auc={val_metrics['roc_auc']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_prec={val_metrics['precision']:.4f} | "
            f"val_rec={val_metrics['recall']:.4f}"
        )
        
        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "roc_auc": float(val_metrics["roc_auc"]),
            "f1": float(val_metrics["f1"]),
            "precision": float(val_metrics["precision"]),
            "recall": float(val_metrics["recall"]),
        })

        if not math.isnan(val_metrics["roc_auc"]) and val_metrics["roc_auc"] > best_auc:
            best_auc = val_metrics["roc_auc"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best val AUC: {best_auc:.4f}")
    
    
    ckpt_path = save_ckpt_from_init(
        model,
        extra={"epochs": epochs, "bs": bs, "lr": lr},
        with_values=True,
    )
    
    ckpt_path = Path(ckpt_path)
    metrics_path = ckpt_path.with_suffix(".json") 
    
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    
    return model