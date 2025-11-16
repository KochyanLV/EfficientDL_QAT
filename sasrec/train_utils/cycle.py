import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from typing import Dict, Tuple
from tqdm import tqdm
import json
from pathlib import Path
import logging

from sasrec.train_utils.save_checkpoint import save_ckpt_from_init

logger = logging.getLogger(__name__)


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def compute_ndcg_at_k(predictions: torch.Tensor, targets: torch.Tensor, k: int = 10) -> float:
    """
    Compute NDCG@k for a batch of predictions.
    
    Args:
        predictions: (batch_size, num_items) - predicted scores for all items
        targets: (batch_size,) - ground truth item indices
        k: cutoff for NDCG calculation
        
    Returns:
        NDCG@k score
    """
    batch_size = predictions.size(0)
    
    _, top_k_indices = torch.topk(predictions, k=min(k, predictions.size(1)), dim=1)
    targets_expanded = targets.unsqueeze(1).expand_as(top_k_indices)
    hits = (top_k_indices == targets_expanded).float()
    
    positions = torch.arange(1, k + 1, dtype=torch.float32, device=predictions.device)
    dcg = (hits / torch.log2(positions + 1)).sum(dim=1)
    idcg = 1.0 / math.log2(2)
    ndcg = dcg / idcg
    
    return ndcg.mean().item()


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Compute various metrics for recommendation.
    
    Args:
        logits: (batch_size, num_items) - predicted scores
        targets: (batch_size,) - ground truth item indices
        
    Returns:
        Dictionary with metrics
    """
    with torch.no_grad():
        ndcg_10 = compute_ndcg_at_k(logits, targets, k=10)
        ndcg_5 = compute_ndcg_at_k(logits, targets, k=5)
        ndcg_20 = compute_ndcg_at_k(logits, targets, k=20)
        
        _, top_10 = torch.topk(logits, k=min(10, logits.size(1)), dim=1)
        targets_expanded = targets.unsqueeze(1).expand_as(top_10)
        hit_10 = (top_10 == targets_expanded).any(dim=1).float().mean().item()
        
    return {
        'ndcg@10': ndcg_10,
        'ndcg@5': ndcg_5,
        'ndcg@20': ndcg_20,
        'hit@10': hit_10
    }


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    device: torch.device,
    grad_clip: float = 5.0,
) -> float:
    """Train for one epoch"""
    model.train()
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = 0.0
    n_samples = 0

    for batch in loader:
        batch = to_device(batch, device)
        sequences = batch["sequence"]
        targets = batch["target"]
        
        logits = model(sequences)
        loss = loss_fn(logits, targets)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        bs = sequences.size(0)
        total_loss += loss.item() * bs
        n_samples += bs
    
    return total_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model on validation/test set"""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    
    total_loss = 0.0
    n_samples = 0
    
    all_logits = []
    all_targets = []
    
    for batch in loader:
        batch = to_device(batch, device)
        sequences = batch["sequence"]
        targets = batch["target"]
        
        logits = model(sequences)
        
        loss = loss_fn(logits, targets)
        bs = sequences.size(0)
        total_loss += loss.item() * bs
        n_samples += bs
        
        all_logits.append(logits)
        all_targets.append(targets)
    
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_logits, all_targets)
    avg_loss = total_loss / max(n_samples, 1)
    
    return avg_loss, metrics


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
):
    """
    Train SASRec model with various quantization methods.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    total_steps = epochs * max(len(train_loader), 1)
    def lr_lambda(step):
        return 0.5 * (1 + math.cos(math.pi * step / max(total_steps, 1)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    bs = train_loader.batch_size
    
    best_ndcg = -1.0
    best_state = None
    history = []
    
    for epoch in range(1, epochs + 1):
        logger.info(f"Train model. EPOCH: {epoch} // {epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        
        logger.info(f"Validate model. EPOCH: {epoch} // {epochs}")
        val_loss, val_metrics = evaluate(model, val_loader, device)
        
        scheduler.step()
        
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"NDCG@10={val_metrics['ndcg@10']:.4f} | "
            f"NDCG@5={val_metrics['ndcg@5']:.4f} | "
            f"Hit@10={val_metrics['hit@10']:.4f}"
        )
        
        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "ndcg@10": float(val_metrics["ndcg@10"]),
            "ndcg@5": float(val_metrics["ndcg@5"]),
            "ndcg@20": float(val_metrics["ndcg@20"]),
            "hit@10": float(val_metrics["hit@10"]),
        })
        
        if val_metrics["ndcg@10"] > best_ndcg:
            best_ndcg = val_metrics["ndcg@10"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best NDCG@10: {best_ndcg:.4f}")
    
    ckpt_path = save_ckpt_from_init(
        model,
        extra={"epochs": epochs, "bs": bs, "lr": lr},
        with_values=True,
    )
    
    logger.info(f"Save best model and metrics to: {ckpt_path}")
    
    ckpt_path = Path(ckpt_path)
    metrics_path = ckpt_path.with_suffix(".json")
    
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    
    return model

