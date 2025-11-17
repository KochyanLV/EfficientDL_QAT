import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import math
import json
from pathlib import Path
import logging

from espcn.train_utils.save_checkpoint import save_ckpt_from_init

logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not installed. Install with: pip install wandb")


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (smooth L1).
    Часто используется в SR (ESRGAN, EDSR).
    L = sqrt((x - y)^2 + eps^2)
    """
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()


def ssim(img1, img2, window_size=11, size_average=True):
    """
    SSIM (Structural Similarity Index).
    Простая реализация без зависимостей.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def psnr(mse: float) -> float:
    """
    Calculate PSNR from MSE.
    PSNR = 10 * log10(1 / MSE)
    Assumes pixel values are in [0, 1] range.
    """
    if mse == 0:
        return float('inf')
    return 10 * math.log10(1.0 / mse)


def compute_psnr(sr: torch.Tensor, hr: torch.Tensor) -> float:
    """
    Compute PSNR between super-resolved and high-resolution images.
    
    Args:
        sr: Super-resolved images (B, C, H, W) in [0, 1]
        hr: High-resolution images (B, C, H, W) in [0, 1]
    
    Returns:
        Average PSNR across batch
    """
    mse = torch.mean((sr - hr) ** 2).item()
    return psnr(mse)


def compute_ssim(sr: torch.Tensor, hr: torch.Tensor) -> float:
    """
    Compute SSIM between super-resolved and high-resolution images.
    
    Args:
        sr: Super-resolved images (B, C, H, W) in [0, 1]
        hr: High-resolution images (B, C, H, W) in [0, 1]
    
    Returns:
        Average SSIM across batch
    """
    return ssim(sr, hr).item()


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    device: torch.device,
    loss_fn,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    n_samples = 0

    for lr_batch, hr_batch in loader:
        lr_batch = lr_batch.to(device)
        hr_batch = hr_batch.to(device)
        
        sr_batch = model(lr_batch)
        
        loss = loss_fn(sr_batch, hr_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = lr_batch.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

    return total_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    loss_fn,
) -> tuple[float, float, float]:
    """
    Evaluate model and return loss, PSNR, and SSIM.
    """
    model.eval()

    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n_samples = 0
    
    for lr_batch, hr_batch in loader:
        lr_batch = lr_batch.to(device)
        hr_batch = hr_batch.to(device)
        
        sr_batch = model(lr_batch)
        
        loss = loss_fn(sr_batch, hr_batch)
        
        sr_clamped = sr_batch.clamp(0, 1)
        hr_clamped = hr_batch.clamp(0, 1)
        
        batch_psnr = compute_psnr(sr_clamped, hr_clamped)
        batch_ssim = compute_ssim(sr_clamped, hr_clamped)
        
        bs = lr_batch.size(0)
        total_loss += loss.item() * bs
        total_psnr += batch_psnr * bs
        total_ssim += batch_ssim * bs
        n_samples += bs

    avg_loss = total_loss / max(n_samples, 1)
    avg_psnr = total_psnr / max(n_samples, 1)
    avg_ssim = total_ssim / max(n_samples, 1)
    return avg_loss, avg_psnr, avg_ssim


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    loss_type: str = 'l1',
    use_wandb: bool = False,
    wandb_project: str = 'espcn-qat',
    wandb_name: str = None,
):
    """
    Train ESPCN model.
    
    Args:
        loss_type: 'l1', 'l2', or 'charbonnier'
        use_wandb: Enable Weights & Biases logging
        wandb_project: W&B project name
        wandb_name: W&B run name (defaults to model class name)
    """
    model.to(device)
    
    # Separate learning rates for quantization parameters (LSQ scales, PACT alpha, etc.)
    quant_params = []
    model_params = []
    
    for name, param in model.named_parameters():
        if 'alpha' in name or 'scale' in name or 's_' in name:
            quant_params.append(param)
        else:
            model_params.append(param)
    
    if quant_params:
        # Quantization params need higher LR (10x)
        optimizer = optim.Adam([
            {'params': model_params, 'lr': lr},
            {'params': quant_params, 'lr': lr * 10}
        ], weight_decay=weight_decay)
        logger.info(f"Using separate LR for quantization params: {lr*10:.2e}")
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Initialize W&B if requested
    if use_wandb and WANDB_AVAILABLE:
        if wandb_name is None:
            wandb_name = model.__class__.__name__
        
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            config={
                'model': model.__class__.__name__,
                'epochs': epochs,
                'batch_size': train_loader.batch_size,
                'lr': lr,
                'weight_decay': weight_decay,
                'loss_type': loss_type,
                'optimizer': 'Adam',
                'scale_factor': getattr(model, 'scale_factor', None),
                'feature_dim': getattr(model, 'feature_dim', None),
            }
        )
        wandb.watch(model, log='all', log_freq=100)
        logger.info(f"W&B logging enabled: {wandb_project}/{wandb_name}")
    elif use_wandb and not WANDB_AVAILABLE:
        logger.warning("W&B requested but not available. Install with: pip install wandb")
        use_wandb = False

    # Loss function
    if loss_type == 'l1':
        loss_fn = nn.L1Loss()
        logger.info("Using L1 Loss")
    elif loss_type == 'charbonnier':
        loss_fn = CharbonnierLoss()
        logger.info("Using Charbonnier Loss")
    else:
        loss_fn = nn.MSELoss()
        logger.info("Using L2 (MSE) Loss")

    total_steps = epochs * max(len(train_loader), 1)
    def lr_lambda(step):
        return 0.5 * (1 + math.cos(math.pi * step / max(total_steps, 1)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    bs = train_loader.batch_size

    best_psnr = -1.0
    best_state = None
    history = []
    global_step = 0
    
    # Warmup: disable quantization for first 2 epochs
    warmup_epochs = 2
    if hasattr(model, 'quant_enabled'):
        model.quant_enabled = False
        logger.info(f"Warmup: quantization disabled for first {warmup_epochs} epochs")
    
    for epoch in range(1, epochs + 1):
        # Enable quantization after warmup
        if hasattr(model, 'quant_enabled') and epoch > warmup_epochs:
            if not model.quant_enabled:
                model.quant_enabled = True
                logger.info("Warmup complete - quantization enabled")
        
        logger.info(f"Train model. EPOCH: {epoch} // {epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        
        logger.info(f"Validate model. EPOCH: {epoch} // {epochs}")
        val_loss, val_psnr, val_ssim = evaluate(model, val_loader, device, loss_fn)
        scheduler.step()
        global_step += len(train_loader)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_psnr={val_psnr:.2f} dB | "
            f"val_ssim={val_ssim:.4f}"
        )
        
        # Log to W&B
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'val/loss': val_loss,
                'val/psnr': val_psnr,
                'val/ssim': val_ssim,
                'learning_rate': scheduler.get_last_lr()[0],
            })
        
        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "psnr": float(val_psnr),
            "ssim": float(val_ssim),
        })

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            # Only save state after warmup (to avoid shape mismatch issues)
            if not (hasattr(model, 'quant_enabled') and epoch <= warmup_epochs):
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
            # Log best metrics to W&B
            if use_wandb and WANDB_AVAILABLE:
                wandb.run.summary['best_psnr'] = best_psnr
                wandb.run.summary['best_ssim'] = val_ssim
                wandb.run.summary['best_epoch'] = epoch

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    print(f"Best val PSNR: {best_psnr:.2f} dB")
    
    # Get best epoch info
    best_epoch_idx = max(range(len(history)), key=lambda i: history[i]['psnr'])
    best_epoch_info = history[best_epoch_idx]
    best_ssim = best_epoch_info['ssim']
    best_epoch_num = best_epoch_info['epoch']
    
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
    
    # Save model to W&B
    if use_wandb and WANDB_AVAILABLE:
        wandb.save(str(ckpt_path))
        wandb.save(str(metrics_path))
        wandb.finish()
        logger.info("W&B run finished and saved")
    
    # Return model and best metrics
    return {
        'model': model,
        'best_psnr': float(best_psnr),
        'best_ssim': float(best_ssim),
        'best_epoch': int(best_epoch_num),
        'history': history
    }

