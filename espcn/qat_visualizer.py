"""
QAT Visualizer for ESPCN - visualize quantization effects in conv layers
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path


class ESPCNQATVisualizer:
    """
    Captures and visualizes quantization effects in ESPCN models.
    Works with LSQ, PACT, AdaRound, APoT, DoReFa, STE.
    """
    
    def __init__(self, output_dir: str = "./qat_viz"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.activations = {}
        self.weights = {}
        
    def capture_activations(self, model, lr_batch):
        """Capture activations during forward pass."""
        model.eval()
        
        hooks = []
        captured = {}
        
        def make_hook(name):
            def hook(module, input, output):
                captured[name] = {
                    'input': input[0].detach().cpu() if isinstance(input, tuple) else input.detach().cpu(),
                    'output': output.detach().cpu()
                }
            return hook
        
        # Attach hooks to conv layers
        hooks.append(model.conv1.register_forward_hook(make_hook('conv1')))
        hooks.append(model.conv2.register_forward_hook(make_hook('conv2')))
        hooks.append(model.conv3.register_forward_hook(make_hook('conv3')))
        
        # Forward pass
        with torch.no_grad():
            _ = model(lr_batch)
        
        # Remove hooks
        for h in hooks:
            h.remove()
        
        self.activations = captured
        
        # Capture weights
        self.weights = {
            'conv1': model.conv1.weight.detach().cpu(),
            'conv2': model.conv2.weight.detach().cpu(),
            'conv3': model.conv3.weight.detach().cpu(),
        }
        
        return captured
    
    def plot_weight_distribution(self, model_name: str):
        """Plot weight distributions for all conv layers."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, (name, weight) in enumerate(self.weights.items()):
            w_flat = weight.flatten().numpy()
            
            axes[idx].hist(w_flat, bins=50, alpha=0.7, edgecolor='black', color='blue')
            axes[idx].set_title(f'{name} Weights\nMean: {w_flat.mean():.4f}, Std: {w_flat.std():.4f}')
            axes[idx].set_xlabel('Weight Value')
            axes[idx].set_ylabel('Count')
            axes[idx].grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_name} - Weight Distributions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / f'{model_name}_weights.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_activation_distribution(self, model_name: str):
        """Plot activation distributions for all conv layers."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        for idx, (name, data) in enumerate(self.activations.items()):
            # Input activations
            inp = data['input'][0].flatten().numpy()[:10000]  # Sample for speed
            axes[0, idx].hist(inp, bins=50, alpha=0.7, edgecolor='black', color='green')
            axes[0, idx].set_title(f'{name} Input\nMean: {inp.mean():.4f}, Std: {inp.std():.4f}')
            axes[0, idx].set_xlabel('Value')
            axes[0, idx].grid(True, alpha=0.3)
            
            # Output activations
            out = data['output'][0].flatten().numpy()[:10000]
            axes[1, idx].hist(out, bins=50, alpha=0.7, edgecolor='black', color='orange')
            axes[1, idx].set_title(f'{name} Output\nMean: {out.mean():.4f}, Std: {out.std():.4f}')
            axes[1, idx].set_xlabel('Value')
            axes[1, idx].grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_name} - Activation Distributions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / f'{model_name}_activations.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_quantization_comparison(self, models_dict, lr_batch):
        """
        Compare quantization effects across different QAT methods.
        
        Args:
            models_dict: {'Base': model_base, 'LSQ': model_lsq, ...}
            lr_batch: Sample LR batch for inference
        """
        method_names = list(models_dict.keys())
        n_methods = len(method_names)
        
        fig, axes = plt.subplots(n_methods, 3, figsize=(15, 4*n_methods))
        if n_methods == 1:
            axes = axes.reshape(1, -1)
        
        for row, (method, model) in enumerate(models_dict.items()):
            # Capture for this model
            self.capture_activations(model, lr_batch)
            
            # Plot conv1, conv2, conv3 weights
            for col, conv_name in enumerate(['conv1', 'conv2', 'conv3']):
                w = self.weights[conv_name].flatten().numpy()
                
                axes[row, col].hist(w, bins=50, alpha=0.7, edgecolor='black')
                axes[row, col].set_title(f'{method} - {conv_name}\nMean: {w.mean():.4f}, Std: {w.std():.4f}')
                axes[row, col].set_xlabel('Weight Value')
                axes[row, col].grid(True, alpha=0.3)
        
        plt.suptitle('QAT Methods Comparison - Weight Distributions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / 'qat_methods_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def get_model_statistics(self, model, model_name: str, lr_batch):
        """Get quantization statistics for a model."""
        self.capture_activations(model, lr_batch)
        
        stats = {
            'model_name': model_name,
            'weights': {},
            'activations': {},
        }
        
        # Weight stats
        for name, weight in self.weights.items():
            w_flat = weight.flatten().numpy()
            stats['weights'][name] = {
                'mean': float(w_flat.mean()),
                'std': float(w_flat.std()),
                'min': float(w_flat.min()),
                'max': float(w_flat.max()),
            }
        
        # Activation stats
        for name, data in self.activations.items():
            inp = data['input'][0].flatten().numpy()
            out = data['output'][0].flatten().numpy()
            
            stats['activations'][name] = {
                'input_mean': float(inp.mean()),
                'input_std': float(inp.std()),
                'output_mean': float(out.mean()),
                'output_std': float(out.std()),
            }
        
        return stats


def visualize_qat_methods(models_dict, lr_batch, output_dir='./qat_viz', use_wandb=False):
    """
    Visualize all QAT methods and optionally log to W&B.
    
    Args:
        models_dict: {'Base': model, 'LSQ': model, 'PACT': model, ...}
        lr_batch: Sample LR batch (B, C, H, W)
        output_dir: Directory to save plots
        use_wandb: If True, log visualizations to W&B
    
    Returns:
        Dictionary of visualization paths
    """
    visualizer = ESPCNQATVisualizer(output_dir=output_dir)
    viz_paths = {}
    
    # Generate visualizations for each method
    for method_name, model in models_dict.items():
        print(f"Visualizing {method_name}...")
        
        # Capture activations
        visualizer.capture_activations(model, lr_batch)
        
        # Generate plots
        weight_path = visualizer.plot_weight_distribution(method_name)
        activation_path = visualizer.plot_activation_distribution(method_name)
        
        viz_paths[method_name] = {
            'weights': weight_path,
            'activations': activation_path,
        }
        
        # Get statistics
        stats = visualizer.get_model_statistics(model, method_name, lr_batch)
        viz_paths[method_name]['stats'] = stats
    
    # Comparison plot
    comparison_path = visualizer.plot_quantization_comparison(models_dict, lr_batch)
    viz_paths['comparison'] = comparison_path
    
    # Log to W&B if enabled
    if use_wandb:
        try:
            import wandb
            
            # Log individual method visualizations
            for method_name, paths in viz_paths.items():
                if method_name == 'comparison':
                    continue
                
                wandb.log({
                    f'{method_name}/weights_dist': wandb.Image(paths['weights']),
                    f'{method_name}/activations_dist': wandb.Image(paths['activations']),
                })
                
                # Log statistics as summary
                if 'stats' in paths:
                    for key, value in paths['stats'].items():
                        if isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                wandb.run.summary[f'{method_name}/{key}/{subkey}'] = subvalue
            
            # Log comparison
            wandb.log({'qat_comparison': wandb.Image(comparison_path)})
            
            print("✅ Logged visualizations to W&B")
        except Exception as e:
            print(f"⚠️ Could not log to W&B: {e}")
    
    print(f"\n✅ Visualizations saved to: {output_dir}")
    return viz_paths


def visualize_sr_results(models_dict, lr_batch, hr_batch, output_dir='./qat_viz', use_wandb=False):
    """
    Visualize super-resolution results for all QAT methods.
    Shows LR → SR → HR comparison.
    
    Args:
        models_dict: {'Base': model, 'LSQ': model, ...}
        lr_batch: Low-resolution images (B, C, H, W)
        hr_batch: High-resolution ground truth (B, C, H, W)
        output_dir: Where to save visualizations
        use_wandb: Log to W&B
    
    Returns:
        Path to saved visualization
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_methods = len(models_dict)
    n_samples = min(4, lr_batch.size(0))  # Show up to 4 samples
    
    # Create figure: rows = methods, cols = samples × 3 (LR, SR, HR)
    fig, axes = plt.subplots(n_methods, n_samples * 3, figsize=(n_samples * 9, n_methods * 3))
    
    if n_methods == 1:
        axes = axes.reshape(1, -1)
    
    # Generate SR for all methods
    sr_results = {}
    for method_name, model in models_dict.items():
        model.eval()
        with torch.no_grad():
            sr_batch = model(lr_batch).clamp(0, 1)
        sr_results[method_name] = sr_batch
    
    # Plot
    for row, (method_name, sr_batch) in enumerate(sr_results.items()):
        for sample_idx in range(n_samples):
            # LR
            col_lr = sample_idx * 3
            lr_img = lr_batch[sample_idx].permute(1, 2, 0).cpu().numpy()
            axes[row, col_lr].imshow(lr_img)
            axes[row, col_lr].set_title(f'{method_name} - LR\n{lr_img.shape[0]}×{lr_img.shape[1]}')
            axes[row, col_lr].axis('off')
            
            # SR
            col_sr = sample_idx * 3 + 1
            sr_img = sr_batch[sample_idx].permute(1, 2, 0).cpu().numpy()
            axes[row, col_sr].imshow(sr_img)
            
            # Calculate PSNR for this sample
            mse = ((sr_batch[sample_idx] - hr_batch[sample_idx]) ** 2).mean().item()
            psnr = 10 * np.log10(1.0 / (mse + 1e-8))
            
            axes[row, col_sr].set_title(f'{method_name} - SR\n{sr_img.shape[0]}×{sr_img.shape[1]}\nPSNR: {psnr:.2f} dB')
            axes[row, col_sr].axis('off')
            
            # HR (ground truth) - same for all methods
            col_hr = sample_idx * 3 + 2
            hr_img = hr_batch[sample_idx].permute(1, 2, 0).cpu().numpy()
            axes[row, col_hr].imshow(hr_img)
            axes[row, col_hr].set_title(f'HR (GT)\n{hr_img.shape[0]}×{hr_img.shape[1]}')
            axes[row, col_hr].axis('off')
    
    plt.suptitle('Super-Resolution Results: LR → SR → HR Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / 'sr_results_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ SR results visualization saved to: {save_path}")
    
    # Log to W&B
    if use_wandb:
        try:
            import wandb
            wandb.log({'sr_results': wandb.Image(str(save_path))})
            
            # Also log individual SR images
            for method_name, sr_batch in sr_results.items():
                for i in range(min(4, n_samples)):
                    wandb.log({
                        f'{method_name}/sample_{i}/lr': wandb.Image(lr_batch[i]),
                        f'{method_name}/sample_{i}/sr': wandb.Image(sr_batch[i]),
                        f'{method_name}/sample_{i}/hr': wandb.Image(hr_batch[i]),
                    })
            
            print("✅ SR results logged to W&B")
        except Exception as e:
            print(f"⚠️ Could not log to W&B: {e}")
    
    return str(save_path)


