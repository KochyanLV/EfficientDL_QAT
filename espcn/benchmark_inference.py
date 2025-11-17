"""
Benchmark inference speed and memory for ESPCN models
"""
import torch
import time
import numpy as np
from pathlib import Path
import json


def get_model_size_mb(model):
    """Calculate model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def remove_wandb_hooks(model):
    """Remove W&B hooks from model to avoid errors during benchmark."""
    # Remove all forward hooks (W&B hooks are forward hooks)
    for module in model.modules():
        module._forward_hooks.clear()
        module._forward_pre_hooks.clear()


def benchmark_inference(model, input_tensor, device, num_runs=100, warmup=10):
    """
    Benchmark inference time and memory.
    
    Args:
        model: ESPCN model
        input_tensor: Input LR images (B, C, H, W)
        device: torch.device
        num_runs: Number of inference runs
        warmup: Warmup runs (not counted)
    
    Returns:
        Dictionary with timing and memory stats
    """
    model.eval()
    model.to(device)
    input_tensor = input_tensor.to(device)
    
    # Remove W&B hooks to avoid logging errors
    remove_wandb_hooks(model)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    
    # Synchronize GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark timing
    times = []
    for _ in range(num_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            _ = model(input_tensor)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    times = np.array(times)
    
    # Memory measurement
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(input_tensor)
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    else:
        peak_memory_mb = 0.0  # CPU memory harder to measure
    
    # Model size
    model_size_mb = get_model_size_mb(model)
    num_params = count_parameters(model)
    
    stats = {
        'mean_time_ms': float(times.mean() * 1000),
        'std_time_ms': float(times.std() * 1000),
        'min_time_ms': float(times.min() * 1000),
        'max_time_ms': float(times.max() * 1000),
        'median_time_ms': float(np.median(times) * 1000),
        'fps': float(input_tensor.size(0) / times.mean()),
        'peak_memory_mb': float(peak_memory_mb),
        'model_size_mb': float(model_size_mb),
        'num_parameters': int(num_params),
        'num_runs': num_runs,
        'batch_size': input_tensor.size(0),
        'input_shape': list(input_tensor.shape),
        'device': str(device),
    }
    
    return stats


def benchmark_all_models(models_dict, test_loader, device, num_runs=100):
    """
    Benchmark all models and save results.
    
    Args:
        models_dict: {'Base': model, 'LSQ': model, ...}
        test_loader: DataLoader for test data
        device: torch.device
        num_runs: Number of inference runs per model
    
    Returns:
        Dictionary with all benchmark results
    """
    # Get a sample batch
    lr_batch, hr_batch = next(iter(test_loader))
    lr_batch = lr_batch.to(device)
    
    print("=" * 70)
    print(" Benchmarking Inference Speed & Memory")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Input shape: {lr_batch.shape}")
    print(f"Number of runs: {num_runs}")
    print()
    
    results = {}
    
    for method_name, model in models_dict.items():
        print(f"Benchmarking {method_name}...")
        stats = benchmark_inference(model, lr_batch, device, num_runs=num_runs)
        results[method_name] = stats
        
        print(f"  Time: {stats['mean_time_ms']:.2f} ± {stats['std_time_ms']:.2f} ms")
        print(f"  FPS: {stats['fps']:.1f}")
        print(f"  Model size: {stats['model_size_mb']:.2f} MB")
        print(f"  Parameters: {stats['num_parameters']:,}")
        if device.type == 'cuda':
            print(f"  Peak memory: {stats['peak_memory_mb']:.2f} MB")
        print()
    
    # Save results
    output_dir = Path('espcn/benchmarks')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / 'inference_benchmark.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Benchmark results saved to: {results_path}")
    
    return results


def plot_benchmark_results(results, output_dir='espcn/benchmarks'):
    """
    Plot benchmark comparison charts.
    
    Args:
        results: Dictionary from benchmark_all_models
        output_dir: Where to save plots
    """
    import matplotlib.pyplot as plt
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    methods = list(results.keys())
    
    # Extract metrics
    times = [results[m]['mean_time_ms'] for m in methods]
    memory = [results[m]['model_size_mb'] for m in methods]
    params = [results[m]['num_parameters'] / 1e6 for m in methods]  # In millions
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Inference time
    axes[0].bar(methods, times, color='skyblue', edgecolor='black')
    axes[0].set_title('Inference Time (ms)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Time (ms)')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for i, v in enumerate(times):
        axes[0].text(i, v + max(times)*0.02, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Model size
    axes[1].bar(methods, memory, color='lightcoral', edgecolor='black')
    axes[1].set_title('Model Size (MB)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Size (MB)')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].tick_params(axis='x', rotation=45)
    
    for i, v in enumerate(memory):
        axes[1].text(i, v + max(memory)*0.02, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Number of parameters
    axes[2].bar(methods, params, color='lightgreen', edgecolor='black')
    axes[2].set_title('Parameters (Millions)', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Params (M)')
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].tick_params(axis='x', rotation=45)
    
    for i, v in enumerate(params):
        axes[2].text(i, v + max(params)*0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('ESPCN QAT Methods - Inference Benchmark', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / 'inference_benchmark.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Benchmark plot saved to: {save_path}")
    
    return str(save_path)


def plot_performance_metrics(results, metrics_dict, output_dir='espcn/benchmarks'):
    """
    Plot PSNR vs Time and PSNR vs Model Size.
    
    Args:
        results: Benchmark results
        metrics_dict: {'Base': {'psnr': 25.3, 'ssim': 0.68}, ...}
        output_dir: Where to save plots
    """
    import matplotlib.pyplot as plt
    
    output_dir = Path(output_dir)
    
    methods = list(results.keys())
    times = [results[m]['mean_time_ms'] for m in methods]
    sizes = [results[m]['model_size_mb'] for m in methods]
    psnrs = [metrics_dict[m]['psnr'] for m in methods if m in metrics_dict]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. PSNR vs Inference Time
    colors = ['red' if m == 'Base' else 'blue' for m in methods[:len(psnrs)]]
    axes[0].scatter(times[:len(psnrs)], psnrs, s=200, c=colors, alpha=0.6, edgecolors='black', linewidths=2)
    
    for i, method in enumerate(methods[:len(psnrs)]):
        axes[0].annotate(method, (times[i], psnrs[i]), 
                        fontsize=10, ha='center', va='bottom', fontweight='bold')
    
    axes[0].set_xlabel('Inference Time (ms)', fontsize=12)
    axes[0].set_ylabel('PSNR (dB)', fontsize=12)
    axes[0].set_title('Quality vs Speed', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # 2. PSNR vs Model Size
    axes[1].scatter(sizes[:len(psnrs)], psnrs, s=200, c=colors, alpha=0.6, edgecolors='black', linewidths=2)
    
    for i, method in enumerate(methods[:len(psnrs)]):
        axes[1].annotate(method, (sizes[i], psnrs[i]), 
                        fontsize=10, ha='center', va='bottom', fontweight='bold')
    
    axes[1].set_xlabel('Model Size (MB)', fontsize=12)
    axes[1].set_ylabel('PSNR (dB)', fontsize=12)
    axes[1].set_title('Quality vs Size', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('ESPCN QAT - Performance Trade-offs', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / 'performance_tradeoffs.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Performance plot saved to: {save_path}")
    
    return str(save_path)

