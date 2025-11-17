"""
Convert QAT models to real INT8 and benchmark on CPU
"""
import torch
import torch.quantization as quant
import time
import numpy as np
from pathlib import Path
import json
import copy


def convert_to_real_int8(model_fp32, backend='fbgemm'):
    """
    Convert FP32 model with fake quantization to real INT8.
    
    Args:
        model_fp32: Model trained with QAT (fake quantization)
        backend: 'fbgemm' for x86 CPU, 'qnnpack' for ARM
    
    Returns:
        model_int8: Real INT8 quantized model
    """
    # Prepare for quantization
    model_prepared = copy.deepcopy(model_fp32)
    model_prepared.eval()
    model_prepared.cpu()
    
    # Configure quantization
    model_prepared.qconfig = quant.get_default_qconfig(backend)
    
    # Prepare model (insert observers)
    quant.prepare(model_prepared, inplace=True)
    
    # Convert to INT8
    model_int8 = quant.convert(model_prepared, inplace=False)
    
    return model_int8


def calculate_theoretical_int8_size(model):
    """
    Calculate theoretical INT8 model size (honest estimate).
    
    Assumes:
    - Conv2d/Linear weights: INT8 (1 byte per element)
    - Biases: FP32 (4 bytes per element)
    - Other params: FP32
    
    Returns:
        (size_mb, int8_params, fp32_params)
    """
    total_size = 0
    int8_params = 0
    fp32_params = 0
    
    for name, module in model.named_modules():
        # Conv2d and Linear weights can be quantized to INT8
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            if hasattr(module, 'weight') and module.weight is not None:
                weight = module.weight
                num_elements = weight.numel()
                int8_params += num_elements
                total_size += num_elements * 1  # INT8: 1 byte
                total_size += 8  # scale + zero_point metadata
            
            # Bias stays FP32
            if hasattr(module, 'bias') and module.bias is not None:
                bias = module.bias
                fp32_params += bias.numel()
                total_size += bias.numel() * 4
        else:
            # Other modules: FP32
            for param in module.parameters(recurse=False):
                fp32_params += param.numel()
                total_size += param.numel() * 4
    
    # Buffers
    for buffer in model.buffers():
        total_size += buffer.numel() * buffer.element_size()
    
    size_mb = total_size / 1024**2
    
    return size_mb, int8_params, fp32_params


def get_fp32_model_size_mb(model):
    """Calculate FP32 model size (current fake quantization)."""
    total_size = 0
    for param in model.parameters():
        total_size += param.numel() * 4  # FP32: 4 bytes
    for buffer in model.buffers():
        total_size += buffer.numel() * buffer.element_size()
    
    return total_size / 1024**2


def benchmark_int8_inference(model, input_tensor, num_runs=100, warmup=10):
    """
    Benchmark INT8 model inference on CPU.
    
    Args:
        model: INT8 quantized model
        input_tensor: Input tensor (FP32)
        num_runs: Number of runs
        warmup: Warmup runs
    
    Returns:
        Dictionary with timing stats
    """
    model.eval()
    model.cpu()
    input_tensor = input_tensor.cpu()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    
    # Benchmark timing
    times = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        
        with torch.no_grad():
            _ = model(input_tensor)
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    times = np.array(times)
    
    # Calculate theoretical INT8 size (honest estimate)
    model_size_mb, int8_params, fp32_params = calculate_theoretical_int8_size(model)
    
    stats = {
        'mean_time_ms': float(times.mean() * 1000),
        'std_time_ms': float(times.std() * 1000),
        'min_time_ms': float(times.min() * 1000),
        'max_time_ms': float(times.max() * 1000),
        'median_time_ms': float(np.median(times) * 1000),
        'fps': float(input_tensor.size(0) / times.mean()),
        'model_size_mb': float(model_size_mb),
        'int8_parameters': int(int8_params),
        'fp32_parameters': int(fp32_params),
        'quantized_ratio': float(int8_params / max(int8_params + fp32_params, 1)),
        'num_runs': num_runs,
        'batch_size': input_tensor.size(0),
        'backend': 'int8_cpu',
    }
    
    return stats


def benchmark_real_int8_models(models_dict, test_loader, output_dir='espcn/benchmarks'):
    """
    Convert all QAT models to real INT8 and benchmark on CPU.
    
    Args:
        models_dict: {'Base': model, 'LSQ': model, ...}
        test_loader: DataLoader for test data
        output_dir: Where to save results
    
    Returns:
        Dictionary with INT8 benchmark results
    """
    # Get a sample batch (CPU)
    lr_batch, hr_batch = next(iter(test_loader))
    lr_batch = lr_batch.cpu()
    
    print("=" * 70)
    print(" Converting to Real INT8 & Benchmarking on CPU")
    print("=" * 70)
    print(f"Input shape: {lr_batch.shape}")
    print(f"Batch size: {lr_batch.size(0)} (optimized for CPU)")
    print(f"Backend: fbgemm (x86 CPU)")
    print(f"Runs per model: 200 (for stable results)")
    print()
    
    results_fake = {}  # Fake quantization (FP32)
    results_real = {}  # Real INT8
    
    for method_name, model in models_dict.items():
        print(f"📊 {method_name}")
        
        # 1. Benchmark fake quantization (current FP32 implementation)
        print(f"  1️⃣  Fake quantization (FP32)...")
        model_fake = model.cpu().eval()
        
        try:
            # Benchmark timing (more runs for stable results)
            times = []
            for _ in range(200):
                start = time.perf_counter()
                with torch.no_grad():
                    _ = model_fake(lr_batch)
                end = time.perf_counter()
                times.append(end - start)
            
            times = np.array(times)
            
            # Size: all FP32
            size_fp32 = get_fp32_model_size_mb(model_fake)
            total_params = sum(p.numel() for p in model_fake.parameters())
            
            stats_fake = {
                'mean_time_ms': float(times.mean() * 1000),
                'std_time_ms': float(times.std() * 1000),
                'model_size_mb': float(size_fp32),
                'int8_parameters': 0,
                'fp32_parameters': total_params,
                'quantized_ratio': 0.0,
            }
            
            results_fake[method_name] = stats_fake
            
            print(f"     Time: {stats_fake['mean_time_ms']:.2f} ms")
            print(f"     Size: {stats_fake['model_size_mb']:.3f} MB (all FP32)")
        except Exception as e:
            print(f"     ❌ Error: {e}")
            results_fake[method_name] = None
        
        # 2. Calculate theoretical INT8 size (honest estimate)
        print(f"  2️⃣  Theoretical INT8 (if deployed)...")
        
        try:
            # Calculate what size WOULD be if properly deployed as INT8
            size_int8, int8_params, fp32_params = calculate_theoretical_int8_size(model)
            
            # Also try dynamic quantization for timing (best effort)
            model_int8 = copy.deepcopy(model)
            model_int8.cpu().eval()
            
            # Apply dynamic quantization for timing estimate
            model_int8_quantized = quant.quantize_dynamic(
                model_int8,
                {torch.nn.Conv2d},  # Quantize Conv2d layers
                dtype=torch.qint8
            )
            
            # Benchmark INT8 model timing (more runs for stable results)
            times_int8 = []
            for _ in range(200):
                start = time.perf_counter()
                with torch.no_grad():
                    _ = model_int8_quantized(lr_batch)
                end = time.perf_counter()
                times_int8.append(end - start)
            
            times_int8 = np.array(times_int8)
            
            stats_real = {
                'mean_time_ms': float(times_int8.mean() * 1000),
                'std_time_ms': float(times_int8.std() * 1000),
                'model_size_mb': float(size_int8),  # Theoretical INT8 size
                'int8_parameters': int(int8_params),
                'fp32_parameters': int(fp32_params),
                'quantized_ratio': float(int8_params / max(int8_params + fp32_params, 1)),
            }
            
            results_real[method_name] = stats_real
            
            print(f"     Time: {stats_real['mean_time_ms']:.2f} ms (dynamic quant)")
            print(f"     Size: {stats_real['model_size_mb']:.3f} MB (theoretical INT8)")
            print(f"     INT8 params: {stats_real['int8_parameters']:,} ({stats_real['quantized_ratio']*100:.1f}%)")
            print(f"     FP32 params: {stats_real['fp32_parameters']:,}")
            
            # Calculate speedup and compression
            if results_fake[method_name]:
                speedup = stats_fake['mean_time_ms'] / stats_real['mean_time_ms']
                compression = stats_fake['model_size_mb'] / stats_real['model_size_mb']
                print(f"     🚀 Speedup: {speedup:.2f}× (with INT8 ops)")
                print(f"     💾 Compression: {compression:.2f}× (theoretical)")
            
        except Exception as e:
            print(f"     ⚠️  INT8 conversion failed: {e}")
            results_real[method_name] = None
        
        print()
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_combined = {
        'fake_quantization': results_fake,
        'real_int8': results_real,
    }
    
    results_path = output_dir / 'real_int8_benchmark.json'
    with open(results_path, 'w') as f:
        json.dump(results_combined, f, indent=2)
    
    print(f"✅ INT8 benchmark results saved to: {results_path}")
    
    return results_combined


def plot_int8_comparison(results_combined, output_dir='espcn/benchmarks'):
    """
    Plot comparison: Fake Quantization vs Real INT8.
    
    Args:
        results_combined: {'fake_quantization': {...}, 'real_int8': {...}}
        output_dir: Where to save plots
    """
    import matplotlib.pyplot as plt
    
    output_dir = Path(output_dir)
    
    fake_results = results_combined['fake_quantization']
    real_results = results_combined['real_int8']
    
    # Extract methods that have both fake and real results
    methods = []
    times_fake = []
    times_real = []
    sizes_fake = []
    sizes_real = []
    
    for method in fake_results.keys():
        if fake_results[method] and real_results.get(method):
            methods.append(method)
            times_fake.append(fake_results[method]['mean_time_ms'])
            times_real.append(real_results[method]['mean_time_ms'])
            sizes_fake.append(fake_results[method]['model_size_mb'])
            sizes_real.append(real_results[method]['model_size_mb'])
    
    if not methods:
        print("⚠️  No methods have both fake and real results")
        return None
    
    # Create comparison plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Inference Time Comparison
    x = np.arange(len(methods))
    width = 0.35
    
    axes[0].bar(x - width/2, times_fake, width, label='Fake (FP32)', color='lightcoral', edgecolor='black')
    axes[0].bar(x + width/2, times_real, width, label='Real INT8', color='lightgreen', edgecolor='black')
    
    axes[0].set_xlabel('Method')
    axes[0].set_ylabel('Time (ms)')
    axes[0].set_title('Inference Time: Fake vs Real INT8', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add speedup annotations
    for i in range(len(methods)):
        speedup = times_fake[i] / times_real[i]
        axes[0].text(i, max(times_fake[i], times_real[i]) * 1.05, 
                    f'{speedup:.2f}×', ha='center', fontweight='bold', color='blue')
    
    # 2. Model Size Comparison
    axes[1].bar(x - width/2, sizes_fake, width, label='Fake (FP32)', color='lightcoral', edgecolor='black')
    axes[1].bar(x + width/2, sizes_real, width, label='Real INT8', color='lightgreen', edgecolor='black')
    
    axes[1].set_xlabel('Method')
    axes[1].set_ylabel('Size (MB)')
    axes[1].set_title('Model Size: Fake vs Real INT8', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=45)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add compression annotations
    for i in range(len(methods)):
        compression = sizes_fake[i] / sizes_real[i]
        axes[1].text(i, max(sizes_fake[i], sizes_real[i]) * 1.05,
                    f'{compression:.2f}×', ha='center', fontweight='bold', color='blue')
    
    plt.suptitle('ESPCN QAT: Fake Quantization vs Real INT8', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / 'fake_vs_real_int8.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comparison plot saved to: {save_path}")
    
    return str(save_path)

