"""
QAT Debugger - Visualize quantization effects in QAT layers during inference
Adapted for ESPCN Conv2d layers
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional
import os


class QATDebugger:
    """
    Captures and visualizes QAT layer activations, weights, and quantization effects
    Supports both Conv2d and Linear layers
    """
    
    def __init__(self, output_dir: str = "./qat_debug", method_name: str = ""):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.captured_data = {}
        self.method_name = method_name
    
    def attach_to_layer(self, layer, layer_name: str):
        """Attach debugger to a QAT layer (Conv2d or Linear with quantizers)"""
        
        # For ESPCN, Conv2d layers don't have explicit quantizers
        # Quantization happens through model methods (quant_conv1_out, etc.)
        # So we'll capture raw Conv2d operations and analyze them
        if not hasattr(layer, 'weight'):
            return self
        
        def capture_hook(module, input, output):
            print(f"🔍 Hook triggered for {layer_name}!")
            try:
                with torch.no_grad():
                    # Get input activation
                    x_raw = input[0]
                    print(f"  - Input shape: {x_raw.shape}")
                    
                    # Get weight
                    w_raw = module.weight
                    
                    # Try to get quantizers - different methods have different structures
                    act_q = None
                    w_q = None
                    
                    # Try common patterns
                    if hasattr(module, 'activation_quantizer'):
                        act_q = module.activation_quantizer
                    if hasattr(module, 'weight_quantizer'):
                        w_q = module.weight_quantizer
                    
                    # Initialize storage (even if no quantizers - show FP32)
                    data = {
                        'x_raw': x_raw.detach().cpu(),
                        'w_raw': w_raw.detach().cpu(),
                        'y_final': output.detach().cpu(),
                    }
                
                # 1. Quantize weight (if quantizer present)
                try:
                    if w_q is not None and hasattr(w_q, 'scale'):
                        s_w = w_q.scale
                        data['s_w'] = s_w.detach().cpu()
                        
                        # Get quantization parameters
                        qmin_w = getattr(w_q, 'qmin', -128)
                        qmax_w = getattr(w_q, 'qmax', 127)
                        num_bits_w = getattr(w_q, 'num_bits', 8)
                        
                        # Quantize weights
                        if hasattr(w_q, 'per_channel') and w_q.per_channel:
                            # Per-channel quantization
                            s_w_expanded = s_w.view(-1, 1, 1, 1) if w_raw.dim() == 4 else s_w.view(-1, 1)
                            code_w = torch.round(w_raw / s_w_expanded)
                        else:
                            code_w = torch.round(w_raw / s_w)
                        
                        code_w = torch.clamp(code_w, qmin_w, qmax_w)
                        
                        if hasattr(w_q, 'per_channel') and w_q.per_channel:
                            w_quant = code_w * s_w_expanded
                        else:
                            w_quant = code_w * s_w
                        
                        data['code_w'] = code_w.detach().cpu()
                        data['w_quant'] = w_quant.detach().cpu()
                        data['qmin_w'] = qmin_w
                        data['qmax_w'] = qmax_w
                        data['num_bits_w'] = num_bits_w
                    else:
                        # No quantizer - FP32 layer
                        data['w_quant'] = w_raw.detach().cpu()
                        data['code_w'] = None
                        data['qmin_w'] = None
                        data['qmax_w'] = None
                        data['num_bits_w'] = 32  # FP32
                except Exception as e:
                    print(f"Warning: Could not quantize weights for {layer_name}: {e}")
                    data['w_quant'] = w_raw.detach().cpu()
                    data['code_w'] = None
                    data['num_bits_w'] = 32
                
                # 2. Process activations (for weight-only, activations stay FP32)
                if act_q is not None and hasattr(act_q, 'scale'):
                    try:
                        s_x = act_q.scale
                        data['s_x'] = s_x.detach().cpu()
                        
                        qmin_x = getattr(act_q, 'qmin', -128)
                        qmax_x = getattr(act_q, 'qmax', 127)
                        num_bits_x = getattr(act_q, 'num_bits', 8)
                        
                        # Quantize activation
                        if hasattr(act_q, 'symmetric') and act_q.symmetric:
                            code_x = torch.round(x_raw / s_x)
                        else:
                            zp_x = getattr(act_q, 'zero_point', 0)
                            code_x = torch.round(x_raw / s_x + zp_x)
                        
                        code_x = torch.clamp(code_x, qmin_x, qmax_x)
                        
                        if hasattr(act_q, 'symmetric') and act_q.symmetric:
                            x_quant = code_x * s_x
                        else:
                            zp_x = getattr(act_q, 'zero_point', 0)
                            x_quant = (code_x - zp_x) * s_x
                        
                        data['code_x'] = code_x.detach().cpu()
                        data['x_quant'] = x_quant.detach().cpu()
                        data['qmin_x'] = qmin_x
                        data['qmax_x'] = qmax_x
                        data['num_bits_x'] = num_bits_x
                    except Exception as e:
                        print(f"Warning: Could not quantize activations for {layer_name}: {e}")
                        data['x_quant'] = x_raw.detach().cpu()
                else:
                    # Weight-only quantization - activations are FP32
                    data['x_quant'] = x_raw.detach().cpu()
                    data['code_x'] = None
                    data['s_x'] = None
                    data['qmin_x'] = None
                    data['qmax_x'] = None
                    data['num_bits_x'] = 32  # FP32
                
                # 3. Compute FP and quantized outputs
                try:
                    # Full precision output
                    if isinstance(module, torch.nn.Conv2d):
                        y_fp = F.conv2d(x_raw, w_raw, module.bias, 
                                       module.stride, module.padding, 
                                       module.dilation, module.groups)
                    else:
                        y_fp = F.linear(x_raw, w_raw, module.bias)
                    
                    data['y_fp'] = y_fp.detach().cpu()
                    
                    # Quantized output
                    x_for_quant = data['x_quant'].to(x_raw.device)
                    w_for_quant = data['w_quant'].to(w_raw.device)
                    
                    if isinstance(module, torch.nn.Conv2d):
                        y_quant = F.conv2d(x_for_quant, w_for_quant, module.bias,
                                          module.stride, module.padding,
                                          module.dilation, module.groups)
                    else:
                        y_quant = F.linear(x_for_quant, w_for_quant, module.bias)
                    
                    data['y_quant'] = y_quant.detach().cpu()
                except Exception as e:
                    print(f"Warning: Could not compute outputs for {layer_name}: {e}")
                    data['y_fp'] = output.detach().cpu()
                    data['y_quant'] = output.detach().cpu()
                
                # Store everything (always!)
                self.captured_data[layer_name] = data
                print(f"Debug: Captured data for {layer_name}, keys: {list(data.keys())}")
            except Exception as e:
                print(f"ERROR in capture_hook for {layer_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Register hook
        layer.register_forward_hook(capture_hook)
        return self
    
    def plot_layer(self, layer_name: str, sample_idx: int = 0):
        """Plot quantization effects for a specific layer"""
        
        if layer_name not in self.captured_data:
            print(f"No data for layer {layer_name}")
            return None
        
        data = self.captured_data[layer_name]
        
        # Create figure
        fig = plt.figure(figsize=(20, 12))
        
        # Extract one sample for visualization
        def get_sample(tensor, idx=sample_idx, max_samples=5000):
            if tensor is None:
                return np.array([])
            if tensor.dim() > 2:
                return tensor[idx].flatten()[:max_samples].numpy()
            elif tensor.dim() == 2:
                if idx < tensor.shape[0]:
                    return tensor[idx].flatten()[:max_samples].numpy()
                else:
                    return tensor[0].flatten()[:max_samples].numpy()
            else:
                return tensor.flatten()[:max_samples].numpy()
        
        # 1. Raw activation X
        ax1 = plt.subplot(3, 4, 1)
        x_raw = get_sample(data['x_raw'])
        if len(x_raw) > 0:
            ax1.hist(x_raw, bins=50, alpha=0.7, edgecolor='black', color='blue')
            ax1.set_title(f'1. Raw Activation X\nMean: {x_raw.mean():.4f}, Std: {x_raw.std():.4f}')
        else:
            ax1.set_title('1. Raw Activation X\n(No data)')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Count')
        ax1.grid(True, alpha=0.3)
        
        # 2. Quantized activation codes
        ax2 = plt.subplot(3, 4, 2)
        code_x = get_sample(data.get('code_x'))
        if len(code_x) > 0 and data.get('num_bits_x') != 32:
            ax2.hist(code_x, bins=min(50, 2**data['num_bits_x']), alpha=0.7, color='orange', edgecolor='black')
            ax2.set_title(f'2. Activation Codes ({data["num_bits_x"]}-bit)\nRange: [{data["qmin_x"]}, {data["qmax_x"]}]')
            if data.get('qmin_x') is not None:
                ax2.axvline(data['qmin_x'], color='r', linestyle='--', linewidth=2, label='qmin')
                ax2.axvline(data['qmax_x'], color='r', linestyle='--', linewidth=2, label='qmax')
            ax2.legend()
        else:
            ax2.set_title('2. Activation Codes\n(FP32 - No quantization)')
        ax2.set_xlabel('Code Value')
        ax2.grid(True, alpha=0.3)
        
        # 3. Dequantized activation
        ax3 = plt.subplot(3, 4, 3)
        x_quant = get_sample(data.get('x_quant'))
        if len(x_quant) > 0:
            ax3.hist(x_quant, bins=50, alpha=0.7, color='green', edgecolor='black')
            ax3.set_title(f'3. Dequantized Activation\nMean: {x_quant.mean():.4f}')
        else:
            ax3.set_title('3. Dequantized Activation\n(No data)')
        ax3.set_xlabel('Value')
        ax3.grid(True, alpha=0.3)
        
        # 4. Activation quantization error
        ax4 = plt.subplot(3, 4, 4)
        if len(x_raw) > 0 and len(x_quant) > 0:
            x_error = x_raw[:len(x_quant)] - x_quant[:len(x_raw)]
            ax4.hist(x_error, bins=50, alpha=0.7, color='red', edgecolor='black')
            ax4.set_title(f'4. Activation Quant Error\nMAE: {np.abs(x_error).mean():.6f}')
        else:
            ax4.set_title('4. Activation Quant Error\n(No quantization)')
        ax4.set_xlabel('Error')
        ax4.grid(True, alpha=0.3)
        
        # 5. Raw weights W
        ax5 = plt.subplot(3, 4, 5)
        w_raw = data['w_raw'].flatten()[:5000].numpy()
        ax5.hist(w_raw, bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax5.set_title(f'5. Raw Weights W\nMean: {w_raw.mean():.4f}, Std: {w_raw.std():.4f}')
        ax5.set_xlabel('Value')
        ax5.grid(True, alpha=0.3)
        
        # 6. Quantized weight codes
        ax6 = plt.subplot(3, 4, 6)
        code_w = get_sample(data.get('code_w'), max_samples=5000)
        num_bits_w = data.get('num_bits_w', 32)
        if len(code_w) > 0 and num_bits_w != 32:
            ax6.hist(code_w, bins=min(50, 2**num_bits_w), alpha=0.7, color='brown', edgecolor='black')
            ax6.set_title(f'6. Weight Codes ({num_bits_w}-bit)\nRange: [{data["qmin_w"]}, {data["qmax_w"]}]')
            if data.get('qmin_w') is not None:
                ax6.axvline(data['qmin_w'], color='r', linestyle='--', linewidth=2, label='qmin')
                ax6.axvline(data['qmax_w'], color='r', linestyle='--', linewidth=2, label='qmax')
            ax6.legend()
        else:
            ax6.set_title('6. Weight Codes\n(FP32 - No quantization)')
        ax6.set_xlabel('Code Value')
        ax6.grid(True, alpha=0.3)
        
        # 7. Dequantized weights
        ax7 = plt.subplot(3, 4, 7)
        w_quant = data.get('w_quant', data['w_raw']).flatten()[:5000].numpy()
        ax7.hist(w_quant, bins=50, alpha=0.7, color='cyan', edgecolor='black')
        ax7.set_title(f'7. Dequantized Weights\nMean: {w_quant.mean():.4f}')
        ax7.set_xlabel('Value')
        ax7.grid(True, alpha=0.3)
        
        # 8. Weight quantization error
        ax8 = plt.subplot(3, 4, 8)
        w_error = w_raw[:len(w_quant)] - w_quant[:len(w_raw)]
        ax8.hist(w_error, bins=50, alpha=0.7, color='red', edgecolor='black')
        ax8.set_title(f'8. Weight Quant Error\nMAE: {np.abs(w_error).mean():.6f}')
        ax8.set_xlabel('Error')
        ax8.grid(True, alpha=0.3)
        
        # 9. Full precision output
        ax9 = plt.subplot(3, 4, 9)
        y_fp = get_sample(data.get('y_fp'))
        if len(y_fp) > 0:
            ax9.hist(y_fp, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax9.set_title(f'9. FP Output\nMean: {y_fp.mean():.4f}, Std: {y_fp.std():.4f}')
        else:
            ax9.set_title('9. FP Output\n(No data)')
        ax9.set_xlabel('Value')
        ax9.grid(True, alpha=0.3)
        
        # 10. Quantized output
        ax10 = plt.subplot(3, 4, 10)
        y_quant = get_sample(data.get('y_quant'))
        if len(y_quant) > 0:
            ax10.hist(y_quant, bins=50, alpha=0.7, color='orange', edgecolor='black')
            ax10.set_title(f'10. Quantized Output\nMean: {y_quant.mean():.4f}')
        else:
            ax10.set_title('10. Quantized Output\n(No data)')
        ax10.set_xlabel('Value')
        ax10.grid(True, alpha=0.3)
        
        # 11. FP vs Quantized scatter
        ax11 = plt.subplot(3, 4, 11)
        if len(y_fp) > 0 and len(y_quant) > 0:
            min_len = min(500, len(y_fp), len(y_quant))
            ax11.scatter(y_fp[:min_len], y_quant[:min_len], alpha=0.5, s=2)
            ax11.plot([y_fp.min(), y_fp.max()], [y_fp.min(), y_fp.max()], 'r--', linewidth=2, label='y=x')
            ax11.set_title('11. FP vs Quantized')
            ax11.legend()
        else:
            ax11.set_title('11. FP vs Quantized\n(No data)')
        ax11.set_xlabel('Full Precision')
        ax11.set_ylabel('Quantized')
        ax11.grid(True, alpha=0.3)
        
        # 12. Output quantization error
        ax12 = plt.subplot(3, 4, 12)
        if len(y_fp) > 0 and len(y_quant) > 0:
            min_len = min(len(y_fp), len(y_quant))
            y_error = y_fp[:min_len] - y_quant[:min_len]
            ax12.hist(y_error, bins=50, alpha=0.7, color='red', edgecolor='black')
            mse = (y_error**2).mean()
            ax12.set_title(f'12. Output Error\nMSE: {mse:.6f}')
        else:
            ax12.set_title('12. Output Error\n(No data)')
        ax12.set_xlabel('Error (FP - Quant)')
        ax12.grid(True, alpha=0.3)
        
        plt.suptitle(f'QAT Layer: {layer_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Generate unique filename with method name
        if self.method_name:
            filename = f'{self.method_name}_{layer_name.replace(".", "_")}_qat.png'
        else:
            filename = f'{layer_name.replace(".", "_")}_qat.png'
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved QAT visualization for {layer_name}: {save_path}")
        
        return save_path
    
    def set_method_name(self, method_name):
        """Set method name for unique file naming"""
        self.method_name = method_name


def remove_wandb_hooks(model):
    """Remove W&B hooks from model to avoid errors."""
    for module in model.modules():
        # Clear forward hooks (W&B hooks are forward hooks)
        module._forward_hooks.clear()
        module._forward_pre_hooks.clear()


def debug_espcn_model(model, lr_image, method_name='model', output_dir="./qat_debug"):
    """
    Debug an ESPCN QAT model by capturing quantization pipeline
    
    Args:
        model: The ESPCN QAT model
        lr_image: Low-resolution input image (1, C, H, W)
        method_name: Name of the QAT method (for file naming)
        output_dir: Where to save visualizations
    
    Returns:
        List of saved plot paths
    """
    debugger = QATDebugger(output_dir=output_dir, method_name=method_name)
    
    # Remove W&B hooks to avoid logging errors
    remove_wandb_hooks(model)
    
    # Find Conv2d layers by name (ESPCN has conv1, conv2, conv3)
    # We'll monitor conv1 (first convolutional layer)
    target_layer = None
    target_name = None
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and 'conv1' in name:
            target_layer = module
            target_name = name
            break
    
    # Fallback: just get first Conv2d layer
    if target_layer is None:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                target_name = name
                break
    
    if target_layer is None:
        print(f"Warning: No Conv2d layers found in {method_name}")
        return []
    
    print(f"Found Conv2d layer '{target_name}' in {method_name}")
    
    # Attach debugger to layer
    debugger.attach_to_layer(target_layer, target_name)
    print(f"  ✓ Attached to {target_name}")
    
    # IMPORTANT: ESPCN uses F.conv2d directly AND applies quantization through model methods
    # We need to manually simulate the quantization pipeline
    model.eval()
    with torch.no_grad():
        print(f"  🔧 Simulating quantized forward pass...")
        
        # Get quantized weights (if model has quantization method)
        if hasattr(model, 'quant_conv1_weight'):
            print(f"  ✓ Model has quant_conv1_weight - applying quantization")
            w_raw = target_layer.weight
            
            # Try to apply quantization (may fail for AdaRound with uninitialized params)
            try:
                w_quant = model.quant_conv1_weight(w_raw)
            except Exception as e:
                print(f"  ⚠️  Quantization failed: {e}")
                print(f"  ℹ️  Using raw weights instead (no quantization)")
                w_quant = w_raw
            
            # Store both raw and quantized weights for comparison
            import torch.nn.functional as F
            
            # Manually create data to capture quantization effect
            x_raw = lr_image
            
            # Forward with raw weights (FP32)
            y_fp = F.conv2d(x_raw, w_raw, target_layer.bias, padding=target_layer.padding)
            
            # Forward with quantized weights
            y_quant_noact = F.conv2d(x_raw, w_quant, target_layer.bias, padding=target_layer.padding)
            
            # Apply activation quantization (if model has it)
            if hasattr(model, 'quant_conv1_out'):
                print(f"  ✓ Applying activation quantization (quant_conv1_out)")
                y_quant = model.quant_conv1_out(y_quant_noact)
                
                # Also try to quantize input activation if model quantizes it
                x_quant = x_raw  # By default same as raw
            else:
                y_quant = y_quant_noact
                x_quant = x_raw
            
            # Try to extract activation quantization codes
            code_x = None
            num_bits_x = 32
            qmin_x = None
            qmax_x = None
            s_x = None
            
            # Try to find activation quantizer
            act_quantizer = None
            act_quantizer_attrs = [
                'lsq_act1',    # LSQ
                'pact_act1',   # PACT
                'apot_act1',   # APoT
                'aq1',         # DoReFa
                'ste_quant',   # STE (shared)
            ]
            
            print(f"  🔍 Looking for activation quantizer...")
            for attr_name in act_quantizer_attrs:
                if hasattr(model, attr_name):
                    act_quantizer = getattr(model, attr_name)
                    print(f"  ✓ Found activation quantizer: {attr_name}")
                    break
            
            if act_quantizer is not None:
                act_quantizer_type = type(act_quantizer).__name__
                is_dorefa_act = 'dorefa' in act_quantizer_type.lower()
                is_apot_act = 'apot' in act_quantizer_type.lower()
                
                # Extract activation quantization parameters
                if hasattr(act_quantizer, 'bits'):
                    num_bits_x = act_quantizer.bits
                elif hasattr(act_quantizer, 'bits_a'):
                    num_bits_x = act_quantizer.bits_a
                elif hasattr(model, 'bits'):
                    num_bits_x = model.bits
                
                if num_bits_x != 32:
                    # PACT and DoReFa use unsigned [0, 255], LSQ/APoT/STE use signed [-128, 127]
                    is_pact_act = 'pact' in act_quantizer_type.lower()
                    
                    if is_dorefa_act or is_pact_act:
                        qmin_x = 0
                        qmax_x = 2**num_bits_x - 1
                        method_name = "PACT" if is_pact_act else "DoReFa"
                        print(f"  ✓ Activation bits = {num_bits_x} ({method_name} unsigned), range=[{qmin_x}, {qmax_x}]")
                    elif is_apot_act:
                        qmin_x = -(2**(num_bits_x-1))
                        qmax_x = 2**(num_bits_x-1) - 1
                        print(f"  ✓ Activation bits = {num_bits_x} (APoT special), range=[{qmin_x}, {qmax_x}]")
                        print(f"  ℹ️  APoT activations use powers-of-two encoding (like weights)")
                    else:
                        qmin_x = -(2**(num_bits_x-1))
                        qmax_x = 2**(num_bits_x-1) - 1
                        print(f"  ✓ Activation bits = {num_bits_x}, range=[{qmin_x}, {qmax_x}]")
                    
                    # Compute activation codes (different for DoReFa, PACT, and APoT!)
                    if is_apot_act:
                        # APoT activations: special encoding, don't compute standard codes
                        print(f"  ℹ️  Skipping activation code extraction for APoT (special encoding)")
                        code_x = None  # Don't show misleading codes
                    elif is_pact_act:
                        # PACT activation: clamp to [0, alpha], then quantize
                        print(f"  🔧 Computing PACT activation codes...")
                        alpha_val = act_quantizer.alpha.item() if hasattr(act_quantizer, 'alpha') else 6.0
                        Q = act_quantizer.Q if hasattr(act_quantizer, 'Q') else 255
                        
                        # Clip to [0, alpha]
                        y_clipped = torch.clamp(y_quant_noact, min=0, max=alpha_val)
                        # Scale = alpha / Q
                        s_act = alpha_val / Q
                        # Quantize
                        code_x = torch.round(y_clipped / s_act)
                        code_x = torch.clamp(code_x, qmin_x, qmax_x)
                        code_x = code_x.detach().cpu()
                        print(f"  ✅ Extracted PACT activation codes (unsigned, range [0, {qmax_x}])")
                        print(f"     Alpha={alpha_val:.4f}, Scale={s_act:.6f}")
                    elif is_dorefa_act:
                        # DoReFa activation quantization
                        print(f"  🔧 Computing DoReFa activation codes...")
                        act_t = torch.tanh(y_quant_noact)
                        amax_act = act_t.abs().max().clamp(min=1e-8)
                        
                        # Check if signed (from model config)
                        if hasattr(act_quantizer, 'signed') and act_quantizer.signed:
                            # Signed: normalize to [-1, 1], then quantize
                            act_n = act_t / amax_act  # [-1, 1]
                            # Map to [0, 1] for quantization
                            act_01 = act_n / 2 + 0.5
                        else:
                            # Unsigned: normalize to [0, 1]
                            act_01 = act_t / (2 * amax_act) + 0.5
                        
                        n_levels = 2**num_bits_x - 1
                        code_x = torch.round(act_01 * n_levels)
                        code_x = code_x.detach().cpu()
                        print(f"  ✅ Extracted DoReFa activation codes (range [0, {n_levels}])")
                    else:
                        # Try to extract activation scale for other methods
                        if hasattr(act_quantizer, 's') and act_quantizer.s is not None:
                            s_x = act_quantizer.s
                            print(f"  ✓ Found scale 's'")
                        elif hasattr(act_quantizer, 'scale') and act_quantizer.scale is not None:
                            s_x = act_quantizer.scale
                            print(f"  ✓ Found scale 'scale'")
                        elif hasattr(act_quantizer, 'alpha') and act_quantizer.alpha is not None:
                            # PACT activation: scale = alpha / Q
                            if hasattr(act_quantizer, 'Q'):
                                s_x = act_quantizer.alpha.clamp(min=1e-8) / act_quantizer.Q
                                print(f"  ✓ Computed PACT activation scale = alpha/Q = {s_x:.6f}")
                            else:
                                s_x = act_quantizer.alpha
                                print(f"  ✓ Found scale 'alpha'")
                        elif hasattr(model, '_compute_scale'):
                            # STE computes scale dynamically
                            s_x = model._compute_scale(y_quant_noact)
                            print(f"  ✓ Computed activation scale (STE) = {s_x:.6f}")
                        
                        # Compute activation INT8 codes
                        if s_x is not None:
                            if s_x.numel() > 1:
                                code_x = torch.round(y_quant_noact / s_x.view(1, -1, 1, 1))
                            else:
                                code_x = torch.round(y_quant_noact / s_x)
                            code_x = torch.clamp(code_x, qmin_x, qmax_x)
                            code_x = code_x.detach().cpu()
                            print(f"  ✅ Extracted INT{num_bits_x} activation codes!")
            
            # Try to extract weight quantization parameters and codes
            code_w = None
            num_bits_w = 32
            qmin_w = None
            qmax_w = None
            s_w = None
            
            # Try to find weight quantizer for conv1
            quantizer = None
            quantizer_attrs = [
                'lsq_w1',      # LSQ
                'wq1',         # PACT, AdaRound, DoReFa
                'apot_w1',     # APoT
                'ste_quant',   # STE (shared quantizer)
            ]
            
            print(f"  🔍 Looking for quantizer in model attributes...")
            for attr_name in quantizer_attrs:
                if hasattr(model, attr_name):
                    quantizer = getattr(model, attr_name)
                    print(f"  ✓ Found weight quantizer: {attr_name}")
                    break
                else:
                    print(f"     ✗ {attr_name} not found")
            
            if quantizer is None:
                print(f"  ⚠️  No quantizer found! Available attrs: {[a for a in dir(model) if 'quant' in a.lower() or 'lsq' in a.lower() or 'wq' in a.lower() or 'apot' in a.lower()]}")
            
            if quantizer is not None:
                quantizer_type = type(quantizer).__name__
                print(f"  🔍 Quantizer type: {quantizer_type}")
                print(f"  🔍 Quantizer attributes: {[a for a in dir(quantizer) if not a.startswith('_')][:10]}")
                
                # Special handling for APoT (doesn't use simple round(w/scale))
                is_apot = 'apot' in quantizer_type.lower() or 'apot' in str(type(quantizer)).lower()
                is_dorefa = 'dorefa' in quantizer_type.lower()
                
                if is_apot:
                    print(f"  ℹ️  APoT uses special encoding (sum of powers-of-two)")
                    print(f"  ℹ️  Cannot extract standard INT8 codes - skipping Panel 6")
                
                if is_dorefa:
                    print(f"  ℹ️  DoReFa uses tanh normalization (no scale parameter)")
                    print(f"  ℹ️  Will compute codes using DoReFa pipeline")
                
                # Extract quantization parameters
                # Try different attribute names for bits
                if hasattr(quantizer, 'num_bits'):
                    num_bits_w = quantizer.num_bits
                    print(f"  ✓ num_bits = {num_bits_w}")
                elif hasattr(quantizer, 'bits'):
                    num_bits_w = quantizer.bits
                    print(f"  ✓ bits = {num_bits_w}")
                elif hasattr(quantizer, 'bits_w'):
                    num_bits_w = quantizer.bits_w
                    print(f"  ✓ bits_w = {num_bits_w}")
                elif hasattr(model, 'bits'):
                    num_bits_w = model.bits  # STE stores bits at model level
                    print(f"  ✓ num_bits from model = {num_bits_w}")
                else:
                    print(f"  ⚠️  num_bits not found, using default 32")
                
                if num_bits_w != 32:
                    qmin_w = -(2**(num_bits_w-1))
                    qmax_w = 2**(num_bits_w-1) - 1
                    print(f"  ✓ qmin={qmin_w}, qmax={qmax_w}")
                    
                    # Try to extract scale (try all possible attribute names)
                    print(f"  🔍 Looking for scale...")
                    if hasattr(quantizer, 's') and quantizer.s is not None:
                        s_w = quantizer.s
                        print(f"  ✓ Found scale 's': shape={s_w.shape}, mean={s_w.mean():.6f}")
                    elif hasattr(quantizer, 'scale') and quantizer.scale is not None:
                        s_w = quantizer.scale
                        print(f"  ✓ Found scale 'scale': shape={s_w.shape}, mean={s_w.mean():.6f}")
                    elif hasattr(quantizer, 'alpha') and quantizer.alpha is not None:
                        # APoT/AdaRound uses alpha as scale parameter
                        s_w = quantizer.alpha
                        print(f"  ✓ Found scale 'alpha': shape={s_w.shape}, mean={s_w.mean():.6f}")
                    elif hasattr(quantizer, 'Q'):
                        # UniformWeightQuant (PACT) computes scale dynamically like STE
                        Q = quantizer.Q
                        if hasattr(quantizer, 'per_channel') and quantizer.per_channel:
                            # Per-channel: max per output channel
                            ch_axis = getattr(quantizer, 'ch_axis', 0)
                            w_perm = w_raw.transpose(0, ch_axis).contiguous().flatten(1)
                            s_w = (w_perm.abs().max(dim=1).values / max(Q, 1)).clamp(min=1e-8)
                            print(f"  ✓ Computed per-channel scale (PACT): shape={s_w.shape}, mean={s_w.mean():.6f}")
                        else:
                            # Per-tensor
                            s_w = (w_raw.abs().max() / max(Q, 1)).clamp(min=1e-8)
                            print(f"  ✓ Computed per-tensor scale (PACT): {s_w:.6f}")
                    else:
                        # For STE or others: compute scale dynamically (like in the model)
                        if hasattr(model, 'Q'):
                            Q = model.Q
                            s_w = w_raw.abs().max() / Q
                            print(f"  ✓ STE: computed dynamic scale = {s_w:.6f}")
                        elif hasattr(model, '_compute_scale'):
                            # Try to use model's scale computation
                            s_w = model._compute_scale(w_raw)
                            print(f"  ✓ Computed scale using model method = {s_w.mean():.6f}")
                        else:
                            print(f"  ⚠️  Scale not found! Quantizer attrs: {[a for a in dir(quantizer) if 'scale' in a.lower() or 'alpha' in a.lower() or 's' == a]}")
                    
                    # Compute INT8 codes (different for each method!)
                    if is_apot:
                        print(f"  ℹ️  Skipping code extraction for APoT (uses powers-of-two encoding)")
                        code_w = None  # Don't show misleading codes
                        num_bits_w = 32  # Mark as special case
                    elif is_dorefa:
                        # DoReFa specific: compute codes using tanh normalization
                        print(f"  🔧 Computing DoReFa codes (tanh normalization)...")
                        w_t = torch.tanh(w_raw)
                        amax = w_t.abs().max().clamp(min=1e-8)
                        w_n = w_t / (2 * amax) + 0.5  # → [0, 1]
                        
                        # Quantize in [0, 1]
                        n_levels = 2**num_bits_w - 1
                        code_w = torch.round(w_n * n_levels)
                        code_w = code_w.detach().cpu()
                        
                        # DoReFa uses unsigned codes [0, 255] for 8-bit
                        qmin_w = 0
                        qmax_w = n_levels
                        
                        print(f"  ✅ Extracted DoReFa codes (unsigned, range [0, {n_levels}])")
                        print(f"     Codes stats: min={code_w.min():.0f}, max={code_w.max():.0f}, mean={code_w.float().mean():.2f}")
                    elif s_w is not None:
                        print(f"  🔧 Computing INT8 codes...")
                        if s_w.numel() > 1:  # Per-channel
                            print(f"     Per-channel quantization (scale shape: {s_w.shape})")
                            s_w_expanded = s_w.view(-1, 1, 1, 1)
                            code_w = torch.round(w_raw / s_w_expanded)
                        else:  # Per-tensor
                            print(f"     Per-tensor quantization (scale: {s_w.item():.6f})")
                            code_w = torch.round(w_raw / s_w)
                        code_w = torch.clamp(code_w, qmin_w, qmax_w)
                        code_w = code_w.detach().cpu()
                        print(f"  ✅ Extracted INT{num_bits_w} weight codes (range [{qmin_w}, {qmax_w}])")
                        print(f"     Codes stats: min={code_w.min():.0f}, max={code_w.max():.0f}, mean={code_w.float().mean():.2f}")
                    else:
                        print(f"  ❌ Cannot compute codes: s_w is None!")
            
            # Manually populate debugger data
            debugger.captured_data[target_name] = {
                'x_raw': x_raw.detach().cpu(),
                'w_raw': w_raw.detach().cpu(),
                'w_quant': w_quant.detach().cpu(),
                'y_fp': y_fp.detach().cpu(),
                'y_quant': y_quant.detach().cpu(),
                'y_final': y_quant.detach().cpu(),
                'x_quant': x_quant.detach().cpu(),
                'code_x': code_x,
                'code_w': code_w,
                'num_bits_x': num_bits_x,
                'num_bits_w': num_bits_w,
                'qmin_x': qmin_x,
                'qmax_x': qmax_x,
                'qmin_w': qmin_w,
                'qmax_w': qmax_w,
                's_w': s_w.detach().cpu() if s_w is not None and hasattr(s_w, 'detach') else None,
                's_x': s_x.detach().cpu() if s_x is not None and hasattr(s_x, 'detach') else None,
            }
            
            print(f"  ✓ Captured quantization data manually")
        else:
            # No quantization - just call module directly
            print(f"  ℹ️  No quantization method - calling module directly")
            _ = target_layer(lr_image)
    
    # Generate visualization
    saved_paths = []
    plot_path = debugger.plot_layer(target_name, sample_idx=0)
    if plot_path:
        saved_paths.append(plot_path)
    
    return saved_paths

