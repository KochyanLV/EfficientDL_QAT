# ESPCN with Quantization Aware Training

This module implements ESPCN (Efficient Sub-Pixel CNN) for super-resolution with various Quantization Aware Training (QAT) methods.

## Model Architecture

ESPCN uses convolutional layers for feature extraction followed by sub-pixel convolution (PixelShuffle) for upsampling.

**Architecture:**
- Conv1: 3 → 64 channels, 5x5 kernel
- Conv2: 64 → 32 channels, 3x3 kernel  
- Conv3: 32 → 3×(scale²) channels, 3x3 kernel
- PixelShuffle: Rearrange to HR image

## Quantization Methods

All methods use **INT8** quantization:

1. **LSQ** (Learned Step Size Quantization)
   - Learnable quantization scale
   - Per-tensor for activations, per-channel for weights

2. **PACT** (Parameterized Clipping Activation)
   - Learnable clipping parameter
   - Uniform weight quantization

3. **AdaRound** (Adaptive Rounding)
   - Learnable rounding for weights
   - Weights-only quantization

4. **APoT** (Additive Powers-of-Two)
   - Quantization to powers of 2
   - Hardware-efficient

5. **DoReFa**
   - Deterministic quantization
   - Separate methods for weights and activations

6. **STE** (Straight-Through Estimator)
   - Simple fake quantization baseline
   - Direct gradient pass-through

## Dataset

Uses **Food101** - высококачественные изображения для super-resolution!

**Food101 (×4 upscaling):**
- ✅ 200-400 training images (default 200 for fast experiments)
- ✅ 50-100 validation images (default 50)
- ✅ High resolution (~512×384 average)
- ✅ Professional photography
- ✅ No authentication required!
- ✅ Bicubic downsampling applied during training

**Dataset:** `food101` from HuggingFace  
**Adjust samples:** Change `NUM_TRAIN` and `NUM_VAL` in `training.py`

## Metric

**PSNR (Peak Signal-to-Noise Ratio)** - measures reconstruction quality in dB. Higher is better.

## Usage

```bash
cd /Users/alina_burykina/study/itmo/EfficientDL_QAT
python -m espcn.training
```

## Structure

```
espcn/
├── model/
│   ├── Base.py              # Base ESPCN model
│   ├── EspcnLSQ.py          # LSQ quantization
│   ├── EspcnPact.py         # PACT quantization
│   ├── EspcnAdaRound.py     # AdaRound quantization
│   ├── EspcnApot.py         # APoT quantization
│   ├── EspcnDorefa.py       # DoReFa quantization
│   └── EspcnSTE.py          # STE quantization
├── espcn_utils/
│   ├── load_dataset.py      # Dataset loading
│   └── seeds.py             # Random seed utilities
├── train_utils/
│   ├── cycle.py             # Training loop with PSNR
│   ├── loaders.py           # DataLoader creation
│   └── save_checkpoint.py   # Checkpoint saving
├── training.py              # Main training script
└── README.md                # This file
```

## Checkpoints

Model checkpoints and metrics are saved to `espcn/checkpoints/` after training.

