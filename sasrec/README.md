
# SASRec - Sequential Recommendation with QAT

This directory contains the implementation of **SASRec** (Self-Attentive Sequential Recommendation) model with various **Quantization-Aware Training (QAT)** methods for INT8 quantization.

## Model Architecture

**SASRec** uses a self-attention mechanism (Transformer encoder) to model user behavior sequences for next-item prediction in recommendation systems.

Key components:
- **Item Embedding**: Learns representations for items
- **Positional Encoding**: Encodes sequential position information
- **Multi-head Self-Attention Blocks**: Captures item-item relationships
- **Feed-Forward Networks**: Non-linear transformations
- **Prediction Head**: Scores all items for next-item prediction

## Evaluation Metric

**NDCG@10** (Normalized Discounted Cumulative Gain at 10)
- Primary metric for ranking quality in recommendation systems
- Measures how well the model ranks relevant items in top-10 predictions
- Range: [0, 1], higher is better

Additional metrics:
- NDCG@5, NDCG@20
- Hit@10 (recall at 10)

## Quantization Methods

All methods use **fake quantization** (INT8 simulation) for training:

1. **LSQ** (Learned Step Size Quantization)
   - Learnable quantization step sizes
   - Per-tensor for activations, per-channel for weights
   - Gradient scaling for stable training

2. **PACT** (Parameterized Clipping Activation)
   - Learnable activation clipping thresholds
   - Uniform quantization in learned range
   - L2 regularization on clipping parameters

3. **AdaRound** (Adaptive Rounding)
   - Learnable rounding for weight quantization
   - Soft-to-hard rounding during training
   - Regularization to push towards binary decisions

4. **APoT** (Additive Powers-of-Two)
   - Non-uniform quantization using powers of 2
   - Reduces to bit-shift operations in hardware
   - RCF (Range Constraint Fine-tuning) for dynamic ranges

5. **DoReFa** (DoReFa-Net)
   - Quantizes both activations and weights
   - Tanh normalization for signed activations
   - k-bit uniform quantization in [0,1] or [-1,1]

6. **Fake STE** (Straight-Through Estimator)
   - Simple baseline with learnable scales
   - STE for gradient flow through rounding
   - Per-tensor quantization

## Directory Structure

```
sasrec/
├── model/
│   ├── Base.py              # Base SASRec model with quantization hooks
│   ├── SasrecLSQ.py         # LSQ quantization
│   ├── SasrecPact.py        # PACT quantization
│   ├── SasrecAdaRound.py    # AdaRound quantization
│   ├── SasrecApot.py        # APoT quantization
│   ├── SasrecDorefa.py      # DoReFa quantization
│   └── SasrecSTE.py         # Fake STE quantization
├── sasrec_utils/
│   ├── load_dataset.py      # Dataset loading and preparation
│   └── seeds.py             # Random seed utilities
├── train_utils/
│   ├── cycle.py             # Training loop with NDCG@10 calculation
│   ├── loaders.py           # DataLoader creation
│   └── save_checkpoint.py   # Model checkpointing utilities
├── checkpoints/             # Saved model checkpoints
├── training.py              # Main training script
└── Readme.md               # This file
```

## Usage

### Training All Models

```bash
cd /path/to/EfficientDL_QAT
python -m sasrec.training
```

This will train:
1. Base SASRec (no quantization) - baseline
2. SASRec + LSQ
3. SASRec + PACT
4. SASRec + AdaRound
5. SASRec + APoT
6. SASRec + DoReFa
7. SASRec + Fake STE

### Training Individual Models

```python
from sasrec.model.SasrecLSQ import QuantSASRecLSQ
from sasrec.train_utils.cycle import fit
from sasrec.train_utils.loaders import make_loaders

# Create dataloaders
train_loader, test_loader, num_users, num_items = make_loaders(
    batch_size=32, 
    max_len=50
)

# Create model
model = QuantSASRecLSQ(
    num_items=num_items,
    embed_dim=128,
    num_heads=4,
    num_blocks=2,
    max_len=50,
    dropout=0.2,
    bits=8
)

# Train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = fit(model, train_loader, test_loader, device, epochs=10, lr=1e-3)
```

## Hyperparameters

Default configuration:
- **EPOCHS**: 10
- **BATCH_SIZE**: 32
- **EMBED_DIM**: 128
- **NUM_HEADS**: 4
- **NUM_BLOCKS**: 2
- **MAX_LEN**: 50 (maximum sequence length)
- **DROPOUT**: 0.2
- **LR**: 1e-3
- **BITS**: 8 (INT8 quantization)

## Dataset

Currently uses synthetic sequential recommendation data for demonstration. For real experiments, replace with:
- MovieLens (100K, 1M, 20M)
- Amazon product reviews
- Steam games dataset
- Any other sequential recommendation dataset

## Results

After training, results are saved in:
- `checkpoints/*.pt` - Model state dictionaries
- `checkpoints/*.json` - Training history with metrics per epoch

Metrics tracked:
- Training loss
- Validation loss
- NDCG@10 (primary metric)
- NDCG@5, NDCG@20
- Hit@10

## Quantization Hooks

The `BaseSASRec` model provides hooks for quantization:

```python
def quant_embed_out(self, x):
    """Quantize after embedding layer"""
    
def quant_attn_out(self, x, block_idx: int = 0):
    """Quantize after attention block"""
    
def quant_ffn_out(self, x, block_idx: int = 0):
    """Quantize after feed-forward block"""
    
def quant_head_weight(self, w):
    """Quantize prediction head weights"""
```

Each quantization method overrides these hooks with its specific quantization logic.

## Next Steps

1. **Real Dataset**: Replace synthetic data with actual recommendation dataset
2. **Hyperparameter Tuning**: Experiment with different configurations
3. **INT8 Conversion**: Convert best model to real INT8 using PyTorch quantization
4. **Inference Speed**: Measure CPU inference speed of INT8 vs FP32
5. **Quality Analysis**: Compare NDCG@10 across all methods

## References

- SASRec: [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781)
- LSQ: [Learned Step Size Quantization](https://arxiv.org/abs/1902.08153)
- PACT: [PACT: Parameterized Clipping Activation for Quantized Neural Networks](https://arxiv.org/abs/1805.06085)
- AdaRound: [Up or Down? Adaptive Rounding for Post-Training Quantization](https://arxiv.org/abs/2004.10568)
- APoT: [Additive Powers-of-Two Quantization](https://arxiv.org/abs/1909.13144)
- DoReFa: [DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks](https://arxiv.org/abs/1606.06160)

