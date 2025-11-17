#!/bin/bash

# Run ESPCN QAT experiments
# Trains 7 models with different quantization methods

echo "=========================================="
echo "  ESPCN QAT Experiments"
echo "=========================================="
echo ""
echo "This will train 7 ESPCN models:"
echo "  1. Base (no quantization)"
echo "  2. LSQ"
echo "  3. PACT"
echo "  4. AdaRound"
echo "  5. APoT"
echo "  6. DoReFa"
echo "  7. STE"
echo ""
echo "Dataset: Food101 (×4 upscaling, 200 train + 50 val)"
echo "Metrics: PSNR (dB) + SSIM (0-1)"
echo "Loss: L1 (standard for SR)"
echo "Estimated time: ~1.5 hours on CPU, ~20 min on GPU"
echo ""
read -p "Press Enter to start or Ctrl+C to cancel..."

echo ""
echo "Starting training..."
python -m espcn.training

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Training failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "  ✅ Training Complete!"
echo "=========================================="
echo ""
echo "Analyze results:"
echo "  python analyze_espcn_results.py"
echo ""
echo "Results saved to: espcn/checkpoints/"
echo ""

