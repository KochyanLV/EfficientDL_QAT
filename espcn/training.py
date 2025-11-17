from espcn.model.Base import BaseESPCN
from espcn.model.EspcnLSQ import QuantESPCNLSQ
from espcn.model.EspcnPact import QuantESPCNPACT
from espcn.model.EspcnAdaRound import QuantESPCNAdaRound
from espcn.model.EspcnApot import QuantESPCNAPoT
from espcn.model.EspcnDorefa import QuantESPCNDoReFa
from espcn.model.EspcnSTE import QuantESPCNSTE

from espcn.train_utils.cycle import fit
from espcn.train_utils.loaders import make_loaders
from espcn.espcn_utils.seeds import set_seed

import torch
import logging
import sys
import os
import json
import glob

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


EPOCHS = 30  # More epochs for better convergence with warmup
BATCH_SIZE = 16
SCALE_FACTOR = 4  # ×4 SR
FEATURE_DIM = 64
PATCH_SIZE = 192  # HR patch size (LR = 48×48 for ×4)
NUM_TRAIN = 200  # Number of training images (200 for fast experiments)
NUM_VAL = 50     # Number of validation images
LR = 1e-3
LOSS_TYPE = 'l1'  # 'l1', 'l2', or 'charbonnier'

# W&B logging
USE_WANDB = True  # Set to True to enable W&B logging
WANDB_PROJECT = 'espcn-qat'  # W&B project name


if __name__ == "__main__":
    set_seed(42)
    
    # Load high-resolution images for SR training (Food101)
    # Food101: No auth required, high-quality images (~512×384)
    train_loader, test_loader = make_loaders(
        batch_size=BATCH_SIZE, 
        scale_factor=SCALE_FACTOR, 
        patch_size=PATCH_SIZE,
        num_train=NUM_TRAIN,
        num_test=NUM_VAL
    )
    
    logger.info(f"Device: {device}")
    logger.info(f"Scale factor: {SCALE_FACTOR}×")
    logger.info(f"Loss function: {LOSS_TYPE.upper()}")
    logger.info(f"W&B logging: {'enabled' if USE_WANDB else 'disabled'}")
    
    # Base model without quantization
    logger.info("=" * 80)
    logger.info("Model: Base ESPCN (no quantization)")
    logger.info("=" * 80)
    model_base = BaseESPCN(
        scale_factor=SCALE_FACTOR,  # ×4 upscaling
        num_channels=3,
        feature_dim=FEATURE_DIM,
    )
    result_base = fit(model_base, train_loader, test_loader, device, epochs=EPOCHS, lr=LR, loss_type=LOSS_TYPE,
                      use_wandb=USE_WANDB, wandb_project=WANDB_PROJECT, wandb_name='Base')
    model_base = result_base['model']
    
    # LSQ quantization
    logger.info("=" * 80)
    logger.info("Model: ESPCN + LSQ")
    logger.info("=" * 80)
    model_lsq = QuantESPCNLSQ(
        scale_factor=SCALE_FACTOR,
        num_channels=3,
        feature_dim=FEATURE_DIM,
        bits=8
    )
    result_lsq = fit(model_lsq, train_loader, test_loader, device, epochs=EPOCHS, lr=LR, loss_type=LOSS_TYPE,
                     use_wandb=USE_WANDB, wandb_project=WANDB_PROJECT, wandb_name='LSQ')
    model_lsq = result_lsq['model']
    
    # PACT quantization
    logger.info("=" * 80)
    logger.info("Model: ESPCN + PACT")
    logger.info("=" * 80)
    model_pact = QuantESPCNPACT(
        scale_factor=SCALE_FACTOR,
        num_channels=3,
        feature_dim=FEATURE_DIM,
        bits_act=8,
        bits_w=8,
        pact_init_alpha=6.0
    )
    result_pact = fit(model_pact, train_loader, test_loader, device, epochs=EPOCHS, lr=LR, loss_type=LOSS_TYPE,
                      use_wandb=USE_WANDB, wandb_project=WANDB_PROJECT, wandb_name='PACT')
    model_pact = result_pact['model']
    
    # AdaRound quantization
    logger.info("=" * 80)
    logger.info("Model: ESPCN + AdaRound")
    logger.info("=" * 80)
    model_ada = QuantESPCNAdaRound(
        scale_factor=SCALE_FACTOR,
        num_channels=3,
        feature_dim=FEATURE_DIM,
        bits_w=8
    )
    result_ada = fit(model_ada, train_loader, test_loader, device, epochs=EPOCHS, lr=LR, loss_type=LOSS_TYPE,
                     use_wandb=USE_WANDB, wandb_project=WANDB_PROJECT, wandb_name='AdaRound')
    model_ada = result_ada['model']
    
    # APoT quantization
    logger.info("=" * 80)
    logger.info("Model: ESPCN + APoT")
    logger.info("=" * 80)
    model_apot = QuantESPCNAPoT(
        scale_factor=SCALE_FACTOR,
        num_channels=3,
        feature_dim=FEATURE_DIM,
        bits=8,
        k=1,
        init_alpha_act=6.0,
        init_alpha_w=2.0,
        use_weight_norm_w=True
    )
    result_apot = fit(model_apot, train_loader, test_loader, device, epochs=EPOCHS, lr=LR, loss_type=LOSS_TYPE,
                      use_wandb=USE_WANDB, wandb_project=WANDB_PROJECT, wandb_name='APoT')
    model_apot = result_apot['model']
    
    # DoReFa quantization
    logger.info("=" * 80)
    logger.info("Model: ESPCN + DoReFa")
    logger.info("=" * 80)
    model_dorefa = QuantESPCNDoReFa(
        scale_factor=SCALE_FACTOR,
        num_channels=3,
        feature_dim=FEATURE_DIM,
        bits_a=8,
        bits_w=8,
        act_signed=True,
        act_preproc="tanh"
    )
    result_dorefa = fit(model_dorefa, train_loader, test_loader, device, epochs=EPOCHS, lr=LR, loss_type=LOSS_TYPE,
                        use_wandb=USE_WANDB, wandb_project=WANDB_PROJECT, wandb_name='DoReFa')
    model_dorefa = result_dorefa['model']
    
    # STE quantization
    logger.info("=" * 80)
    logger.info("Model: ESPCN + STE")
    logger.info("=" * 80)
    model_ste = QuantESPCNSTE(
        scale_factor=SCALE_FACTOR,
        num_channels=3,
        feature_dim=FEATURE_DIM,
        bits=8
    )
    result_ste = fit(model_ste, train_loader, test_loader, device, epochs=EPOCHS, lr=LR, loss_type=LOSS_TYPE,
                     use_wandb=USE_WANDB, wandb_project=WANDB_PROJECT, wandb_name='STE')
    model_ste = result_ste['model']
    
    logger.info("=" * 80)
    logger.info("All models trained successfully!")
    logger.info("=" * 80)
    
    # Benchmark inference speed and memory
    logger.info("=" * 80)
    logger.info("Benchmarking inference speed & memory...")
    logger.info("=" * 80)
    
    from espcn.benchmark_inference import benchmark_all_models, plot_benchmark_results, plot_performance_metrics
    
    # Collect all models
    models_dict = {
        'Base': model_base,
        'LSQ': model_lsq,
        'PACT': model_pact,
        'AdaRound': model_ada,
        'APoT': model_apot,
        'DoReFa': model_dorefa,
        'STE': model_ste,
    }
    
    # Run benchmark
    benchmark_results = benchmark_all_models(models_dict, test_loader, device, num_runs=100)
    
    # Plot benchmark results
    bench_plot = plot_benchmark_results(benchmark_results, output_dir='espcn/benchmarks')
    
    # Collect PSNR/SSIM metrics from training results (already in memory!)
    metrics_dict = {
        'Base': {'psnr': result_base['best_psnr'], 'ssim': result_base['best_ssim']},
        'LSQ': {'psnr': result_lsq['best_psnr'], 'ssim': result_lsq['best_ssim']},
        'PACT': {'psnr': result_pact['best_psnr'], 'ssim': result_pact['best_ssim']},
        'AdaRound': {'psnr': result_ada['best_psnr'], 'ssim': result_ada['best_ssim']},
        'APoT': {'psnr': result_apot['best_psnr'], 'ssim': result_apot['best_ssim']},
        'DoReFa': {'psnr': result_dorefa['best_psnr'], 'ssim': result_dorefa['best_ssim']},
        'STE': {'psnr': result_ste['best_psnr'], 'ssim': result_ste['best_ssim']},
    }
    
    # Plot performance trade-offs
    perf_plot = plot_performance_metrics(benchmark_results, metrics_dict, output_dir='espcn/benchmarks')
    
    # Log to W&B
    if USE_WANDB:
        import wandb
        wandb.init(project=WANDB_PROJECT, name='Benchmark-Summary', reinit=True)
        
        # Log benchmark metrics
        for method, stats in benchmark_results.items():
            wandb.log({
                f'{method}/inference_time_ms': stats['mean_time_ms'],
                f'{method}/model_size_mb': stats['model_size_mb'],
                f'{method}/fps': stats['fps'],
            })
        
        # Log plots
        wandb.log({
            'benchmark/inference': wandb.Image(bench_plot),
            'benchmark/tradeoffs': wandb.Image(perf_plot) if metrics_dict else None,
        })
        
        wandb.finish()
    
    logger.info(f"Benchmark results saved to: espcn/benchmarks/")
    
    # Visualize quantization effects for all methods
    logger.info("=" * 80)
    logger.info("Generating QAT visualizations...")
    logger.info("=" * 80)
    
    from espcn.qat_visualizer import visualize_qat_methods, visualize_sr_results
    
    # Get a sample batch for visualization
    sample_lr, sample_hr = next(iter(test_loader))
    sample_lr = sample_lr[:4].to(device)  # Use 4 samples
    sample_hr = sample_hr[:4].to(device)
    
    # Generate QAT layer visualizations
    logger.info("Generating QAT layer visualizations...")
    viz_paths = visualize_qat_methods(
        models_dict, 
        sample_lr, 
        output_dir='espcn/qat_visualizations',
        use_wandb=USE_WANDB
    )
    
    # Generate SR results comparison
    logger.info("Generating SR results comparison...")
    sr_viz_path = visualize_sr_results(
        models_dict,
        sample_lr,
        sample_hr,
        output_dir='espcn/qat_visualizations',
        use_wandb=USE_WANDB
    )
    
    logger.info(f"Visualizations saved to: espcn/qat_visualizations/")
    
    # QAT Debugger: Generate 12-panel quantization analysis for each method
    logger.info("=" * 80)
    logger.info("Generating QAT debug visualizations (12-panel analysis)...")
    logger.info("=" * 80)
    
    from espcn.qat_debugger import debug_espcn_model
    
    qat_debug_paths = []
    lr_one = sample_lr[:1]  # Use single image for debug
    
    for method_name, model in models_dict.items():
        logger.info(f"Debugging {method_name}...")
        paths = debug_espcn_model(
            model, 
            lr_one, 
            method_name=method_name,
            output_dir='espcn/qat_debug'
        )
        qat_debug_paths.extend(paths)
    
    logger.info(f"QAT debug visualizations saved to: espcn/qat_debug/")
    
    # Log QAT debug plots to W&B
    if USE_WANDB and qat_debug_paths:
        import wandb
        wandb.init(project=WANDB_PROJECT, name='QAT-Debug', reinit=True)
        
        for plot_path in qat_debug_paths:
            method = os.path.basename(plot_path).split('_')[0]
            wandb.log({f'qat_debug/{method}': wandb.Image(plot_path)})
        
        wandb.finish()
        logger.info(f"Logged {len(qat_debug_paths)} QAT debug plots to W&B")
    
    logger.info("=" * 80)
    logger.info("Experiment complete!")
    logger.info("=" * 80)

