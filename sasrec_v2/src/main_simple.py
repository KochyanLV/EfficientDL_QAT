import argparse
import logging
import os

import torch
from torch.nn import init
from torch import optim

from utils import get_device, DatasetArgs, ModelArgs, OptimizerArgs, TrainerArgs
from dataset import Dataset
from model import SASRec
from trainer import Trainer


logger = logging.getLogger()


def get_simple_args() -> argparse.Namespace:
    """Simplified argument parser without mlflow"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--random_seed", default=42, type=int)
    
    # Dataset arguments
    parser.add_argument("--max_seq_len", default=50, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--data_root", default="../data", type=str)
    parser.add_argument("--data_filename", default="movie-lens_1m.txt", type=str)
    
    # Model arguments
    parser.add_argument("--hidden_dim", default=50, type=int)
    parser.add_argument("--num_blocks", default=2, type=int)
    parser.add_argument("--dropout_p", default=0.5, type=float)
    parser.add_argument("--share_item_emb", action="store_true", default=False)
    
    # Optimizer arguments
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.999, type=float)
    parser.add_argument("--eps", default=1e-8, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    
    # Trainer arguments
    parser.add_argument("--device", default="", type=str)
    parser.add_argument("--evaluate_k", default=10, type=int)
    parser.add_argument("--num_epochs", default=50, type=int)
    parser.add_argument("--early_stop_epoch", default=20, type=int)
    parser.add_argument("--use_scheduler", action="store_true", default=False)
    parser.add_argument("--warmup_ratio", default=0.05, type=float)
    parser.add_argument("--scheduler_type", default="onecycle", type=str)
    parser.add_argument("--output_dir", default="../outputs", type=str)
    
    args = parser.parse_args()
    args.device = get_device()
    
    return args


def main() -> None:
    args = get_simple_args()
    
    torch.manual_seed(args.random_seed)
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        format="[%(asctime)s - %(levelname)s - %(filename)s: %(lineno)d] %(message)s",
        level=log_level,
        handlers=[logging.StreamHandler()],
    )
    
    if args.debug:
        logger.warning("Debugging mode is turned on.")
        args.num_epochs = 1
    
    logger.info("Starting SASRec training...")
    logger.info(f"Device: {args.device}")
    
    # Create dataset
    dataset_args = DatasetArgs(args)
    dataset = Dataset(**vars(dataset_args))
    
    # Create model
    args.num_items = dataset.num_items
    model_args = ModelArgs(args)
    model = SASRec(**vars(model_args))
    
    # Xavier initialization
    for param in model.parameters():
        try:
            init.xavier_uniform_(param.data)
        except ValueError:
            continue
    
    model = model.to(args.device)
    
    # Create optimizer
    optimizer_args = OptimizerArgs(args)
    optimizer = optim.Adam(params=model.parameters(), **vars(optimizer_args))
    
    # Create save directory
    args.save_dir = os.path.join(args.output_dir, "sasrec_base")
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create trainer
    trainer_args = TrainerArgs(args)
    trainer_args.resume_training = False
    trainer = Trainer(
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        **vars(trainer_args),
    )
    
    # Train
    best_results = trainer.train()
    best_ndcg_epoch, best_model_state_dict, _ = best_results
    
    # Test
    model.load_state_dict(best_model_state_dict)
    logger.info(f"Testing with model checkpoint from epoch {best_ndcg_epoch}...")
    test_ndcg, test_hit_rate = trainer.evaluate(mode="test", model=model)
    
    test_result_msg = (
        f"\nTest Results:\n"
        f"  nDCG@{trainer_args.evaluate_k}: {test_ndcg: 0.6f}\n"
        f"  Hit@{trainer_args.evaluate_k}:  {test_hit_rate: 0.6f}"
    )
    logger.info(test_result_msg)


if __name__ == "__main__":
    main()

