"""
Training script for SASRec with various quantization methods.
Similar to lstm/training.py but adapted for SASRec architecture.
"""
import sys
import os
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from torch.nn import init
from torch import optim

from src.dataset import Dataset
from src.model import (
    SASRec,
    QuantSASRecLSQ,
    QuantSASRecPACT,
    QuantSASRecAdaRound,
    QuantSASRecAPoT,
    QuantSASRecDoReFa,
    QuantSASRecSTE,
)
from src.trainer import Trainer

# Import global config
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore")


def create_dataset():
    """Create dataset from movie-lens data"""
    data_filepath = os.path.join(os.path.dirname(__file__), config.DATA_FILEPATH)
    
    dataset = Dataset(
        batch_size=config.BATCH_SIZE,
        max_seq_len=config.MAX_SEQ_LEN,
        data_filepath=data_filepath,
        debug=config.DEBUG,
    )
    
    logger.info(f"Dataset loaded: {dataset.num_users} users, {dataset.num_items} items")
    logger.info(f"Train samples: {len(dataset.user2items_train)}")
    logger.info(f"Valid samples: {len(dataset.user2items_valid)}")
    logger.info(f"Test samples: {len(dataset.user2items_test)}")
    
    return dataset


def train_model(model, dataset, device, model_name="model"):
    """Train a single model"""
    logger.info(f"=" * 80)
    logger.info(f"Training: {model_name}")
    logger.info(f"=" * 80)
    
    # Xavier initialization
    for param in model.parameters():
        try:
            init.xavier_uniform_(param.data)
        except ValueError:
            continue
    
    model = model.to(device)
    
    # Create optimizer
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=config.LR,
        betas=(config.BETA1, config.BETA2),
        eps=config.EPS,
        weight_decay=config.WEIGHT_DECAY,
    )
    
    # Create save directory
    save_dir = os.path.join(os.path.dirname(__file__), config.OUTPUT_DIR, model_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Create trainer
    trainer = Trainer(
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        evaluate_k=config.EVALUATE_K,
        max_lr=config.LR,
        num_epochs=config.EPOCHS,
        early_stop_epoch=config.EARLY_STOP_EPOCH,
        warmup_ratio=config.WARMUP_RATIO,
        use_scheduler=config.USE_SCHEDULER,
        scheduler_type=config.SCHEDULER_TYPE,
        save_dir=save_dir,
        resume_training=False,
        device=str(device),
    )
    
    # Train
    best_results = trainer.train()
    best_ndcg_epoch, best_model_state_dict, _ = best_results
    
    # Test
    model.load_state_dict(best_model_state_dict)
    logger.info(f"Testing with model checkpoint from epoch {best_ndcg_epoch}...")
    test_ndcg, test_hit_rate = trainer.evaluate(mode="test", model=model)
    
    test_result_msg = (
        f"\nTest Results for {model_name}:\n"
        f"  nDCG@{config.EVALUATE_K}: {test_ndcg: 0.6f}\n"
        f"  Hit@{config.EVALUATE_K}:  {test_hit_rate: 0.6f}"
    )
    logger.info(test_result_msg)
    
    return model, test_ndcg, test_hit_rate


if __name__ == "__main__":
    torch.manual_seed(config.RANDOM_SEED)
    device = config.DEVICE
    logger.info(f"Device: {device}")
    
    # Create dataset
    dataset = create_dataset()
    num_items = dataset.num_items
    
    # 1. Base SASRec (no quantization)
    model_base = SASRec(
        num_items=num_items,
        num_blocks=config.NUM_BLOCKS,
        hidden_dim=config.HIDDEN_DIM,
        max_seq_len=config.MAX_SEQ_LEN,
        dropout_p=config.DROPOUT,
        share_item_emb=config.SHARE_ITEM_EMB,
        device=str(device),
    )
    train_model(model_base, dataset, device, model_name="sasrec_base")
    
    # 2. SASRec + LSQ
    model_lsq = QuantSASRecLSQ(
        num_items=num_items,
        num_blocks=config.NUM_BLOCKS,
        hidden_dim=config.HIDDEN_DIM,
        max_seq_len=config.MAX_SEQ_LEN,
        dropout_p=config.DROPOUT,
        share_item_emb=config.SHARE_ITEM_EMB,
        device=str(device),
        bits=config.BITS,
    )
    train_model(model_lsq, dataset, device, model_name="sasrec_lsq")
    
    # 3. SASRec + PACT
    model_pact = QuantSASRecPACT(
        num_items=num_items,
        num_blocks=config.NUM_BLOCKS,
        hidden_dim=config.HIDDEN_DIM,
        max_seq_len=config.MAX_SEQ_LEN,
        dropout_p=config.DROPOUT,
        share_item_emb=config.SHARE_ITEM_EMB,
        device=str(device),
        bits_act=config.BITS,
        pact_init_alpha=config.PACT_INIT_ALPHA,
    )
    train_model(model_pact, dataset, device, model_name="sasrec_pact")
    
    # 4. SASRec + AdaRound
    model_ada = QuantSASRecAdaRound(
        num_items=num_items,
        num_blocks=config.NUM_BLOCKS,
        hidden_dim=config.HIDDEN_DIM,
        max_seq_len=config.MAX_SEQ_LEN,
        dropout_p=config.DROPOUT,
        share_item_emb=config.SHARE_ITEM_EMB,
        device=str(device),
        bits_w=config.BITS,
    )
    train_model(model_ada, dataset, device, model_name="sasrec_adaround")
    
    # 5. SASRec + APoT
    model_apot = QuantSASRecAPoT(
        num_items=num_items,
        num_blocks=config.NUM_BLOCKS,
        hidden_dim=config.HIDDEN_DIM,
        max_seq_len=config.MAX_SEQ_LEN,
        dropout_p=config.DROPOUT,
        share_item_emb=config.SHARE_ITEM_EMB,
        device=str(device),
        bits=config.BITS,
        k=config.APOT_K,
        init_alpha_act=config.APOT_INIT_ALPHA_ACT,
        use_weight_norm_w=config.APOT_USE_WEIGHT_NORM,
    )
    train_model(model_apot, dataset, device, model_name="sasrec_apot")
    
    # 6. SASRec + DoReFa
    model_dorefa = QuantSASRecDoReFa(
        num_items=num_items,
        num_blocks=config.NUM_BLOCKS,
        hidden_dim=config.HIDDEN_DIM,
        max_seq_len=config.MAX_SEQ_LEN,
        dropout_p=config.DROPOUT,
        share_item_emb=config.SHARE_ITEM_EMB,
        device=str(device),
        bits_a=config.BITS,
        act_signed=config.DOREFA_ACT_SIGNED,
        act_preproc=config.DOREFA_ACT_PREPROC,
    )
    train_model(model_dorefa, dataset, device, model_name="sasrec_dorefa")
    
    # 7. SASRec + Fake STE
    model_ste = QuantSASRecSTE(
        num_items=num_items,
        num_blocks=config.NUM_BLOCKS,
        hidden_dim=config.HIDDEN_DIM,
        max_seq_len=config.MAX_SEQ_LEN,
        dropout_p=config.DROPOUT,
        share_item_emb=config.SHARE_ITEM_EMB,
        device=str(device),
        bits=config.BITS,
    )
    train_model(model_ste, dataset, device, model_name="sasrec_ste")
    
    logger.info("=" * 80)
    logger.info("All models trained successfully!")
    logger.info("=" * 80)

