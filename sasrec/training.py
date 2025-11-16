from sasrec.model.SasrecAdaRound import QuantSASRecAdaRound
from sasrec.model.SasrecApot import QuantSASRecAPoT
from sasrec.model.SasrecDorefa import QuantSASRecDoReFa
from sasrec.model.SasrecLSQ import QuantSASRecLSQ
from sasrec.model.SasrecPact import QuantSASRecPACT
from sasrec.model.SasrecSTE import QuantSASRecSTE
from sasrec.model.Base import BaseSASRec

from sasrec.train_utils.cycle import fit
from sasrec.train_utils.loaders import make_loaders
from sasrec.sasrec_utils.seeds import set_seed

import torch

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 32
EMBED_DIM = 128
NUM_HEADS = 4
NUM_BLOCKS = 2
MAX_LEN = 50
DROPOUT = 0.2
LR = 1e-3


if __name__ == "__main__":
    set_seed(42)
    
    # Create dataloaders
    train_loader, test_loader, num_users, num_items = make_loaders(
        batch_size=BATCH_SIZE, 
        max_len=MAX_LEN
    )
    
    logger.info(f"Number of items (including padding): {num_items}")
    logger.info(f"Device: {device}")
    
    logger.info("=" * 80)
    logger.info("Model: Base SASRec (no quantization)")
    logger.info("=" * 80)
    model_base = BaseSASRec(
        num_items=num_items,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        max_len=MAX_LEN,
        dropout=DROPOUT,
    )
    model_base = fit(model_base, train_loader, test_loader, device, epochs=EPOCHS, lr=LR)
    
    logger.info("=" * 80)
    logger.info("Model: SASRec + LSQ")
    logger.info("=" * 80)
    model_lsq = QuantSASRecLSQ(
        num_items=num_items,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        max_len=MAX_LEN,
        dropout=DROPOUT,
        bits=8
    )
    model_lsq = fit(model_lsq, train_loader, test_loader, device, epochs=EPOCHS, lr=LR)
    
    logger.info("=" * 80)
    logger.info("Model: SASRec + PACT")
    logger.info("=" * 80)
    model_pact = QuantSASRecPACT(
        num_items=num_items,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        max_len=MAX_LEN,
        dropout=DROPOUT,
        bits_act=8,
        bits_w=8,
        pact_init_alpha=6.0
    )
    model_pact = fit(model_pact, train_loader, test_loader, device, epochs=EPOCHS, lr=LR)
    
    logger.info("=" * 80)
    logger.info("Model: SASRec + AdaRound")
    logger.info("=" * 80)
    model_ada = QuantSASRecAdaRound(
        num_items=num_items,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        max_len=MAX_LEN,
        dropout=DROPOUT,
        bits_w=8
    )
    model_ada = fit(model_ada, train_loader, test_loader, device, epochs=EPOCHS, lr=LR)
    
    logger.info("=" * 80)
    logger.info("Model: SASRec + APoT")
    logger.info("=" * 80)
    model_apot = QuantSASRecAPoT(
        num_items=num_items,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        max_len=MAX_LEN,
        dropout=DROPOUT,
        bits=8,
        k=1,
        init_alpha_act=6.0,
        init_alpha_w=2.0,
        use_weight_norm_w=True
    )
    model_apot = fit(model_apot, train_loader, test_loader, device, epochs=EPOCHS, lr=LR)
    
    logger.info("=" * 80)
    logger.info("Model: SASRec + DoReFa")
    logger.info("=" * 80)
    model_dorefa = QuantSASRecDoReFa(
        num_items=num_items,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        max_len=MAX_LEN,
        dropout=DROPOUT,
        bits_a=8,
        bits_w=8,
        act_signed=True,
        act_preproc="tanh"
    )
    model_dorefa = fit(model_dorefa, train_loader, test_loader, device, epochs=EPOCHS, lr=LR)
    
    logger.info("=" * 80)
    logger.info("Model: SASRec + Fake STE")
    logger.info("=" * 80)
    model_fake = QuantSASRecSTE(
        num_items=num_items,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        max_len=MAX_LEN,
        dropout=DROPOUT,
        bits=8
    )
    model_fake = fit(model_fake, train_loader, test_loader, device, epochs=EPOCHS, lr=LR)
    
    logger.info("=" * 80)
    logger.info("All models trained successfully!")
    logger.info("=" * 80)

