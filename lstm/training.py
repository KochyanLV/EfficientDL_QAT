from lstm.model.LstmAdaRound import QuantLSTMAdaRound
from lstm.model.LstmApot import QuantLSTMAPoT
from lstm.model.LstmDorefa import QuantLSTMDoReFa
from lstm.model.LstmLSQ import QuantLSTMLSQ
from lstm.model.LstmPact import QuantLSTMPACT
from lstm.model.LstmSTE import QuantLSTMSTE
from lstm.model.Base import BaseLSTM

from lstm.train_utils.cycle import fit
from lstm.train_utils.loaders import make_loaders
from lstm.lstm_utils.seeds import set_seed

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


EPOCHS = 5
BATCH_SIZE = 8
EMB_DIM = 128
HIDDEN_DIM = 256
LR = 1e-3


if __name__ == "__main__":
    set_seed(42)
    
    train_loader, test_loader, tokenizer = make_loaders(tokenizer_path="lstm/own_tokenizer/", batch_size=BATCH_SIZE)
    # base without quant
    logger.info("Model: Base")
    model_base = BaseLSTM(
        vocab_size=tokenizer.vocab_size,
        emb_dim=EMB_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=1,
    )
    model_base = model_base.to(device)
    
    
    # lsq quant
    logger.info("Model: LSQ")
    model_lsq = QuantLSTMLSQ(
        vocab_size=tokenizer.vocab_size,
        emb_dim=EMB_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=1,
        bits=8
    )
    model_lsq = fit(model_lsq, train_loader, test_loader, device, epochs=EPOCHS, lr=LR)
    
    # pact quant
    logger.info("Model: PACT")
    model_pact = QuantLSTMPACT(    
        vocab_size=tokenizer.vocab_size,
        emb_dim=EMB_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=1,
        bits_act=8, 
        bits_w=8, 
        pact_init_alpha=6.0
    )
    model_pact = fit(model_pact, train_loader, test_loader, device, epochs=EPOCHS, lr=LR)
        
    # adaround quant
    logger.info("Model: AdaRound")
    model_ada = QuantLSTMAdaRound(
        vocab_size=tokenizer.vocab_size,
        emb_dim=EMB_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=1,
        bits_w=8
    )
    model_ada = fit(model_ada, train_loader, test_loader, device, epochs=EPOCHS, lr=LR)
    
    # apot quant
    logger.info("Model: APOT")
    model_apot = QuantLSTMAPoT(
        vocab_size=tokenizer.vocab_size,
        emb_dim=EMB_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=1,
        bits=8,
        k=1,
        init_alpha_act=6.0,
        init_alpha_w=2.0,
        use_weight_norm_w=True
    )
    model_apot = fit(model_apot, train_loader, test_loader, device, epochs=EPOCHS, lr=LR)
    
    # dorefa quant
    logger.info("Model: DoREFA")
    model_dorefa = QuantLSTMDoReFa(
        vocab_size=tokenizer.vocab_size,
        emb_dim=EMB_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=1,
        bits_a=8,
        bits_w=8,
        act_signed=True,
        act_preproc="tanh",
    )
    model_dorefa = fit(model_dorefa, train_loader, test_loader, device, epochs=EPOCHS, lr=LR)
    
    # fake quant
    logger.info("Model: Fake")
    model_fake = QuantLSTMSTE(
        vocab_size=tokenizer.vocab_size,
        emb_dim=EMB_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=1,
        bits=8
    )
    model_fake = fit(model_fake, train_loader, test_loader, device, epochs=EPOCHS, lr=LR)
    