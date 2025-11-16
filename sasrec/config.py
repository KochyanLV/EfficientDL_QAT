"""
Global configuration for SASRec training with quantization methods.
"""
import torch

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset parameters
DATA_FILEPATH = "data/movie-lens_1m.txt"
BATCH_SIZE = 128
MAX_SEQ_LEN = 50

# Model architecture
HIDDEN_DIM = 50
NUM_BLOCKS = 2
DROPOUT = 0.5
SHARE_ITEM_EMB = False

# Training parameters
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 0.0
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8

# Scheduler (optional)
USE_SCHEDULER = False
SCHEDULER_TYPE = "onecycle"
WARMUP_RATIO = 0.05

# Evaluation
EVALUATE_K = 10
EARLY_STOP_EPOCH = 10

# Quantization parameters
BITS = 8  # INT8 quantization
PACT_INIT_ALPHA = 6.0
APOT_K = 1
APOT_INIT_ALPHA_ACT = 6.0
APOT_USE_WEIGHT_NORM = False
DOREFA_ACT_SIGNED = True
DOREFA_ACT_PREPROC = "tanh"

# Paths
OUTPUT_DIR = "outputs"
LOG_DIR = "logs"

# Other
RANDOM_SEED = 42
DEBUG = False

