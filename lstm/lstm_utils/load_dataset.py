import datasets
import torch
import random
import numpy as np    

import logging
logger = logging.getLogger(__name__)
    
def load_data(dataset_name: str = 'stanfordnlp/imdb'):
    logger.info("Load data")
    ds = datasets.load_dataset(dataset_name)
    ds.pop('unsupervised')
    return ds




