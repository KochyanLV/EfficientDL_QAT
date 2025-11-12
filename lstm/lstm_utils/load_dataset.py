import datasets
import torch
import random
import numpy as np    
    
    
def load_data(dataset_name: str = 'stanfordnlp/imdb'):
    ds = datasets.load_dataset(dataset_name)
    ds.pop('unsupervised')
    return ds




