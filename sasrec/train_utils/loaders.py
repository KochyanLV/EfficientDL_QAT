import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List
import logging

from sasrec.sasrec_utils.load_dataset import load_movielens, prepare_sequences

logger = logging.getLogger(__name__)


class SASRecDataset(Dataset):
    """
    Dataset for SASRec training.
    Each sample contains:
        - sequence: list of item IDs (variable length, will be padded)
        - target: next item ID to predict
        - user_id: user identifier
    """
    def __init__(self, data: Dict[str, List], max_len: int = 50, num_items: int = 500):
        self.user_ids = data['user_id']
        self.sequences = data['sequence']
        self.targets = data['target']
        self.max_len = max_len
        self.num_items = num_items
        
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        target = self.targets[idx]
        user_id = self.user_ids[idx]
        
        seq_len = len(seq)
        if seq_len < self.max_len:
            padded_seq = [0] * (self.max_len - seq_len) + seq
        else:
            padded_seq = seq[-self.max_len:]
        
        return {
            'sequence': torch.LongTensor(padded_seq),
            'target': torch.LongTensor([target]),
            'user_id': torch.LongTensor([user_id]),
            'seq_len': torch.LongTensor([min(seq_len, self.max_len)])
        }


def collate_fn(batch):
    """Collate function for DataLoader"""
    sequences = torch.stack([item['sequence'] for item in batch])
    targets = torch.cat([item['target'] for item in batch])
    user_ids = torch.cat([item['user_id'] for item in batch])
    seq_lens = torch.cat([item['seq_len'] for item in batch])
    
    return {
        'sequence': sequences,
        'target': targets,
        'user_id': user_ids,
        'seq_len': seq_lens
    }


def make_loaders(batch_size: int = 32, max_len: int = 50, num_workers: int = 0):
    """
    Create train and test dataloaders for SASRec.
    
    Args:
        batch_size: batch size for training
        max_len: maximum sequence length
        num_workers: number of workers for DataLoader
        
    Returns:
        train_loader, test_loader, num_users, num_items
    """
    logger.info("Creating dataloaders for SASRec")
    
    user_sequences, num_users, num_items = load_movielens()
    train_data, test_data = prepare_sequences(user_sequences, max_len=max_len)
    
    train_dataset = SASRecDataset(train_data, max_len=max_len, num_items=num_items)
    test_dataset = SASRecDataset(test_data, max_len=max_len, num_items=num_items)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    logger.info(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    logger.info(f"Num users: {num_users}, Num items: {num_items}")
    
    return train_loader, test_loader, num_users, num_items + 1

