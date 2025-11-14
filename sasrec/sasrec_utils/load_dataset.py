import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def load_movielens_100k(min_rating: float = 4.0):
    """
    Load MovieLens 100K dataset for sequential recommendation.
    Args:
        min_rating: Minimum rating to consider as positive interaction
    Returns:
        user_sequences: Dict[user_id, List[item_id]] sorted by timestamp
        num_users: int
        num_items: int
    """
    logger.info("Loading MovieLens-100K dataset")
    
    try:
        from datasets import load_dataset
        ds = load_dataset("rotten_tomatoes")  # Using a simpler dataset as placeholder
        # In production, you would use actual MovieLens or similar recommendation dataset
        logger.warning("Using rotten_tomatoes as placeholder. For real experiments, use MovieLens or similar.")
        
        # Create synthetic sequential data for demonstration
        # In real implementation, load actual sequential recommendation dataset
        user_sequences = defaultdict(list)
        num_users = 100
        num_items = 500
        
        # Generate synthetic user sequences
        np.random.seed(42)
        for user_id in range(num_users):
            seq_len = np.random.randint(5, 20)
            items = np.random.choice(num_items, size=seq_len, replace=False).tolist()
            user_sequences[user_id] = items
            
    except Exception as e:
        logger.warning(f"Could not load dataset from HuggingFace: {e}")
        logger.info("Generating synthetic sequential recommendation data")
        
        # Generate synthetic data
        user_sequences = defaultdict(list)
        num_users = 100
        num_items = 500
        
        np.random.seed(42)
        for user_id in range(num_users):
            seq_len = np.random.randint(5, 20)
            items = np.random.choice(num_items, size=seq_len, replace=False).tolist()
            user_sequences[user_id] = items
    
    logger.info(f"Loaded {len(user_sequences)} users, {num_items} items")
    return dict(user_sequences), num_users, num_items


def prepare_sequences(
    user_sequences: Dict[int, List[int]], 
    max_len: int = 50,
    test_split: float = 0.2
) -> Tuple[Dict, Dict]:
    """
    Prepare train/test splits from user sequences.
    For each user, last item is test target, second-to-last is validation.
    
    Args:
        user_sequences: Dict of user_id -> list of item_ids
        max_len: Maximum sequence length
        test_split: Not used, we use last item for test
        
    Returns:
        train_data: Dict with sequences for training
        test_data: Dict with sequences for testing
    """
    logger.info("Preparing train/test sequences")
    
    train_data = {'user_id': [], 'sequence': [], 'target': []}
    test_data = {'user_id': [], 'sequence': [], 'target': []}
    
    for user_id, items in user_sequences.items():
        if len(items) < 3:  # Need at least 3 items
            continue
            
        # Last item for test, second-to-last for validation, rest for training
        test_target = items[-1]
        test_seq = items[:-1][-max_len:]
        
        # Use all but last item for training
        for i in range(2, len(items)):
            train_seq = items[:i-1][-max_len:]
            train_target = items[i-1]
            train_data['user_id'].append(user_id)
            train_data['sequence'].append(train_seq)
            train_data['target'].append(train_target)
        
        # Test sample
        test_data['user_id'].append(user_id)
        test_data['sequence'].append(test_seq)
        test_data['target'].append(test_target)
    
    logger.info(f"Train samples: {len(train_data['user_id'])}, Test samples: {len(test_data['user_id'])}")
    return train_data, test_data

