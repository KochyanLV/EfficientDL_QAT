from espcn.espcn_utils.load_dataset import load_data, SuperResolutionDataset
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)


def make_loaders(batch_size: int = 16, scale_factor: int = 4, 
                 patch_size: int = 192, num_train: int = 800, num_test: int = 100):
    """
    Create train and test dataloaders for super-resolution.
    
    Args:
        batch_size: Batch size for training
        scale_factor: Upscaling factor (2, 3, or 4)
        patch_size: Size of HR patches to extract
        num_train: Number of training images from DIV2K (max 800)
        num_test: Number of validation images from DIV2K (max 100)
    
    Returns:
        train_loader, test_loader
    """
    logger.info("Creating data loaders for ESPCN")
    
    train_images, test_images = load_data(
        num_train=num_train, 
        num_test=num_test, 
        scale_factor=scale_factor
    )
    
    train_dataset = SuperResolutionDataset(train_images, scale_factor=scale_factor, patch_size=patch_size)
    test_dataset = SuperResolutionDataset(test_images, scale_factor=scale_factor, patch_size=patch_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    logger.info(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    return train_loader, test_loader

