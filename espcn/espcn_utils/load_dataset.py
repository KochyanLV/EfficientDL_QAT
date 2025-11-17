import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from datasets import load_dataset as hf_load_dataset
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class SuperResolutionDataset(Dataset):
    """
    Dataset for super-resolution using DIV2K or similar datasets.
    DIV2K already provides LR-HR pairs, so we just convert them to tensors.
    """
    def __init__(self, images, scale_factor=4, patch_size=192):
        self.images = images
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.lr_patch_size = patch_size // scale_factor
        
        self.to_tensor = transforms.ToTensor()
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        item = self.images[idx]
        
        # Extract image from dataset item
        if isinstance(item, dict):
            # ImageNet and similar: {'image': PIL_Image, ...}
            if 'image' in item:
                hr_image = item['image']
            elif 'img' in item:
                hr_image = item['img']
            else:
                raise ValueError(f"Could not find image in dict with keys: {item.keys()}")
        else:
            hr_image = item
        
        # Convert to tensor
        if isinstance(hr_image, Image.Image):
            hr_image = self.to_tensor(hr_image)
        
        # Get image dimensions
        c, h, w = hr_image.shape
        
        # Ensure image is large enough for patch extraction
        if h < self.patch_size or w < self.patch_size:
            # Resize to minimum required size
            scale = max(self.patch_size / h, self.patch_size / w) * 1.1
            new_h = int(h * scale)
            new_w = int(w * scale)
            hr_image = torch.nn.functional.interpolate(
                hr_image.unsqueeze(0),
                size=(new_h, new_w),
                mode='bicubic',
                align_corners=False
            ).squeeze(0)
            h, w = new_h, new_w
        
        # Random crop HR patch
        top = torch.randint(0, h - self.patch_size + 1, (1,)).item()
        left = torch.randint(0, w - self.patch_size + 1, (1,)).item()
        hr_patch = hr_image[:, top:top+self.patch_size, left:left+self.patch_size]
        
        # Create LR by downsampling HR
        lr_patch = torch.nn.functional.interpolate(
            hr_patch.unsqueeze(0),
            size=(self.lr_patch_size, self.lr_patch_size),
            mode='bicubic',
            align_corners=False
        ).squeeze(0)
        
        hr_patch = hr_patch.clamp(0, 1)
        lr_patch = lr_patch.clamp(0, 1)
        
        return lr_patch, hr_patch


def load_div2k_from_hf(split='train', num_images=None):
    """
    Load high-resolution images for SR from HuggingFace.
    Using Food101 - no auth required, high-quality images.
    """
    logger.info(f"Loading high-resolution images for SR (split: {split})")
    
    # Food101: большие изображения (~512×384), не требует авторизации
    hf_split = 'train' if split == 'train' else 'validation'
    
    logger.info(f"Using Food101 dataset (high-res images, no auth required)")
    dataset = hf_load_dataset('food101', split=hf_split, streaming=True, trust_remote_code=True)
    
    images = []
    target = num_images if num_images else 800
    
    for idx, item in enumerate(dataset):
        if idx >= target:
            break
        images.append(item)
        if (idx + 1) % 100 == 0:
            logger.info(f"Loaded {idx + 1}/{target} images")
    
    logger.info(f"Loaded {len(images)} images for SR")
    return images




def load_data(num_train=200, num_test=50, scale_factor=4):
    """
    Load high-resolution images for super-resolution training.
    
    Uses Food101 which provides:
    - High-quality images (~512×384 average)
    - No authentication required
    - Images loaded in memory
    
    Args:
        num_train: Number of training images (default 200 for fast experiments)
        num_test: Number of validation images (default 50)
        scale_factor: Upscaling factor (2, 3, or 4)
        
    Returns:
        train_images, test_images
    """
    logger.info(f"Loading dataset for {scale_factor}× Super-Resolution")
    train_images = load_div2k_from_hf(split='train', num_images=num_train)
    test_images = load_div2k_from_hf(split='validation', num_images=num_test)
    return train_images, test_images

