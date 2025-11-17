import torch.nn as nn
import torch.nn.functional as F
from espcn.train_utils.save_checkpoint import record_init


@record_init
class BaseESPCN(nn.Module):
    """
    Efficient Sub-Pixel Convolutional Neural Network (ESPCN) for super-resolution.
    
    Architecture:
    - Conv layers for feature extraction
    - Sub-pixel convolution (PixelShuffle) for upsampling
    
    Quantization hooks:
      - quant_conv1_out(x): quantize after first conv
      - quant_conv2_out(x): quantize after second conv
      - quant_conv3_out(x): quantize after third conv (before pixel shuffle)
      - quant_conv1_weight(w): quantize conv1 weights
      - quant_conv2_weight(w): quantize conv2 weights
      - quant_conv3_weight(w): quantize conv3 weights
    
    By default, these are no-op.
    """
    def __init__(self, scale_factor: int = 3, num_channels: int = 3, 
                 feature_dim: int = 64):
        super().__init__()
        self.scale_factor = scale_factor
        self.num_channels = num_channels
        self.feature_dim = feature_dim
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(num_channels, feature_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(feature_dim // 2, num_channels * (scale_factor ** 2), 
                               kernel_size=3, padding=1)
        
        # Sub-pixel convolution (pixel shuffle)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        
    # Quantization hooks for activations (no-op by default)
    def quant_conv1_out(self, x):
        return x
    
    def quant_conv2_out(self, x):
        return x
    
    def quant_conv3_out(self, x):
        return x
    
    # Quantization hooks for weights (no-op by default)
    def quant_conv1_weight(self, w):
        return w
    
    def quant_conv2_weight(self, w):
        return w
    
    def quant_conv3_weight(self, w):
        return w
    
    def forward(self, x):
        """
        Args:
            x: Low-resolution images (B, C, H, W)
        
        Returns:
            Super-resolved images (B, C, H*scale, W*scale)
        """
        # Conv1 with quantization
        w1 = self.quant_conv1_weight(self.conv1.weight)
        x = F.conv2d(x, w1, self.conv1.bias, padding=2)
        x = F.relu(x)
        x = self.quant_conv1_out(x)
        
        # Conv2 with quantization
        w2 = self.quant_conv2_weight(self.conv2.weight)
        x = F.conv2d(x, w2, self.conv2.bias, padding=1)
        x = F.relu(x)
        x = self.quant_conv2_out(x)
        
        # Conv3 with quantization (no ReLU before pixel shuffle)
        w3 = self.quant_conv3_weight(self.conv3.weight)
        x = F.conv2d(x, w3, self.conv3.bias, padding=1)
        x = self.quant_conv3_out(x)
        
        # Sub-pixel convolution (upsampling)
        x = self.pixel_shuffle(x)
        
        return x

