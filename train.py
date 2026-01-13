from utils.data_loader import make_datapath_list, ImageDataset, ImageTransform   
from models.VDTA_IRNet import Discriminator, VDTA_IRNet
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torch.autograd import Variable
from collections import OrderedDict
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import argparse
import time
import torch
import os
import platform

import math
import cv2

import torch.nn.functional as F

# NEW: Import for perceptual loss
from torchvision.models import vgg19, VGG19_Weights

torch.manual_seed(44)
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

# NEW: Perceptual Loss using VGG features
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        self.vgg_layers = nn.ModuleList([
            nn.Sequential(*[vgg[x] for x in range(4)]),   # relu1_2
            nn.Sequential(*[vgg[x] for x in range(4, 9)]), # relu2_2
            nn.Sequential(*[vgg[x] for x in range(9, 18)]), # relu3_4
            nn.Sequential(*[vgg[x] for x in range(18, 27)]) # relu4_4
        ])
        
        for param in self.parameters():
            param.requires_grad = False
            
        # Weights for different layers
        self.layer_weights = [1.0, 0.5, 0.25, 0.125]
    
    def forward(self, pred, target):
        loss = 0
        x = pred
        y = target
        
        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            y = layer(y)
            loss += self.layer_weights[i] * F.l1_loss(x, y)
            
        return loss

# NEW: SSIM Loss for structural similarity
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)
        
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
        
    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window
        
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel
            
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)

# ================  DEGRADED-SPECIFIC LOSSES ================
class ShadowSpecificLoss(nn.Module):
    
    def __init__(self):
        super(ShadowSpecificLoss, self).__init__()
        
    def compute_brightness_consistency_loss(self, output, target, input_img):
        """Ensures brightness consistency"""
        # Compute brightness differences
        input_brightness = torch.mean(input_img, dim=1, keepdim=True)
        output_brightness = torch.mean(output, dim=1, keepdim=True)
        target_brightness = torch.mean(target, dim=1, keepdim=True)
        
        # Degrade regions should have increased brightness
        brightness_diff_output = output_brightness - input_brightness
        brightness_diff_target = target_brightness - input_brightness
        
        brightness_loss = F.l1_loss(brightness_diff_output, brightness_diff_target)
        return brightness_loss
    
    def compute_color_consistency_loss(self, output, target, input_img):
        """Preserves color consistency while removing"""
        # Convert to LAB color space approximation
        output_lab = self.rgb_to_lab_approx(output)
        target_lab = self.rgb_to_lab_approx(target)
        input_lab = self.rgb_to_lab_approx(input_img)
        
        # Focus on preserving color differences relative to input
        # The color shift from input should match the target's shift
        output_color_shift = output_lab[:, 1:] - input_lab[:, 1:]  # Color change in output
        target_color_shift = target_lab[:, 1:] - input_lab[:, 1:]  # Color change in target
        
        # Loss ensures output changes color the same way as target
        color_loss = F.l1_loss(output_color_shift, target_color_shift)
        
        return color_loss
    
    def rgb_to_lab_approx(self, rgb):
        """Approximation of RGB to LAB conversion for differentiable computation"""
        # Simple approximation that preserves gradient flow
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        
        # Approximate L channel
        l = 0.299 * r + 0.587 * g + 0.114 * b
        
        # Approximate a and b channels
        a = 0.5 * (r - g)
        b = 0.5 * (0.25 * (r + g) - b)
        
        return torch.cat([l, a, b], dim=1)
    
    def compute_gradient_preservation_loss(self, output, target):
        """Preserves important gradients while smoothing Degrade boundaries"""
        # Compute gradients using Sobel operators
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=output.dtype, device=output.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=output.dtype, device=output.device).view(1, 1, 3, 3)
        
        # Apply to each channel
        grad_loss = 0
        for i in range(output.shape[1]):
            output_chan = output[:, i:i+1]
            target_chan = target[:, i:i+1]
            
            # Compute gradients
            output_grad_x = F.conv2d(output_chan, sobel_x, padding=1)
            output_grad_y = F.conv2d(output_chan, sobel_y, padding=1)
            target_grad_x = F.conv2d(target_chan, sobel_x, padding=1)
            target_grad_y = F.conv2d(target_chan, sobel_y, padding=1)
            
            # Gradient magnitude
            output_grad_mag = torch.sqrt(output_grad_x**2 + output_grad_y**2 + 1e-8)
            target_grad_mag = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-8)
            
            grad_loss += F.l1_loss(output_grad_mag, target_grad_mag)
        
        return grad_loss / output.shape[1]

# ================ TEXTURE PRESERVATION LOSS ================
class TexturePreservationLoss(nn.Module):
    
    def __init__(self):
        super(TexturePreservationLoss, self).__init__()
        
        # Define texture extraction filters
        self.register_buffer('texture_filters', self.create_texture_filters())
        
    def create_texture_filters(self):
        """Create a set of filters for texture analysis"""
        filters = []
        
        # Horizontal and vertical edge filters
        filters.append(torch.tensor([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=torch.float32))
        filters.append(torch.tensor([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=torch.float32))
        
        # Diagonal filters
        filters.append(torch.tensor([[0, -1, -1], [1, 0, -1], [1, 1, 0]], dtype=torch.float32))
        filters.append(torch.tensor([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]], dtype=torch.float32))
        
        # Combine into tensor
        filters_tensor = torch.stack(filters).unsqueeze(1)  # [4, 1, 3, 3]
        return filters_tensor
    
    def extract_texture_features(self, img):
        """Extract texture features using predefined filters"""
        batch_size, channels, height, width = img.shape
        texture_responses = []
        
        for c in range(channels):
            channel_img = img[:, c:c+1]  # [B, 1, H, W]
            
            # Apply all texture filters
            responses = F.conv2d(channel_img, self.texture_filters, padding=1)  # [B, 4, H, W]
            texture_responses.append(responses)
        
        # Concatenate all channel responses
        all_responses = torch.cat(texture_responses, dim=1)  # [B, 4*C, H, W]
        return all_responses
    
    def forward(self, output, target, weight=1.0):
        """Compute texture preservation loss"""
        output_textures = self.extract_texture_features(output)
        target_textures = self.extract_texture_features(target)
        
        texture_loss = F.l1_loss(output_textures, target_textures)
        return weight * texture_loss

# ================ DARK OBJECT PRESERVATION LOSS ================
class AdvancedDarkObjectPreservationLoss(nn.Module):
   
    def __init__(self, device='cuda'):
        super(AdvancedDarkObjectPreservationLoss, self).__init__()
        self.device = device
        
        # NEW: Advanced dark region detector with multiple thresholds
        self.dark_thresholds = [0.2, 0.3, 0.4, 0.5]  # Multiple thresholds for robustness
        
        # NEW: Edge strength analyzer for object boundaries
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                                    dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                                    dtype=torch.float32).view(1, 1, 3, 3))
        
        # NEW: Multi-directional edge filters for better object detection
        edge_filters = []
        edge_filters.append(torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32))  # Horizontal
        edge_filters.append(torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32))  # Vertical
        edge_filters.append(torch.tensor([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=torch.float32))  # Diagonal 1
        edge_filters.append(torch.tensor([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=torch.float32))  # Diagonal 2
        
        self.register_buffer('multi_edge_filters', torch.stack(edge_filters).unsqueeze(1))  # [4, 1, 3, 3]
        
    def detect_multi_threshold_dark_regions(self, image):
        """NEW: Multi-threshold dark region detection for robustness"""
        # Convert to grayscale
        if image.shape[1] == 3:
            gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        else:
            gray = image
        
        # Create multiple dark masks with different thresholds
        dark_masks = []
        for threshold in self.dark_thresholds:
            dark_mask = (gray < threshold).float()
            dark_masks.append(dark_mask)
        
        # Combine masks using majority voting (at least 2 out of 4 thresholds)
        combined_mask = torch.stack(dark_masks, dim=1).sum(dim=1) >= 2
        combined_mask = combined_mask.float()
        
        return combined_mask
    
    def compute_advanced_edge_strength(self, image):
        
        # Convert to grayscale for edge detection
        if image.shape[1] == 3:
            gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        else:
            gray = image
        
        # Apply all edge filters
        edge_responses = F.conv2d(gray, self.multi_edge_filters, padding=1)  # [B, 4, H, W]
        
        # Compute maximum edge response across all directions
        max_edge_response = torch.max(torch.abs(edge_responses), dim=1, keepdim=True)[0]
        
        # Apply sigmoid to get smooth edge strength
        edge_strength = torch.sigmoid(max_edge_response * 5)  # Scale factor for sharpness
        
        return edge_strength
    
    def compute_texture_variance(self, image):
        """Compute local texture variance to identify textured objects"""
        # Apply Gaussian blur for local averaging
        kernel_size = 5
        sigma = 1.0
        
        # Create Gaussian kernel
        x = torch.arange(kernel_size, dtype=torch.float32, device=image.device) - kernel_size // 2
        gaussian_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        gaussian_2d = gaussian_1d.view(-1, 1) * gaussian_1d.view(1, -1)
        gaussian_kernel = gaussian_2d.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)
        
        # Apply Gaussian smoothing
        if image.shape[1] == 3:
            smoothed = F.conv2d(image, gaussian_kernel, padding=kernel_size//2, groups=3)
        else:
            smoothed = F.conv2d(image, gaussian_kernel[:1], padding=kernel_size//2)
        
        # Compute local variance
        variance = torch.mean((image - smoothed) ** 2, dim=1, keepdim=True)
        
        # Apply sigmoid to normalize variance
        texture_strength = torch.sigmoid(variance * 20)  # Scale factor for sensitivity
        
        return texture_strength
    
    def detect_dark_objects_with_high_confidence(self, image):
        """High-confidence dark object detection using multiple cues"""
        
        # Step 1: Multi-threshold dark region detection
        dark_regions = self.detect_multi_threshold_dark_regions(image)
        
        # Step 2: Advanced edge strength computation
        edge_strength = self.compute_advanced_edge_strength(image)
        
        # Step 3: Texture variance computation
        texture_variance = self.compute_texture_variance(image)
        
        # Step 4: Combine all cues with learned weights
        # Dark objects should have: dark regions + strong edges + high texture variance
        object_confidence = (
            0.4 * dark_regions +           # Base: dark regions
            0.4 * edge_strength +          # Structural: strong boundaries
            0.2 * texture_variance         # Surface: textural details
        )
        
        # Step 5: Apply threshold to create binary object mask
        # Use adaptive threshold based on image statistics
        threshold = torch.mean(object_confidence) + 0.1  # Slightly above average
        threshold = torch.clamp(threshold, 0.3, 0.7)  # Keep in reasonable range
        
        object_mask = (object_confidence > threshold).float()
        
        # Step 6: Morphological operations to clean up the mask
        object_mask = self.morphological_closing(object_mask)
        
        return object_mask, object_confidence
    
    def morphological_closing(self, mask, kernel_size=5):
        """Apply morphological closing to clean up mask"""
        # Create circular kernel
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)
        kernel = kernel / kernel.sum()  # Normalize
        
        # Dilation followed by erosion (closing)
        dilated = F.conv2d(mask, kernel, padding=kernel_size//2)
        dilated = (dilated > 0.1).float()
        
        eroded = F.conv2d(dilated, kernel, padding=kernel_size//2)
        eroded = (eroded > 0.9).float()
        
        return eroded
    
    def compute_object_preservation_loss(self, output, input_image, target_image=None):
        """Compute advanced dark object preservation loss"""
        
        # Detect dark objects with high confidence
        object_mask, object_confidence = self.detect_dark_objects_with_high_confidence(input_image)
        
        # Ensure we have actual dark objects (non-zero mask)
        if torch.sum(object_mask) < 1e-6:
            # If no objects detected, return small positive loss to avoid zero loss issue
            return torch.tensor(0.01, device=output.device, requires_grad=True)
        
        if target_image is not None:
            # Supervised case: preserve dark objects as they appear in target
            
            # Focus on dark object regions
            input_objects = input_image * object_mask
            output_objects = output * object_mask
            target_objects = target_image * object_mask
            
            # Main preservation loss: output should match target in object regions
            main_preservation_loss = F.l1_loss(output_objects, target_objects)
            
            # Additional constraint: output should not be too different from input in object regions
            # This prevents over-brightening of dark objects
            input_preservation_loss = F.l1_loss(output_objects, input_objects) * 0.3
            
            # Confidence-weighted loss: higher confidence regions get more weight
            confidence_weighted_loss = F.l1_loss(output_objects * object_confidence, target_objects * object_confidence)
            
            total_loss = main_preservation_loss + input_preservation_loss + confidence_weighted_loss * 0.5
            
        else:
            # Unsupervised case: preserve dark objects as they appear in input
            
            input_objects = input_image * object_mask
            output_objects = output * object_mask
            
            # Main preservation loss: output should closely match input in object regions
            main_preservation_loss = F.l1_loss(output_objects, input_objects)
            
            # Color consistency loss: preserve color relationships in dark objects
            input_mean_color = torch.mean(input_objects.view(input_objects.shape[0], input_objects.shape[1], -1), dim=2, keepdim=True)
            output_mean_color = torch.mean(output_objects.view(output_objects.shape[0], output_objects.shape[1], -1), dim=2, keepdim=True)
            color_consistency_loss = F.l1_loss(output_mean_color, input_mean_color)
            
            total_loss = main_preservation_loss + color_consistency_loss * 0.2
        
        # Add regularization to ensure loss is not zero
        regularization = torch.mean(object_mask) * 0.001  # Small regularization term
        
        total_loss = total_loss + regularization
        
        return total_loss
    
    def forward(self, output, input_image, target_image=None, weight=1.0):
        """Main forward function"""
        preservation_loss = self.compute_object_preservation_loss(output, input_image, target_image)
        
        # Ensure loss is never exactly zero
        preservation_loss = torch.clamp(preservation_loss, min=1e-6)
        
        return weight * preservation_loss
    
class NonShadowPreservationLoss(nn.Module):
    """
    Ensures non-Degrade regions remain unchanged
    """
    def __init__(self):
        super(NonShadowPreservationLoss, self).__init__()
        
    def forward(self, output, input_img, threshold=0.3):
        # Detect non-Degrade regions (bright areas)
        input_gray = 0.299 * input_img[:, 0:1] + 0.587 * input_img[:, 1:2] + 0.114 * input_img[:, 2:3]
        non_shadow_mask = (input_gray > threshold).float()
        
        # Apply Gaussian blur to smooth mask boundaries
        kernel_size = 5
        sigma = 1.0
        x = torch.arange(kernel_size, dtype=torch.float32, device=input_img.device) - kernel_size // 2
        gaussian_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        gaussian_kernel = gaussian_1d.view(-1, 1) * gaussian_1d.view(1, -1)
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        
        non_shadow_mask = F.conv2d(non_shadow_mask, gaussian_kernel, padding=kernel_size//2)
        
        # Loss: output should match input in non-Degrade regions
        preservation_loss = F.l1_loss(output * non_shadow_mask, input_img * non_shadow_mask)
        
        return preservation_loss

# NEW: Improved Discriminator with Spectral Normalization to prevent mode collapse
class ImprovedDiscriminator(nn.Module):
    def __init__(self, input_channels=6):
        super(ImprovedDiscriminator, self).__init__()
        
        # Use spectral normalization for stability
        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(input_channels, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.utils.spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        return self.model(x)

def get_parser():
    parser = argparse.ArgumentParser(
        prog='Stable Pure VAE VDTA_IRNet with Latent Propagation',
        usage='python3 train.py',
        description='This module demonstrates removal using Pure VAE VDTA_IRNet with full latent propagation.',
        add_help=True)

    parser.add_argument('-e', '--epoch', type=int, default=10000, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('-l', '--load', type=str, default=None, help='the number of checkpoints')
    parser.add_argument('-hor', '--hold_out_ratio', type=float, default=0.993, help='training-validation ratio')
    parser.add_argument('-s', '--image_size', type=int, default=286)
    parser.add_argument('-cs', '--crop_size', type=int, default=256)
    parser.add_argument('-lr', '--lr', type=float, default=5e-5, help='Learning rate')
    
    # VAE specific parameters
    parser.add_argument('--beta_vae', type=float, default=0.0, help='Initial beta weight for VAE KL loss')
    parser.add_argument('--lambda_vae', type=float, default=0.001, help='Weight for total VAE loss')
    parser.add_argument('--beta_warmup', type=int, default=500, help='Number of epochs for beta warmup')
    parser.add_argument('--beta_max', type=float, default=0.00001, help='Maximum beta value after warmup')
    parser.add_argument('--pretrain_epochs', type=int, default=20, help='Epochs to pretrain with minimal VAE loss')
    
    # Stability parameters
    parser.add_argument('--grad_clip', type=float, default=0.5, help='Gradient clipping value')
    parser.add_argument('--kl_tolerance', type=float, default=0.5, help='KL tolerance for free bits')
    parser.add_argument('--max_kl_weight', type=float, default=1.0, help='Maximum weight for KL term')
    
    # ================ NEW: ENHANCED LOSS WEIGHTS ================
    parser.add_argument('--lambda_perceptual', type=float, default=0.8, help='Weight for perceptual loss (INCREASED)')
    parser.add_argument('--lambda_ssim', type=float, default=0.6, help='Weight for SSIM loss (INCREASED)')
    parser.add_argument('--lambda_shadow_specific', type=float, default=0.4, help='NEW: Weight for shadow-specific losses')
    parser.add_argument('--lambda_texture', type=float, default=0.3, help='NEW: Weight for texture preservation')
    parser.add_argument('--lambda_dark_object', type=float, default=0.5, help='NEW: Weight for dark object preservation')
    parser.add_argument('--lambda_gan', type=float, default=0.02, help='Weight for GAN loss (REDUCED)')
    parser.add_argument('--d_update_ratio', type=int, default=5, help='Update discriminator every N iterations (INCREASED)')

    return parser

def get_beta_schedule(epoch, warmup_epochs=500, beta_min=0.0, beta_max=0.00001, pretrain_epochs=20):
    """Very conservative beta scheduling for stable VAE training"""
    if epoch < pretrain_epochs:
        return 0.0
    elif epoch < pretrain_epochs + warmup_epochs:
        progress = (epoch - pretrain_epochs) / warmup_epochs
        progress = progress ** 2
        return beta_min + (beta_max - beta_min) * progress
    else:
        return beta_max

def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]
        new_state_dict[name] = v
    return new_state_dict

def set_requires_grad(nets, requires_grad=False):
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def unnormalize(x):
    x = x.transpose(1, 3)
    x = x * torch.Tensor((0.5, )) + torch.Tensor((0.5, ))
    x = x.transpose(1, 3)
    return x

def evaluate(G1, dataset, device, filename):
    # Evaluation function for enhanced model
    num_eval = min(9, len(dataset))
    img, gt_shadow, gt = zip(*[dataset[i] for i in range(num_eval)])
    img = torch.stack(img)
    gt_shadow = torch.stack(gt_shadow)
    gt = torch.stack(gt)
    print(f"Evaluation - GT shape: {gt.shape}")
    print(f"Evaluation - Img shape: {img.shape}")

    with torch.no_grad():
        if gt_shadow.dim() == 3:
            gt_shadow = gt_shadow.unsqueeze(1)
            
        try:
            latent_sf, shadow_decoder_out = G1.train_set(img.to(device))
            
            grid_rec = make_grid(unnormalize(latent_sf.to(torch.device('cpu'))), nrow=3)
            print(f"Grid shape: {grid_rec.shape}")
            latent_sf = latent_sf.to(torch.device('cpu'))
            shadow_decoder_out = shadow_decoder_out.to(torch.device('cpu'))
            
            grid_removal = make_grid(torch.cat((unnormalize(img), 
                                               unnormalize(gt),
                                               unnormalize(shadow_decoder_out),
                                               unnormalize(latent_sf)), dim=0), nrow=num_eval)
            
            save_image(grid_rec, filename + '_shadow_removal_img.jpg')
            save_image(grid_removal, filename + '_comparison.jpg')
            print(f"Evaluation images saved to {filename}")
            
        except Exception as e:
            print(f"Error during evaluation: {e}")

def plot_log(data, save_model_name='model'):
    # Enhanced plotting with new loss components
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # Main losses
    ax = axes[0, 0]
    if len(data['G']) > 0:
        ax.plot(data['G'], label='G_loss', linewidth=2)
        ax.plot(data['D'], label='D_loss', linewidth=2)
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.set_title('Generator and Discriminator Losses')
        ax.legend()
        ax.grid(True)
    
    # Reconstruction loss
    ax = axes[0, 1]
    if len(data['RECON']) > 0:
        ax.plot(data['RECON'], label='Reconstruction', color='green', linewidth=2)
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.set_title('Reconstruction Loss')
        ax.legend()
        ax.grid(True)
    
    # VAE losses
    ax = axes[0, 2]
    if len(data['VAE_KL']) > 0:
        ax.plot(data['VAE_KL'], label='KL (raw)', color='red', linewidth=2)
        ax.set_xlabel('epoch')
        ax.set_ylabel('KL divergence')
        ax.set_title('KL Divergence')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True)
    
    # Beta schedule
    ax = axes[1, 0]
    if len(data['BETA']) > 0:
        ax.plot(data['BETA'], label='Beta', color='purple', linewidth=2)
        ax.set_xlabel('epoch')
        ax.set_ylabel('beta value')
        ax.set_title('Beta Schedule')
        ax.legend()
        ax.grid(True)
    
    # Enhanced loss components
    ax = axes[1, 1]
    if 'PERCEPTUAL' in data and len(data['PERCEPTUAL']) > 0:
        ax.plot(data['PERCEPTUAL'], label='Perceptual', color='orange', linewidth=2)
        ax.plot(data['SSIM'], label='SSIM', color='blue', linewidth=2)
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.set_title('Perceptual and SSIM Losses')
        ax.legend()
        ax.grid(True)
    
    # ================ NEW: SHADOW-SPECIFIC LOSSES PLOT ================
    ax = axes[1, 2]
    if 'SHADOW_SPECIFIC' in data and len(data['SHADOW_SPECIFIC']) > 0:
        ax.plot(data['SHADOW_SPECIFIC'], label='Shadow Specific', color='brown', linewidth=2)
        ax.plot(data['TEXTURE'], label='Texture', color='pink', linewidth=2)
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.set_title('Shadow-Specific and Texture Losses')
        ax.legend()
        ax.grid(True)
    
    # ================ NEW: DARK OBJECT and NON_Degrade PRESERVATION PLOT ================
    ax = axes[2, 0]
    if 'DARK_OBJECT' in data and len(data['DARK_OBJECT']) > 0:
        ax.plot(data['DARK_OBJECT'], label='Dark Object', color='black', linewidth=2)
        if 'NON_SHADOW' in data and len(data['NON_SHADOW']) > 0:
            ax.plot(data['NON_SHADOW'], label='Non-Shadow Preserve', color='cyan', linewidth=2)
        ax.set_xlabel('epoch')  # MOVED: These should be at this indentation level
        ax.set_ylabel('loss')
        ax.set_title('Object Preservation Losses')
        ax.legend()
        ax.grid(True)
    
    # Learning rate
    ax = axes[2, 1]
    if len(data['LR']) > 0:
        ax.plot(data['LR'], label='Learning Rate', color='darkred', linewidth=2)
        ax.set_xlabel('epoch')
        ax.set_ylabel('lr')
        ax.set_title('Learning Rate')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True)
    
    # ================ NEW: TOTAL ENHANCED LOSS ================
    ax = axes[2, 2]
    if 'TOTAL_ENHANCED' in data and len(data['TOTAL_ENHANCED']) > 0:
        ax.plot(data['TOTAL_ENHANCED'], label='Total Enhanced Loss', color='darkgreen', linewidth=2)
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.set_title('Total Enhanced Loss')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('./logs/' + save_model_name + '_training_metrics.png', dpi=300)
    plt.close()

def check_dir():
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    if not os.path.exists('./result'):
        os.mkdir('./result')

def compute_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def train_model(G1, D1, dataloader, val_dataset, num_epochs, parser, save_model_name='model'):
    check_dir()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    #Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    G1.to(device)
    D1.to(device)
    
    print("device:{}".format(device))

    # Optimizer parameters
    lr = parser.lr
    beta1, beta2 = 0.5, 0.999
    lambda_vae = parser.lambda_vae
    grad_clip = parser.grad_clip
    kl_tolerance = parser.kl_tolerance

    # ================ NEW: INITIALIZE ENHANCED LOSS FUNCTIONS ================
    perceptual_loss = PerceptualLoss().to(device)
    ssim_loss = SSIMLoss().to(device)
    shadow_specific_loss = ShadowSpecificLoss().to(device)
    texture_preservation_loss = TexturePreservationLoss().to(device)
    advanced_dark_object_loss = AdvancedDarkObjectPreservationLoss(device=device).to(device)
    non_shadow_preservation_loss = NonShadowPreservationLoss().to(device)

    
    # Optimizers with enhanced settings
    optimizerG = torch.optim.Adam([{'params': G1.parameters()}], lr=lr, betas=(beta1, beta2), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizerG, 'min', factor=0.8, verbose=True, 
        threshold=0.001, min_lr=1e-7, patience=30  # Reduced patience for faster adaptation
    )
    # Reduced discriminator learning rate for stability
    optimizerD = torch.optim.Adam([{'params': D1.parameters()}], lr=lr*0.3, betas=(beta1, beta2))
   
    # Loss functions
    #criterionGAN = nn.BCEWithLogitsLoss().to(device)
    criterionL1 = nn.L1Loss().to(device)

    # ================ NEW: ENHANCED LOSS WEIGHTS ================
    lambda_dict = {
        'lambda_l1': 3.0,  # Reduced L1 weight to make room for new losses
        'lambda_perceptual': parser.lambda_perceptual,
        'lambda_ssim': parser.lambda_ssim,
        'lambda_shadow_specific': parser.lambda_shadow_specific,
        'lambda_texture': parser.lambda_texture,
        'lambda_dark_object': parser.lambda_dark_object,
        'lambda_gan': parser.lambda_gan
    }

    # Enhanced loss tracking
    g_losses = []
    d_losses = []
    recon_losses = []
    perceptual_losses = []
    ssim_losses = []
    shadow_specific_losses = []  # NEW
    texture_losses = []  # NEW
    dark_object_losses = []  # NEW
    non_shadow_losses = []  # NEW
    total_enhanced_losses = []  # NEW
    vae_kl_losses = []
    beta_values = []
    lr_values = []

    # Best model tracking with enhanced metrics
    best_combined_loss = float('inf')
    patience_counter = 0
    max_patience = 150  # Reduced for faster convergence
    
    # Discriminator update counter
    d_update_counter = 0

    # ================ ENHANCED TRAINING LOOP ================
    for epoch in range(num_epochs + 1):
        G1.train()
        D1.train()

        t_epoch_start = time.time()

        # Get current beta value
        current_beta = get_beta_schedule(
            epoch, 
            warmup_epochs=parser.beta_warmup,
            beta_min=parser.beta_vae,
            beta_max=parser.beta_max,
            pretrain_epochs=parser.pretrain_epochs
        )

        # Initialize epoch losses
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_perceptual_loss = 0.0
        epoch_ssim_loss = 0.0
        epoch_shadow_specific_loss = 0.0  # NEW
        epoch_texture_loss = 0.0  # NEW
        epoch_dark_object_loss = 0.0  # NEW
        epoch_non_shadow_loss = 0.0  # NEW - Non-Degrade preservation
        epoch_total_enhanced_loss = 0.0  # NEW
        epoch_vae_kl_loss = 0.0

        print('-----------')
        print('Epoch {}/{} | Beta: {:.10f} | LR: {:.10f}'.format(
            epoch, num_epochs, current_beta, optimizerG.param_groups[0]["lr"]))
        print('(train)')
        
        data_len = len(dataloader)
        
        # Enhanced batch loop with progress bar
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
        for n, (images, gt_shadow, gt) in enumerate(progress_bar):
            if images.size()[0] == 1:
                continue

            # Move data to device
            images = images.to(device)
            gt = gt.to(device)
            gt_shadow = gt_shadow.to(device)
            
            if gt_shadow.dim() == 3:
                gt_shadow = gt_shadow.unsqueeze(1)

            mini_batch_size = images.size()[0]
            
            # Update discriminator less frequently
            d_update_counter += 1
            should_update_d = (d_update_counter % parser.d_update_ratio == 0)

            # Train Discriminator with enhanced stability
            if epoch >= parser.pretrain_epochs // 2 and should_update_d:
                set_requires_grad([D1], True)
                optimizerD.zero_grad()

                with torch.no_grad():
                    latent_sf, latent_gt = G1(images, gt)

                # Enhanced discriminator training
                fake1 = torch.cat([images, latent_sf], dim=1)
                real1 = torch.cat([images, gt], dim=1)

                out_D1_fake = D1(fake1.detach())
                out_D1_real = D1(real1)

                # WGAN-GP style loss for better stability
                loss_D1_fake = torch.mean(out_D1_fake)
                loss_D1_real = -torch.mean(out_D1_real)
                
                # Gradient penalty
                alpha = torch.rand(mini_batch_size, 1, 1, 1).to(device)
                interpolated = alpha * real1 + (1 - alpha) * fake1.detach()
                interpolated.requires_grad_(True)
                out_interpolated = D1(interpolated)
                
                gradients = torch.autograd.grad(
                    outputs=out_interpolated,
                    inputs=interpolated,
                    grad_outputs=torch.ones_like(out_interpolated),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0]
                
                gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 5  # Reduced GP weight
                
                D_loss = loss_D1_fake + loss_D1_real + gradient_penalty
                D_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(D1.parameters(), grad_clip * 2)
                optimizerD.step()
                epoch_d_loss += D_loss.item()

            # ================ ENHANCED GENERATOR TRAINING ================
            set_requires_grad([D1], False)
            optimizerG.zero_grad()
            
            try:
                latent_sf, latent_gt = G1(images, gt)
                
                # Check for NaN
                if torch.isnan(latent_sf).any() or torch.isnan(latent_gt).any():
                    print(f"NaN detected in generator output at batch {n}")
                    continue
                
                # GAN loss
                if epoch >= parser.pretrain_epochs // 2:
                    fake1 = torch.cat([images, latent_sf], dim=1)
                    out_D1_fake = D1(fake1)
                    G_L_CGAN1 = -torch.mean(out_D1_fake)
                else:
                    G_L_CGAN1 = torch.tensor(0.0).to(device)

                # ================ ENHANCED RECONSTRUCTION LOSSES ================
                # Basic L1 losses
                G_L_l1 = criterionL1(latent_sf, gt)
                G_L_l1_gt = criterionL1(latent_gt, gt)
                
                # Perceptual loss for better visual quality
                G_L_perceptual = perceptual_loss(latent_sf, gt)
                
                # SSIM loss for structural similarity
                G_L_ssim = ssim_loss(latent_sf, gt)
                
                # ================ NEW: Degrade-SPECIFIC LOSSES ================
                G_L_brightness_consistency = shadow_specific_loss.compute_brightness_consistency_loss(
                    latent_sf, gt, images)
                G_L_color_consistency = shadow_specific_loss.compute_color_consistency_loss(
                    latent_sf, gt, images)
                G_L_gradient_preservation = shadow_specific_loss.compute_gradient_preservation_loss(
                    latent_sf, gt)
                
                G_L_shadow_specific = (G_L_brightness_consistency + 
                                     G_L_color_consistency + 
                                     G_L_gradient_preservation) / 3.0
                
                # ================ NEW: TEXTURE PRESERVATION LOSS ================
                G_L_texture = texture_preservation_loss(latent_sf, gt, weight=1.0)
                
                # ================ NEW: DARK OBJECT PRESERVATION LOSS ================
                G_L_dark_object = advanced_dark_object_loss(latent_sf, images, gt, weight=1.0)
                
                # Ensure dark object loss is not zero by adding small regularization if needed
                if G_L_dark_object.item() < 1e-8:
                    G_L_dark_object = G_L_dark_object + torch.tensor(0.001, device=device, requires_grad=True)
                    print("Warning: Dark object loss was near zero, added regularization")
                    
                # ================ NEW: NON-Degrade PRESERVATION LOSS ================
                G_L_non_shadow = non_shadow_preservation_loss(latent_sf, images)
                
                # In train.py, after computing G_L_non_shadow, add:
                if n % 100 == 0:  # Every 100 batches
                    with torch.no_grad():
                        shadow_mask = G1.shadow_mask_generator(images, latent_sf)
                        mask_coverage = shadow_mask.mean().item()
                        print(f"Shadow mask coverage: {mask_coverage:.3f}")
                
                # Track individual losses
                recon_loss = G_L_l1 + G_L_l1_gt
                epoch_recon_loss += recon_loss.item()
                epoch_perceptual_loss += G_L_perceptual.item()
                epoch_ssim_loss += G_L_ssim.item()
                epoch_shadow_specific_loss += G_L_shadow_specific.item()
                epoch_texture_loss += G_L_texture.item()
                epoch_dark_object_loss += G_L_dark_object.item()
                epoch_non_shadow_loss += G_L_non_shadow.item()

                # Compute VAE KL loss with free bits
                vae_kl_loss_raw = G1.compute_total_vae_loss()
                vae_kl_loss = torch.max(vae_kl_loss_raw - kl_tolerance, torch.tensor(0.0).to(device))
                vae_total_loss = current_beta * vae_kl_loss

                # ================ NEW: ENHANCED COMBINED LOSS ================
                if epoch >= parser.pretrain_epochs:
                    G_loss = (
                        lambda_dict["lambda_l1"] * recon_loss + 
                        lambda_dict["lambda_perceptual"] * G_L_perceptual +
                        lambda_dict["lambda_ssim"] * G_L_ssim +
                        lambda_dict["lambda_shadow_specific"] * G_L_shadow_specific +
                        lambda_dict["lambda_texture"] * G_L_texture +
                        lambda_dict["lambda_dark_object"] * G_L_dark_object +
                        1.0 * G_L_non_shadow +  # Strong weight for preservation
                        lambda_dict["lambda_gan"] * G_L_CGAN1 + 
                        lambda_vae * vae_total_loss
                    )
                else:
                    # During pretraining, focus on reconstruction and new losses
                    G_loss = (
                        lambda_dict["lambda_l1"] * recon_loss +
                        lambda_dict["lambda_perceptual"] * G_L_perceptual * 0.5 +
                        lambda_dict["lambda_ssim"] * G_L_ssim * 0.5 +
                        lambda_dict["lambda_shadow_specific"] * G_L_shadow_specific * 0.3 +
                        lambda_dict["lambda_texture"] * G_L_texture * 0.3 +
                        lambda_dict["lambda_dark_object"] * G_L_dark_object * 0.5 +
                        0.8 * G_L_non_shadow  # Still important during pretraining
                    )

                # Check for NaN in loss
                if torch.isnan(G_loss):
                    print(f"NaN loss detected at batch {n}")
                    continue

                G_loss.backward()
                
                # Enhanced gradient clipping
                torch.nn.utils.clip_grad_norm_(G1.parameters(), grad_clip)
                
                optimizerG.step()

                # Track losses
                epoch_g_loss += G_loss.item()
                epoch_total_enhanced_loss += G_loss.item()
                epoch_vae_kl_loss += vae_kl_loss_raw.item()
                epoch_non_shadow_loss += G_L_non_shadow.item()
                
                # Enhanced progress bar
                progress_bar.set_postfix({
                    'G': f'{G_loss.item():.4f}',
                    'R': f'{recon_loss.item():.4f}',
                    'P': f'{G_L_perceptual.item():.4f}',
                    'S': f'{G_L_ssim.item():.4f}',
                    'Sh': f'{G_L_shadow_specific.item():.4f}',
                    'T': f'{G_L_texture.item():.4f}',
                    'DO': f'{G_L_dark_object.item():.4f}',
                    'NS': f'{G_L_non_shadow.item():.4f}'  
                })
                
            except RuntimeError as e:
                print(f"Runtime error in batch {n}: {e}")
                continue

        # End of epoch calculations
        t_epoch_finish = time.time()
        
        # Calculate average losses
        if data_len > 0:
            avg_d_loss = epoch_d_loss / max(1, data_len // parser.d_update_ratio)
            avg_g_loss = epoch_g_loss / data_len
            avg_recon_loss = epoch_recon_loss / data_len
            avg_perceptual_loss = epoch_perceptual_loss / data_len
            avg_ssim_loss = epoch_ssim_loss / data_len
            avg_shadow_specific_loss = epoch_shadow_specific_loss / data_len
            avg_texture_loss = epoch_texture_loss / data_len
            avg_dark_object_loss = epoch_dark_object_loss / data_len
            avg_non_shadow_loss = epoch_non_shadow_loss / data_len 
            avg_total_enhanced_loss = epoch_total_enhanced_loss / data_len
            avg_vae_kl_loss = epoch_vae_kl_loss / data_len
        else:
            print("Warning: No valid batches in this epoch")
            continue
        
        print('-----------')
        print('Losses:')
        print('D:{:.4f} | G:{:.4f} | R:{:.4f} | P:{:.4f} | S:{:.4f}'.format(
            avg_d_loss, avg_g_loss, avg_recon_loss, avg_perceptual_loss, avg_ssim_loss))
        print('Sh:{:.4f} | T:{:.4f} | DO:{:.4f} | NS:{:.4f} | KL:{:.4f}'.format(
            avg_shadow_specific_loss, avg_texture_loss, avg_dark_object_loss, 
            avg_non_shadow_loss, avg_vae_kl_loss))  # Added ND
        print('Total Loss: {:.4f} | Time: {:.2f}s'.format(avg_total_enhanced_loss, t_epoch_finish - t_epoch_start))
        
        # NEW: Special monitoring for dark object loss
        if avg_dark_object_loss < 1e-6:
            print("WARNING: Dark Object Loss is extremely low! This might indicate the loss is not working properly.")
        elif avg_dark_object_loss > 0.001:
            print(f"INFO: Dark Object Loss is active: {avg_dark_object_loss:.6f}")

        # Store losses for plotting
        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)
        recon_losses.append(avg_recon_loss)
        perceptual_losses.append(avg_perceptual_loss)
        ssim_losses.append(avg_ssim_loss)
        shadow_specific_losses.append(avg_shadow_specific_loss)
        texture_losses.append(avg_texture_loss)
        dark_object_losses.append(avg_dark_object_loss)
        non_shadow_losses.append(avg_non_shadow_loss)  
        total_enhanced_losses.append(avg_total_enhanced_loss)
        vae_kl_losses.append(avg_vae_kl_loss)
        beta_values.append(current_beta)
        lr_values.append(optimizerG.param_groups[0]["lr"])
        
        # Enhanced learning rate scheduling based on combined metrics
        combined_metric = (avg_recon_loss + avg_perceptual_loss * 0.5 + 
                          avg_ssim_loss * 0.3 + avg_shadow_specific_loss * 0.2 + 
                          avg_texture_loss * 0.2 + avg_dark_object_loss * 0.3)
        scheduler.step(combined_metric)
        
        # Check for improvement with enhanced metrics
        enhanced_combined_loss = combined_metric
        if enhanced_combined_loss < best_combined_loss:
            best_combined_loss = enhanced_combined_loss
            patience_counter = 0
            # Save best model
            torch.save(G1.state_dict(), 'checkpoints/' + save_model_name + '_AISTD_best.pth')
            #torch.save(G1.state_dict(), 'checkpoints/' + save_model_name + '_SRD_best.pth')
            print(f"NEW BEST! model saved with combined loss: {best_combined_loss:.4f}")
        else:
            patience_counter += 1

        # Enhanced loss plotting
        plot_log({
            'G': g_losses, 'D': d_losses, 'RECON': recon_losses,
            'PERCEPTUAL': perceptual_losses, 'SSIM': ssim_losses,
            'SHADOW_SPECIFIC': shadow_specific_losses, 'TEXTURE': texture_losses,
            'DARK_OBJECT': dark_object_losses,'NON_SHADOW': non_shadow_losses, 'TOTAL_ENHANCED': total_enhanced_losses,
            'VAE_KL': vae_kl_losses, 'BETA': beta_values, 'LR': lr_values
        }, save_model_name)

        # Save checkpoints and evaluate
        if epoch % 10 == 0:
            torch.save(G1.state_dict(), 'checkpoints/' + save_model_name + '_AISTD_' + str(epoch) + '.pth')
            if epoch >= parser.pretrain_epochs // 2:
                torch.save(D1.state_dict(), 'checkpoints/' + save_model_name + '_AISTD_' + str(epoch) + '.pth')
                
        # Save checkpoints and evaluate
        #if epoch % 10 == 0:
            #torch.save(G1.state_dict(), 'checkpoints/' + save_model_name + '_SRD_' + str(epoch) + '.pth')
            #if epoch >= parser.pretrain_epochs // 2:
                #torch.save(D1.state_dict(), 'checkpoints/' + save_model_name + '_SRD_' + str(epoch) + '.pth')

            # Enhanced evaluation
            G1.eval()
            evaluate(G1, val_dataset, device, '{:s}/val_{:d}'.format('result', epoch))
        
        # Enhanced monitoring for instability
        if avg_vae_kl_loss > 1000 and epoch > parser.pretrain_epochs:
            print("WARNING: KL loss is very high - training might be unstable!")
            
        if avg_total_enhanced_loss < 0.1 and epoch > 50:
            print("INFO: loss is converging well!")
            
        # NEW: Special success indicator for dark object loss
        if avg_dark_object_loss > 0.01 and avg_dark_object_loss < 1.0:
            print("SUCCESS: Dark object loss is in healthy range!")
        
        # Early stopping
        if patience_counter > max_patience:
            print(f"Early stopping triggered after {epoch} epochs")
            print(f"Best combined loss: {best_combined_loss:.4f}")
            break

    return G1

def main(parser):
    # Initialize pure VAE model with latent propagation
    print("Initializing Stable Pure VAE VDTA_IRNet with Full Latent Propagation...")
    
    # Use latent dimensions
    latent_dims = [64, 128, 256, 512, 512, 512]
    G1 = VDTA_IRNet(input_channels=3, output_channels=3, latent_dims=latent_dims)
    D1 = ImprovedDiscriminator(input_channels=6)
    
    # Print enhanced model info
    total_params = sum(p.numel() for p in G1.parameters())
    trainable_params = sum(p.numel() for p in G1.parameters() if p.requires_grad)
    print(f"Total generator parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Load checkpoint if specified
    if parser.load is not None:
        print('Load checkpoint ' + parser.load)
        try:
            G1.load_state_dict(fix_model_state_dict(torch.load('./checkpoints/VDTA_IRNet_AISTD_' + parser.load + '.pth')))
            D1.load_state_dict(fix_model_state_dict(torch.load('./checkpoints/VDTA_IRNet_AISTD_D_' + parser.load + '.pth')))
            #G1.load_state_dict(fix_model_state_dict(torch.load('./checkpoints/VDTA_IRNet_SRD_' + parser.load + '.pth')))
            #D1.load_state_dict(fix_model_state_dict(torch.load('./checkpoints/VDTA_IRNet_SRD_D_' + parser.load + '.pth')))
            print("Enhanced checkpoint loaded successfully!")
        except:
            print("Could not load checkpoint, starting fresh")

    # Prepare datasets
    train_img_list, val_img_list = make_datapath_list(phase='train', rate=parser.hold_out_ratio)
    print(f"Training samples: {len(train_img_list['path_A'])}")
    print(f"Validation samples: {len(val_img_list['path_A'])}")
    
    # Data normalization
    mean = (0.5,)
    std = (0.5,)
    size = parser.image_size
    crop_size = parser.crop_size
    batch_size = parser.batch_size
    num_epochs = parser.epoch

    # Create enhanced datasets
    train_dataset = ImageDataset(
        img_list=train_img_list,
        img_transform=ImageTransform(size=size, crop_size=crop_size, mean=mean, std=std),
        phase='train'
    )
    val_dataset = ImageDataset(
        img_list=val_img_list,
        img_transform=ImageTransform(size=size, crop_size=crop_size, mean=mean, std=std),
        phase='test_no_crop'
    )

    # Create dataloader
    num_workers = 0 if platform.system() == 'Windows' else 2
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True if num_workers > 0 else False,
        drop_last=True
    )
    
    print(f"Using {num_workers} workers for data loading on {platform.system()}")
    
    # Print training configuration
    print("\nTraining Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {parser.lr}")
    print(f"  Beta warmup epochs: {parser.beta_warmup}")
    print(f"  Pretrain epochs: {parser.pretrain_epochs}")
    print(f"  Initial beta: {parser.beta_vae}")
    print(f"  Max beta: {parser.beta_max}")
    print(f"  Gradient clipping: {parser.grad_clip}")
    print(f"  KL tolerance: {parser.kl_tolerance}")
    print(f"  Lambda VAE: {parser.lambda_vae}")
    print(f"  Perceptual: {parser.lambda_perceptual}")
    print(f"  SSIM: {parser.lambda_ssim}")
    print(f"  Shadow-specific: {parser.lambda_shadow_specific}")
    print(f"  Texture: {parser.lambda_texture}")
    print(f"  Dark object: {parser.lambda_dark_object}")
    print(f"  GAN: {parser.lambda_gan}")

    # Train model
    print("\nStarting training of Pure VAE LSR_TowardsGIRNet with Latent Propagation...")
    print("This architecture propagates information only through latent codes!")
    print("Expect slower convergence but better disentanglement.\n")
    
    G1 = train_model(
        G1, D1, 
        dataloader=train_dataloader,
        val_dataset=val_dataset,
        num_epochs=num_epochs,
        parser=parser,
        save_model_name='VDTA_IRNet'
    )

    print("\nTraining completed!")
    print("Best model saved at: checkpoints/VDTA_IRNet_AISTD_best.pth")
    #print("Best model saved at: checkpoints/VDTA_IRNet_SRD_best.pth")

if __name__ == "__main__":
    # Required for Windows multiprocessing
    if platform.system() == 'Windows':
        torch.multiprocessing.set_start_method('spawn', force=True)
    
    parser = get_parser().parse_args()
    main(parser)