import os
import glob
import torch
import torch.utils.data as data
from . import ISTD_transforms
from PIL import Image
import random
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

def add_gaussian_noise(image, noise_level):
    """
    Add Gaussian noise to a PIL image
    Args:
        image: PIL Image
        noise_level: noise standard deviation (15, 25, or 50)
    Returns:
        noisy_image: PIL Image with added noise
    """
    # Convert to numpy array
    img_array = np.array(image).astype(np.float32)
    
    # Add Gaussian noise
    noise = np.random.randn(*img_array.shape) * noise_level
    noisy_array = img_array + noise
    
    # Clip to valid range [0, 255]
    noisy_array = np.clip(noisy_array, 0, 255).astype(np.uint8)
    
    # Convert back to PIL Image
    noisy_image = Image.fromarray(noisy_array)
    
    return noisy_image


def make_datapath_list(phase="train", rate=0.8, noise_level=None):
    """
    Make filepath list for BSD400/CBSD68 denoising dataset
    
    Args:
        phase: "train" or "test"
        rate: training-validation split ratio (only for train phase)
        noise_level: None for train (will add noise on-the-fly), 
                     15/25/50 for test (will load pre-noised images)
    
    Returns:
        For train: (path_list, path_list_val) both with keys 'path_A', 'path_B', 'path_C'
        For test: path_list with keys 'path_A', 'path_B', 'path_C'
    """
    random.seed(44)
    
    if phase == 'train':
        # Training on BSD400 dataset
        rootpath = './BSDdataset/train/'
        
        # Get all image files (JPG format)
        files = glob.glob(os.path.join(rootpath, '*.jpg'))
        if len(files) == 0:
            files = glob.glob(os.path.join(rootpath, '*.JPG'))
        
        files.sort()
        random.shuffle(files)  # Shuffle for training
        
        print(f"Found {len(files)} training images in {rootpath}")
        
        path_A = []  # Will store paths to clean images (for adding noise during training)
        path_B = []  # No mask needed for denoising
        path_C = []  # Ground truth (clean images)
        
        for file_path in files:
            path_A.append(file_path)  # Input (will add noise on-the-fly)
            path_B.append(None)       # No mask for denoising
            path_C.append(file_path)  # GT is the same clean image
        
        # Split into train and validation
        num = len(path_A)
        split_idx = int(num * rate)
        
        path_A_train, path_A_val = path_A[:split_idx], path_A[split_idx:]
        path_B_train, path_B_val = [None] * len(path_A_train), [None] * len(path_A_val)
        path_C_train, path_C_val = path_C[:split_idx], path_C[split_idx:]
        
        path_list = {'path_A': path_A_train, 'path_B': path_B_train, 'path_C': path_C_train}
        path_list_val = {'path_A': path_A_val, 'path_B': path_B_val, 'path_C': path_C_val}
        
        print(f"Training split: {len(path_A_train)} images")
        print(f"Validation split: {len(path_A_val)} images")
        
        return path_list, path_list_val
    
    elif phase == 'test':
        # Testing on CBSD68 dataset
        if noise_level is None:
            raise ValueError("noise_level must be specified for test phase (15, 25, or 50)")
        
        # Paths for noisy and clean images
        noisy_path = f'./BSDdataset/CBSD68/noisy{noise_level}/'
        clean_path = './BSDdataset/CBSD68/original/'
        
        # Get noisy image files (PNG format)
        noisy_files = glob.glob(os.path.join(noisy_path, '*.png'))
        if len(noisy_files) == 0:
            noisy_files = glob.glob(os.path.join(noisy_path, '*.PNG'))
        
        noisy_files.sort()
        
        print(f"Found {len(noisy_files)} test images with noise level {noise_level}")
        
        path_A = []  # Noisy input images
        path_B = []  # No mask
        path_C = []  # Clean ground truth images
        
        for noisy_file in noisy_files:
            # Get corresponding clean image filename
            filename = os.path.basename(noisy_file)
            clean_file = os.path.join(clean_path, filename)
            
            if os.path.exists(clean_file):
                path_A.append(noisy_file)
                path_B.append(None)
                path_C.append(clean_file)
            else:
                print(f"Warning: Clean image not found for {filename}")
        
        path_list = {'path_A': path_A, 'path_B': path_B, 'path_C': path_C}
        
        return path_list


class ImageTransformOwn():
    """
    Preprocessing images for own images
    """
    def __init__(self, size=256, mean=(0.5,), std=(0.5,)):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.data_transform(img)


class ImageTransform():
    """
    Preprocessing images for denoising
    """
    def __init__(self, size=286, crop_size=256, mean=(0.5,), std=(0.5,)):
        self.data_transform = {
            'train': ISTD_transforms.Compose([
                ISTD_transforms.Scale(size=size),
                ISTD_transforms.RandomCrop(size=crop_size),
                ISTD_transforms.RandomHorizontalFlip(p=0.5),
                ISTD_transforms.RandomVerticalFlip(p=0.5),
                ISTD_transforms.ToTensor(),
                ISTD_transforms.Normalize(mean, std)
            ]),
            
            'val': ISTD_transforms.Compose([
                ISTD_transforms.Scale(size=size),
                ISTD_transforms.RandomCrop(size=crop_size),
                ISTD_transforms.ToTensor(),
                ISTD_transforms.Normalize(mean, std)
            ]),
            
            'test': ISTD_transforms.Compose([
                ISTD_transforms.Scale(size=size),
                ISTD_transforms.RandomCrop(size=crop_size),
                ISTD_transforms.ToTensor(),
                ISTD_transforms.Normalize(mean, std)
            ]),
            
            'test_no_crop': ISTD_transforms.Compose([
                ISTD_transforms.Resize([256, 256]),
                ISTD_transforms.ToTensor(),
                ISTD_transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, phase, img):
        return self.data_transform[phase](img)


class DenoiseDataset(data.Dataset):
    """
    Dataset class for BSD400/CBSD68 denoising
    
    For training: Loads clean images and adds noise on-the-fly
    For testing: Loads pre-noised images
    """
    def __init__(self, img_list, img_transform, phase, noise_levels=[15, 25, 50]):
        self.img_list = img_list
        self.img_transform = img_transform
        self.phase = phase
        self.noise_levels = noise_levels  # For training: randomly sample from these levels
        
    def __len__(self):
        return len(self.img_list['path_A'])
    
    def __getitem__(self, index):
        """
        Get tensor type preprocessed Image for denoising
        Returns: (noisy_img, dummy_mask, clean_img)
        """
        if self.phase == 'train' or self.phase == 'val':
            # Training/Validation: Load clean image and add noise
            clean_img = Image.open(self.img_list['path_A'][index]).convert('RGB')
            
            # Randomly select noise level for this sample
            noise_level = random.choice(self.noise_levels)
            
            # Add Gaussian noise
            noisy_img = add_gaussian_noise(clean_img, noise_level)
            
            # Ground truth is the clean image
            gt = clean_img
            
        else:
            # Testing: Load pre-noised image
            noisy_img = Image.open(self.img_list['path_A'][index]).convert('RGB')
            gt = Image.open(self.img_list['path_C'][index]).convert('RGB')
        
        # Create dummy mask for compatibility
        dummy_mask = Image.new('L', noisy_img.size, 0)
        
        # Apply transformations
        noisy_img, dummy_mask, gt = self.img_transform(self.phase, [noisy_img, dummy_mask, gt])
        
        return noisy_img, dummy_mask, gt


# Test the data loader
if __name__ == '__main__':
    print("Testing Denoising Data Loader...")
    print("=" * 60)
    
    # Test training data loading
    print("\n1. Testing Training Data Loading:")
    print("-" * 60)
    try:
        train_list, val_list = make_datapath_list(phase='train', rate=0.8)
        print("✓ Successfully loaded training data")
        print(f"  Training samples: {len(train_list['path_A'])}")
        print(f"  Validation samples: {len(val_list['path_A'])}")
        
        # Create dataset
        img_transforms = ImageTransform(size=286, crop_size=256, mean=(0.5,), std=(0.5,))
        train_dataset = DenoiseDataset(
            img_list=train_list,
            img_transform=img_transforms,
            phase='train',
            noise_levels=[15, 25, 50]
        )
        
        # Get a sample
        noisy, mask, clean = train_dataset[0]
        print(f"  Sample shapes: noisy={noisy.shape}, mask={mask.shape}, clean={clean.shape}")
        
    except Exception as e:
        print(f"✗ Error loading training data: {e}")
    
    # Test all noise levels for testing
    for noise_level in [15, 25, 50]:
        print(f"\n2. Testing Test Data Loading (Noise Level {noise_level}):")
        print("-" * 60)
        try:
            test_list = make_datapath_list(phase='test', noise_level=noise_level)
            print(f"✓ Successfully loaded test data for noise level {noise_level}")
            print(f"  Test samples: {len(test_list['path_A'])}")
            
            # Create dataset
            img_transforms = ImageTransform(size=286, crop_size=256, mean=(0.5,), std=(0.5,))
            test_dataset = DenoiseDataset(
                img_list=test_list,
                img_transform=img_transforms,
                phase='test_no_crop'
            )
            
            # Get a sample
            if len(test_dataset) > 0:
                noisy, mask, clean = test_dataset[0]
                print(f"  Sample shapes: noisy={noisy.shape}, mask={mask.shape}, clean={clean.shape}")
            
        except Exception as e:
            print(f"✗ Error loading test data: {e}")
    
    print("\n" + "=" * 60)
    print("Data loader testing complete!")
    print("\nIMPORTANT: Make sure your dataset structure is:")
    print("  ./BSDdataset/train/*.jpg (200 images)")
    print("  ./BSDdataset/CBSD68/noisy15/*.png (68 images)")
    print("  ./BSDdataset/CBSD68/noisy25/*.png (68 images)")
    print("  ./BSDdataset/CBSD68/noisy50/*.png (68 images)")
    print("  ./BSDdataset/CBSD68/original/*.png (68 images)")