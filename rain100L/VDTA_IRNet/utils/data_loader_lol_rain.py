import os
import glob
import torch
import torch.utils.data as data
from . import ISTD_transforms
from PIL import Image
import random
from torchvision import transforms
import matplotlib.pyplot as plt

def make_datapath_list(phase="train", rate=0.8):
    """
    make filepath list for LOL/rain100L/dataset dataset - SAME FUNCTION SIGNATURE AS SHADOW REMOVAL
    LOL/rain100L structure: ./lol/train/train_A (low-light input) and ./lol/train/train_B (ground truth)
    """
    random.seed(44)
    
    #rootpath = './dataset/' + phase + '/'
    #rootpath = './lol/' + phase + '/'
    rootpath = './rain100L/' + phase + '/'
    files_name = os.listdir(rootpath + phase + '_A')

    if phase=='train':
        random.shuffle(files_name)
    elif phase=='test':
        files_name.sort()

    path_A = []
    path_B = []
    path_C = []

    for name in files_name:
        path_A.append(rootpath + phase + '_A/'+name)  # Low-light input images
        path_B.append(None)  # No mask file needed for LOL - set to None
        path_C.append(rootpath + phase + '_B/'+name)  # Ground truth images in B folder

    num = len(path_A)

    if phase=='train':
        path_A, path_A_val = path_A[:int(num*rate)], path_A[int(num*rate):]
        path_B, path_B_val = [None]*len(path_A), [None]*len(path_A_val)  # All None for masks
        path_C, path_C_val = path_C[:int(num*rate)], path_C[int(num*rate):]
        
        path_list = {'path_A': path_A, 'path_B': path_B, 'path_C': path_C}
        path_list_val = {'path_A': path_A_val, 'path_B': path_B_val, 'path_C': path_C_val}
        return path_list, path_list_val

    elif phase=='test':
        path_list = {'path_A': path_A, 'path_B': [None]*len(path_A), 'path_C': path_C}
        return path_list

class ImageTransformOwn():
    """
    preprocessing images for own images
    """
    def __init__(self, size=256, mean=(0.5, ), std=(0.5, )):
        self.data_transform = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize(mean, std)])

    def __call__(self, img):
        return self.data_transform(img)


class ImageTransform():
    """
    preprocessing images - SAME CLASS AS SHADOW REMOVAL
    """
    def __init__(self, size=286, crop_size=256, mean=(0.5, ), std=(0.5, )):
        self.data_transform = {'train': ISTD_transforms.Compose([ISTD_transforms.Scale(size=size),
                                                            ISTD_transforms.RandomCrop(size=crop_size),
                                                            ISTD_transforms.RandomHorizontalFlip(p=0.5),
                                                            ISTD_transforms.RandomVerticalFlip(p=0.5),
                                                            ISTD_transforms.ToTensor(),
                                                            ISTD_transforms.Normalize(mean, std)]),

                                'val': ISTD_transforms.Compose([ISTD_transforms.Scale(size=size),
                                                           ISTD_transforms.RandomCrop(size=crop_size),
                                                           ISTD_transforms.ToTensor(),
                                                           ISTD_transforms.Normalize(mean, std)]),

                                'test': ISTD_transforms.Compose([ISTD_transforms.Scale(size=size),
                                                            ISTD_transforms.RandomCrop(size=crop_size),
                                                            ISTD_transforms.ToTensor(),
                                                            ISTD_transforms.Normalize(mean, std)]),
                                'test_no_crop': ISTD_transforms.Compose([ISTD_transforms.Resize([256,256]),
                                                            ISTD_transforms.ToTensor(),
                                                            ISTD_transforms.Normalize(mean, std)])}

    def __call__(self, phase, img):
        return self.data_transform[phase](img)


class ImageDataset(data.Dataset):
    """
    Dataset class for LOL/rain100L - SAME CLASS SIGNATURE AS SHADOW REMOVAL
    """
    def __init__(self, img_list, img_transform, phase):
        self.img_list = img_list
        self.img_transform = img_transform
        self.phase = phase

    def __len__(self):
        return len(self.img_list['path_A'])

    def __getitem__(self, index):
        '''
        get tensor type preprocessed Image for LOL/rain100L dataset
        SAME RETURN FORMAT AS SHADOW REMOVAL: (img, gt_shadow, gt)
        '''
        img = Image.open(self.img_list['path_A'][index]).convert('RGB')  # Low-light input
        gt = Image.open(self.img_list['path_C'][index]).convert('RGB')   # Ground truth from C path
        
        # Create dummy mask for compatibility with existing training code
        gt_shadow = Image.new('L', img.size, 0)  # Black dummy mask
        
        img, gt_shadow, gt = self.img_transform(self.phase, [img, gt_shadow, gt])

        return img, gt_shadow, gt

if __name__ == '__main__':
    
    # Test AISTD dataset loading
    #img = Image.open('../dataset/train/train_A/test.png').convert('RGB')
    #gt = Image.open('../dataset/train/train_B/test.png').convert('RGB')
    
    # Test rain dataset loading
    img = Image.open('../rain100L/train/train_A/test.png').convert('RGB')
    gt = Image.open('../rain100L/train/train_B/test.png').convert('RGB')
    
    # Test LOL dataset loading
    #img = Image.open('../lol/train/train_A/test.png').convert('RGB')
    #gt = Image.open('../lol/train/train_B/test.png').convert('RGB')



    print(f"LOL/rain100L/dataset Input image size: {img.size}")
    print(f"LOL/rain100L/dataset Ground truth size: {gt.size}")

    f = plt.figure(figsize=(10, 4))
    f.add_subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Low-light Input')
    f.add_subplot(1, 2, 2)
    plt.imshow(gt)
    plt.title('Ground Truth')

    img_transforms = ImageTransform(size=286, crop_size=256, mean=(0.5, ), std=(0.5, ))
    
    # Create dummy mask
    gt_shadow = Image.new('L', img.size, 0)
    img, gt_shadow, gt = img_transforms('train', [img, gt_shadow, gt])

    print("Processed shapes:")
    print(f"img: {img.shape}")
    print(f"gt_shadow (dummy): {gt_shadow.shape}")
    print(f"gt: {gt.shape}")

    plt.tight_layout()
    plt.show()