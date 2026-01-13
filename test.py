from utils.data_loader import make_datapath_list, ImageDataset, ImageTransform
from models.VDTA_IRNet import VDTA_IRNet
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torchvision import transforms
from collections import OrderedDict
from PIL import Image
import argparse
import time
import torch
import os
from tqdm import tqdm

torch.manual_seed(44)
# choose your device
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def get_parser():
    parser = argparse.ArgumentParser(
        prog='Pure VAE VDTA_IRNet Test',
        usage='python test.py',
        description='This module tests removal using Pure VDTA_IRNet.',
        add_help=True)

    parser.add_argument('-l', '--load', type=str, default=None, help='checkpoint to load (number or "best")')
    parser.add_argument('-i', '--image_path', type=str, default=None, help='file path of image you want to test')
    parser.add_argument('-o', '--out_path', type=str, default='./test_result', help='saving path')
    parser.add_argument('-s', '--image_size', type=int, default=286)
    parser.add_argument('-cs', '--crop_size', type=int, default=256)
    parser.add_argument('-rs', '--resized_size', type=int, default=256)

    return parser

def fix_model_state_dict(state_dict):

    # remove 'module.' of dataparallel

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]
        new_state_dict[name] = v
    return new_state_dict

def check_dir():
    if not os.path.exists('./test_result'):
        os.mkdir('./test_result')
    if not os.path.exists('./test_result/shadow_removal_images'):
        os.mkdir('./test_result/shadow_removal_images')
    if not os.path.exists('./test_result/shadow_removal_grid_images'):
        os.mkdir('./test_result/shadow_removal_grid_images')

def unnormalize(x):
    x = x.transpose(1, 3)
    x = x * torch.Tensor((0.5, )) + torch.Tensor((0.5, ))
    x = x.transpose(1, 3)
    return x

def test(G1, test_dataset):

    # Test on datasets

    check_dir()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G1.to(device)
    G1.eval()
    
    print(f"Testing {len(test_dataset)} images...")
    
    # Use tqdm for progress bar
    for n in tqdm(range(len(test_dataset)), desc="Testing"):
        img, gt_shadow, gt = test_dataset[n]
        
        # Get filename
        filename = test_dataset.img_list['path_A'][n].split('/')[-1]
        
        # Add batch dimension
        img = torch.unsqueeze(img, dim=0)
        gt_shadow = torch.unsqueeze(gt_shadow, dim=0)
        gt = torch.unsqueeze(gt, dim=0)

        with torch.no_grad():
            # if - gt_shadow has proper dimensions
            if gt_shadow.dim() == 3:
                gt_shadow = gt_shadow.unsqueeze(1)
                
            # Concatenate image with mask
            #img_mask = torch.cat([img, gt_shadow], dim=1)
            
            # Get degrade-free output
            latent_sf_out = G1.test_set(img.to(device))
            latent_sf_out = latent_sf_out.to(torch.device("cpu"))

        # Create comparison grid: input | ground truth | output
        grid = make_grid(torch.cat([unnormalize(img), 
                                    unnormalize(gt), 
                                    unnormalize(latent_sf_out)], dim=0))
        
        # Save grid image
        save_image(grid, f'./test_result/shadow_removal_grid_images/{filename}')

        # Save individual degrade-free image
        shadow_removal_image = transforms.ToPILImage(mode='RGB')(unnormalize(latent_sf_out)[0, :, :, :])
        shadow_removal_image.save(f'./test_result/shadow_removal_images/{filename}')
    
    print("Testing complete! Results saved in ./test_result/")

def main(parser):
    # Use the same latent dimensions as training
    latent_dims = [64, 128, 256, 512, 512, 512]
    G1 = VDTA_IRNet(input_channels=3, output_channels=3, latent_dims=latent_dims)

    # Load checkpoint
    if parser.load is not None:
        checkpoint_path = f'./checkpoints/VDTA_IRNet_AISTD_{parser.load}.pth'
        #checkpoint_path = f'./checkpoints/VDTA_IRNet_SRD_{parser.load}.pth'
        print(f'Loading checkpoint: {checkpoint_path}')
        
        try:
            G1_weights = torch.load(checkpoint_path)
            G1.load_state_dict(fix_model_state_dict(G1_weights))
            print("Model loaded successfully!")
        except FileNotFoundError:
            print(f"Error: Checkpoint not found at {checkpoint_path}")
            return

    # Data normalization parameters
    mean = (0.5,)
    std = (0.5,)
    size = parser.image_size
    crop_size = parser.crop_size

    # Load test dataset
    print('Loading test dataset...')
    test_img_list = make_datapath_list(phase='test')
    test_dataset = ImageDataset(
        img_list=test_img_list,
        img_transform=ImageTransform(size=size, crop_size=crop_size, mean=mean, std=std),
        phase='test_no_crop'
    )
    
    print(f"Found {len(test_dataset)} test images")
    
    # Start testing
    start_time = time.time()
    test(G1, test_dataset)
    end_time = time.time()
    
    print(f"Total testing time: {end_time - start_time:.2f} seconds")
    print(f"Average time per image: {(end_time - start_time) / len(test_dataset):.2f} seconds")

if __name__ == "__main__":
    parser = get_parser().parse_args()
    main(parser)