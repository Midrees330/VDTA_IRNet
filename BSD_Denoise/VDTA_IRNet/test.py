import os
import time
import torch
import argparse
from tqdm import tqdm
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from collections import OrderedDict
from models.VDTA_IRNet import VDTA_IRNet
from utils.data_loader_denoise import make_datapath_list, DenoiseDataset, ImageTransform

torch.manual_seed(44)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def get_parser():
    parser = argparse.ArgumentParser(
        prog='VDTA_IRNet Denoising Test',
        usage='python test.py',
        description='Test VDTA_IRNet for image denoising.',
        add_help=True
    )
    parser.add_argument('-l', '--load', type=str, default='None', help='checkpoint to load (number or "best")')
    parser.add_argument('-o', '--out_path', type=str, default='./test_result_denoise', help='saving path')
    parser.add_argument('-s', '--image_size', type=int, default=286)
    parser.add_argument('-cs', '--crop_size', type=int, default=256)
    parser.add_argument('--noise_levels', nargs='+', type=int, default=[15, 25, 50],
                        help='Noise levels to test on')
    return parser


def fix_model_state_dict(state_dict):
    """Remove 'module.' prefix from DataParallel models"""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    return new_state_dict


def check_dir(out_path):
    """Ensure output directories exist"""
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for noise_level in [15, 25, 50]:
        noise_dir = os.path.join(out_path, f'noise{noise_level}')
        for subdir in ['denoised_images', 'comparison_grids']:
            os.makedirs(os.path.join(noise_dir, subdir), exist_ok=True)


def unnormalize(x):
    """Unnormalize tensor from [-1, 1] to [0, 1]"""
    x = x.transpose(1, 3)
    x = x * torch.Tensor((0.5,)) + torch.Tensor((0.5,))
    x = x.transpose(1, 3)
    return x


def test_single_noise_level(G1, test_dataset, device, noise_level, out_path):
    """Test model for a single noise level"""

    G1.eval()
    output_dir = os.path.join(out_path, f'noise{noise_level}')

    for n in tqdm(range(len(test_dataset)), desc=f"Noise {noise_level}"):
        noisy, gt_dummy, clean = test_dataset[n]

        # Safe filename extraction for all OS
        filename = os.path.basename(test_dataset.img_list['path_A'][n])
        if not filename.endswith('.png'):
            filename = filename.replace('.jpg', '.png').replace('.JPG', '.png')

        # Add batch dimension
        noisy = noisy.unsqueeze(0).to(device)
        gt_dummy = gt_dummy.unsqueeze(0)
        clean = clean.unsqueeze(0).to(device)

        with torch.no_grad():
            if gt_dummy.dim() == 3:
                gt_dummy = gt_dummy.unsqueeze(1)
            denoised = G1.test_set(noisy)
            denoised = denoised.cpu()

        # Unnormalize for saving
        noisy_un = unnormalize(noisy.cpu())
        clean_un = unnormalize(clean.cpu())
        denoised_un = unnormalize(denoised)

        # Create horizontal (side-by-side) comparison: Noisy | Clean | Denoised
        comparison = torch.cat([noisy_un, clean_un, denoised_un], dim=0)
        grid = make_grid(comparison, nrow=3, padding=5, normalize=False, pad_value=1.0)

        # Save paths
        grid_path = os.path.join(output_dir, 'comparison_grids', filename)
        denoised_path = os.path.join(output_dir, 'denoised_images', filename)
        os.makedirs(os.path.dirname(grid_path), exist_ok=True)
        os.makedirs(os.path.dirname(denoised_path), exist_ok=True)

        # Save images
        save_image(grid, grid_path)
        denoised_image = transforms.ToPILImage()(denoised_un[0])
        denoised_image.save(denoised_path)


def main(parser):
    print("LSR_TowardsGIRNet - Image Denoising Testing")

    # Initialize model
    latent_dims = [64, 128, 256, 512, 512, 512]
    G1 = VDTA_IRNet(input_channels=3, output_channels=3, latent_dims=latent_dims)

    # Load checkpoint
    checkpoint_path = f'./checkpoints/VDTA_IRNet_Denoise_{parser.load}.pth'
    print(f"\nLoading checkpoint: {checkpoint_path}")
    try:
        G1_weights = torch.load(checkpoint_path)
        G1.load_state_dict(fix_model_state_dict(G1_weights))
    except FileNotFoundError:
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G1.to(device)
    print(f"Using device: {device}")

    # Create output directories
    check_dir(parser.out_path)

    mean = (0.5,)
    std = (0.5,)
    size = parser.image_size
    crop_size = parser.crop_size

    all_start_time = time.time()

    # Test across all noise levels
    for noise_level in parser.noise_levels:
        print(f"Noise Level: {noise_level}")
        try:
            test_img_list = make_datapath_list(phase='test', noise_level=noise_level)
            test_dataset = DenoiseDataset(
                img_list=test_img_list,
                img_transform=ImageTransform(size=size, crop_size=crop_size, mean=mean, std=std),
                phase='test_no_crop'
            )
            print(f"Found {len(test_dataset)} test images")

            test_single_noise_level(G1, test_dataset, device, noise_level, parser.out_path)

        except Exception as e:
            print(f" Error testing noise level {noise_level}: {e}")
            continue

    all_end_time = time.time()
    print(f"Total testing time: {all_end_time - all_start_time:.2f} seconds")


if __name__ == "__main__":
    parser = get_parser().parse_args()
    main(parser)
