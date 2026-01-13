from utils.data_loader_denoise import make_datapath_list, DenoiseDataset, ImageTransform   
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

# Import loss functions from original training script
from torchvision.models import vgg19, VGG19_Weights

torch.manual_seed(44)
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

# ========== ALL LOSS CLASSES FROM ORIGINAL train.py ==========

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        self.vgg_layers = nn.ModuleList([
            nn.Sequential(*[vgg[x] for x in range(4)]),
            nn.Sequential(*[vgg[x] for x in range(4, 9)]),
            nn.Sequential(*[vgg[x] for x in range(9, 18)]),
            nn.Sequential(*[vgg[x] for x in range(18, 27)])
        ])
        
        for param in self.parameters():
            param.requires_grad = False
            
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


class ImprovedDiscriminator(nn.Module):
    def __init__(self, input_channels=6):
        super(ImprovedDiscriminator, self).__init__()
        
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
        prog='VDTA_IRNet for Denoising',
        usage='python3 train_denoise.py',
        description='Train VDTA_IRNet for image denoising on BSD400 dataset.',
        add_help=True)

    parser.add_argument('-e', '--epoch', type=int, default=10000, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('-l', '--load', type=str, default="None", help='checkpoint number to load')
    parser.add_argument('-hor', '--hold_out_ratio', type=float, default=0.975, help='training-validation ratio')
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
    
    # Loss weights (adapted for denoising - no shadow-specific losses)
    parser.add_argument('--lambda_perceptual', type=float, default=0.8, help='Weight for perceptual loss')
    parser.add_argument('--lambda_ssim', type=float, default=0.6, help='Weight for SSIM loss')
    parser.add_argument('--lambda_gan', type=float, default=0.02, help='Weight for GAN loss')
    parser.add_argument('--d_update_ratio', type=int, default=5, help='Update discriminator every N iterations')
    
    # Noise levels for training
    parser.add_argument('--noise_levels', nargs='+', type=int, default=[15, 25, 50], 
                        help='Noise levels to train on')

    return parser


def get_beta_schedule(epoch, warmup_epochs=500, beta_min=0.0, beta_max=0.00001, pretrain_epochs=20):
    """Beta scheduling for VAE training"""
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
    x = x * torch.Tensor((0.5,)) + torch.Tensor((0.5,))
    x = x.transpose(1, 3)
    return x


def evaluate(G1, dataset, device, filename):
    """Evaluation function for denoising"""
    num_eval = min(9, len(dataset))
    if num_eval == 0:
        return
        
    noisy, gt_dummy, clean = zip(*[dataset[i] for i in range(num_eval)])
    noisy = torch.stack(noisy)
    gt_dummy = torch.stack(gt_dummy)
    clean = torch.stack(clean)
    
    print(f"Evaluation - GT shape: {clean.shape}")
    print(f"Evaluation - Img shape: {noisy.shape}")

    with torch.no_grad():
        if gt_dummy.dim() == 3:
            gt_dummy = gt_dummy.unsqueeze(1)
        
        try:
            denoised = G1.train_set(noisy.to(device))
            
            denoised = denoised.to(torch.device('cpu'))
            
            # Create individual grid for denoised outputs
            grid_denoised = make_grid(unnormalize(denoised), nrow=3)
            print(f"Grid shape: {grid_denoised.shape}")
            
            # Save comparison: noisy | clean | denoised
            grid_comparison = make_grid(
                torch.cat((unnormalize(noisy), 
                          unnormalize(clean),
                          unnormalize(denoised)), dim=0), 
                nrow=num_eval
            )
            
            # Ensure filename directory exists
            result_dir = os.path.dirname(filename)
            if result_dir and not os.path.exists(result_dir):
                os.makedirs(result_dir)
            
            # Save both grids
            save_image(grid_denoised, filename + '_denoised.jpg')
            save_image(grid_comparison, filename + '_comparison.jpg')
            
        except Exception as e:
            print(f"✗ Error during evaluation: {e}")
            import traceback
            traceback.print_exc()


def plot_log(data, save_model_name='denoise_model'):
    """Plot training metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
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
    
    # VAE KL loss
    ax = axes[0, 2]
    if len(data['VAE_KL']) > 0:
        ax.plot(data['VAE_KL'], label='KL', color='red', linewidth=2)
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
    
    # Perceptual & SSIM losses
    ax = axes[1, 1]
    if 'PERCEPTUAL' in data and len(data['PERCEPTUAL']) > 0:
        ax.plot(data['PERCEPTUAL'], label='Perceptual', color='orange', linewidth=2)
        ax.plot(data['SSIM'], label='SSIM', color='blue', linewidth=2)
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.set_title('Perceptual and SSIM Losses')
        ax.legend()
        ax.grid(True)
    
    # Learning rate
    ax = axes[1, 2]
    if len(data['LR']) > 0:
        ax.plot(data['LR'], label='Learning Rate', color='darkred', linewidth=2)
        ax.set_xlabel('epoch')
        ax.set_ylabel('lr')
        ax.set_title('Learning Rate')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('./logs/' + save_model_name + '_training_metrics.png', dpi=300)
    plt.close()


def check_dir():
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    if not os.path.exists('./result'):
        os.makedirs('./result')
    #print("Created necessary directories: logs, checkpoints, result")


def train_model(G1, D1, dataloader, val_dataset, num_epochs, parser, save_model_name='denoise_model'):
    # Create directories first
    check_dir()
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    G1.to(device)
    D1.to(device)
    
    print("device:{}".format(device))
    print(f"Validation dataset size: {len(val_dataset)}")

    # Optimizer parameters
    lr = parser.lr
    beta1, beta2 = 0.5, 0.999
    lambda_vae = parser.lambda_vae
    grad_clip = parser.grad_clip
    kl_tolerance = parser.kl_tolerance

    # Initialize loss functions
    perceptual_loss = PerceptualLoss().to(device)
    ssim_loss = SSIMLoss().to(device)
    criterionL1 = nn.L1Loss().to(device)

    # Loss weights for denoising
    lambda_dict = {
        'lambda_l1': 3.0,
        'lambda_perceptual': parser.lambda_perceptual,
        'lambda_ssim': parser.lambda_ssim,
        'lambda_gan': parser.lambda_gan
    }

    # Optimizers
    optimizerG = torch.optim.Adam([{'params': G1.parameters()}], lr=lr, betas=(beta1, beta2), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizerG, 'min', factor=0.5, verbose=True,  # Changed: 0.8 → 0.5
        threshold=0.0001, min_lr=1e-7, patience=50    # Changed: 0.001 → 0.0001, 30 → 50
    )
    optimizerD = torch.optim.Adam([{'params': D1.parameters()}], lr=lr*0.3, betas=(beta1, beta2))

    # Loss tracking
    g_losses = []
    d_losses = []
    recon_losses = []
    perceptual_losses = []
    ssim_losses = []
    vae_kl_losses = []
    beta_values = []
    lr_values = []

    # Best model tracking
    best_combined_loss = float('inf')
    patience_counter = 0
    max_patience = 200
    
    # Discriminator update counter
    d_update_counter = 0

    # Training loop
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
        epoch_vae_kl_loss = 0.0

        print('-----------')
        print('Epoch {}/{} | Beta: {:.10f} | LR: {:.10f}'.format(
            epoch, num_epochs, current_beta, optimizerG.param_groups[0]["lr"]))
        print('(train)')
        
        data_len = len(dataloader)
        
        # Batch loop with progress bar
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
        for n, (noisy_images, gt_dummy, clean_images) in enumerate(progress_bar):
            if noisy_images.size()[0] == 1:
                continue

            # Move data to device
            noisy_images = noisy_images.to(device)
            clean_images = clean_images.to(device)
            gt_dummy = gt_dummy.to(device)
            
            if gt_dummy.dim() == 3:
                gt_dummy = gt_dummy.unsqueeze(1)

            mini_batch_size = noisy_images.size()[0]
            
            # Update discriminator less frequently
            d_update_counter += 1
            should_update_d = (d_update_counter % parser.d_update_ratio == 0)

            # Train Discriminator
            if epoch >= parser.pretrain_epochs // 2 and should_update_d:
                set_requires_grad([D1], True)
                optimizerD.zero_grad()

                with torch.no_grad():
                    denoised, latent_gt = G1(noisy_images, clean_images)

                fake1 = torch.cat([noisy_images, denoised], dim=1)
                real1 = torch.cat([noisy_images, clean_images], dim=1)

                out_D1_fake = D1(fake1.detach())
                out_D1_real = D1(real1)

                # WGAN-GP style loss
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
                
                gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 5
                
                D_loss = loss_D1_fake + loss_D1_real + gradient_penalty
                D_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(D1.parameters(), grad_clip * 2)
                optimizerD.step()
                epoch_d_loss += D_loss.item()

            # Train Generator
            set_requires_grad([D1], False)
            optimizerG.zero_grad()
            
            try:
                denoised, latent_gt = G1(noisy_images, clean_images)
                
                # Check for NaN
                if torch.isnan(denoised).any() or torch.isnan(latent_gt).any():
                    print(f"NaN detected in generator output at batch {n}")
                    continue
                
                # GAN loss
                if epoch >= parser.pretrain_epochs // 2:
                    fake1 = torch.cat([noisy_images, denoised], dim=1)
                    out_D1_fake = D1(fake1)
                    G_L_CGAN1 = -torch.mean(out_D1_fake)
                else:
                    G_L_CGAN1 = torch.tensor(0.0).to(device)

                # Reconstruction losses
                G_L_l1 = criterionL1(denoised, clean_images)
                G_L_l1_gt = criterionL1(latent_gt, clean_images)
                
                # Perceptual loss
                G_L_perceptual = perceptual_loss(denoised, clean_images)
                
                # SSIM loss
                G_L_ssim = ssim_loss(denoised, clean_images)
                
                # Track losses
                recon_loss = G_L_l1 + G_L_l1_gt
                epoch_recon_loss += recon_loss.item()
                epoch_perceptual_loss += G_L_perceptual.item()
                epoch_ssim_loss += G_L_ssim.item()

                # VAE KL loss
                vae_kl_loss_raw = G1.compute_total_vae_loss()
                vae_kl_loss = torch.max(vae_kl_loss_raw - kl_tolerance, torch.tensor(0.0).to(device))
                vae_total_loss = current_beta * vae_kl_loss

                # Combined loss
                if epoch >= parser.pretrain_epochs:
                    G_loss = (
                        lambda_dict["lambda_l1"] * recon_loss + 
                        lambda_dict["lambda_perceptual"] * G_L_perceptual +
                        lambda_dict["lambda_ssim"] * G_L_ssim +
                        lambda_dict["lambda_gan"] * G_L_CGAN1 + 
                        lambda_vae * vae_total_loss
                    )
                else:
                    # Pretraining
                    G_loss = (
                        lambda_dict["lambda_l1"] * recon_loss +
                        lambda_dict["lambda_perceptual"] * G_L_perceptual * 0.5 +
                        lambda_dict["lambda_ssim"] * G_L_ssim * 0.5
                    )

                # Check for NaN
                if torch.isnan(G_loss):
                    print(f"NaN loss detected at batch {n}")
                    continue

                G_loss.backward()
                torch.nn.utils.clip_grad_norm_(G1.parameters(), grad_clip)
                optimizerG.step()

                # Track losses
                epoch_g_loss += G_loss.item()
                epoch_vae_kl_loss += vae_kl_loss_raw.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'G': f'{G_loss.item():.4f}',
                    'R': f'{recon_loss.item():.4f}',
                    'P': f'{G_L_perceptual.item():.4f}',
                    'S': f'{G_L_ssim.item():.4f}'
                })
                
            except RuntimeError as e:
                print(f"Runtime error in batch {n}: {e}")
                continue

        # End of epoch
        t_epoch_finish = time.time()
        
        # Calculate average losses
        if data_len > 0:
            avg_d_loss = epoch_d_loss / max(1, data_len // parser.d_update_ratio)
            avg_g_loss = epoch_g_loss / data_len
            avg_recon_loss = epoch_recon_loss / data_len
            avg_perceptual_loss = epoch_perceptual_loss / data_len
            avg_ssim_loss = epoch_ssim_loss / data_len
            avg_vae_kl_loss = epoch_vae_kl_loss / data_len
        else:
            print("Warning: No valid batches in this epoch")
            continue
        
        print('-----------')
        print('Losses:')
        print('D:{:.4f} | G:{:.4f} | R:{:.4f} | P:{:.4f} | S:{:.4f} | KL:{:.4f}'.format(
            avg_d_loss, avg_g_loss, avg_recon_loss, avg_perceptual_loss, avg_ssim_loss, avg_vae_kl_loss))
        print('Time: {:.2f}s'.format(t_epoch_finish - t_epoch_start))

        # Store losses
        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)
        recon_losses.append(avg_recon_loss)
        perceptual_losses.append(avg_perceptual_loss)
        ssim_losses.append(avg_ssim_loss)
        vae_kl_losses.append(avg_vae_kl_loss)
        beta_values.append(current_beta)
        lr_values.append(optimizerG.param_groups[0]["lr"])
        
        # Learning rate scheduling
        combined_metric = avg_recon_loss + avg_perceptual_loss * 0.5 + avg_ssim_loss * 0.3
        scheduler.step(combined_metric)
        
        # Check for improvement
        if combined_metric < best_combined_loss:
            best_combined_loss = combined_metric
            patience_counter = 0
            torch.save(G1.state_dict(), 'checkpoints/' + save_model_name + 'VDTA_IRNet_Denoise_best.pth')
            print(f"NEW BEST! model saved with combined loss: {best_combined_loss:.4f}")
        else:
            patience_counter += 1

        # Plot losses
        plot_log({
            'G': g_losses, 'D': d_losses, 'RECON': recon_losses,
            'PERCEPTUAL': perceptual_losses, 'SSIM': ssim_losses,
            'VAE_KL': vae_kl_losses, 'BETA': beta_values, 'LR': lr_values
        }, save_model_name)

        # Save checkpoints and evaluate
        if epoch % 10 == 0:
            print(f"\n--- Saving checkpoint for epoch {epoch} ---")
            torch.save(G1.state_dict(), 'checkpoints/' + save_model_name + '_Denoise_' + str(epoch) + '.pth')
            if epoch >= parser.pretrain_epochs // 2:
                torch.save(D1.state_dict(), 'checkpoints/' + save_model_name + '_Denoise_' + str(epoch) + '.pth')

            print(f"Running evaluation for epoch {epoch}...")
            G1.eval()
            evaluate(G1, val_dataset, device, '{:s}/val_{:d}'.format('result', epoch))
            G1.train()  # Set back to training mode
        
        # Early stopping
        if patience_counter > max_patience:
            print(f"Early stopping triggered after {epoch} epochs")
            print(f"Best combined loss: {best_combined_loss:.4f}")
            break

    return G1


def main(parser):
    print("=" * 80)
    print("Initializing LSR_TowardsGIRNet for Image Denoising")
    print("=" * 80)
    
    # Initialize model
    latent_dims = [64, 128, 256, 512, 512, 512]
    G1 = VDTA_IRNet(input_channels=3, output_channels=3, latent_dims=latent_dims)
    D1 = ImprovedDiscriminator(input_channels=6)
    
    # Print model info
    total_params = sum(p.numel() for p in G1.parameters())
    trainable_params = sum(p.numel() for p in G1.parameters() if p.requires_grad)
    print(f"Total generator parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Load checkpoint if specified
    if parser.load is not None:
        print('Load checkpoint ' + parser.load)
        try:
            G1.load_state_dict(fix_model_state_dict(torch.load('./checkpoints/VDTA_IRNet_Denoise_' + parser.load + '.pth')))
            D1.load_state_dict(fix_model_state_dict(torch.load('./checkpoints/VDTA_IRNet_Denoise_D_' + parser.load + '.pth')))
            print("Checkpoint loaded successfully!")
        except:
            print("Could not load checkpoint, starting fresh")

    # Prepare datasets
    print("\nLoading BSD400 training dataset...")
    train_img_list, val_img_list = make_datapath_list(phase='train', rate=parser.hold_out_ratio)
    print(f"Training samples: {len(train_img_list['path_A'])}")
    print(f"Validation samples: {len(val_img_list['path_A'])}")
    
    # Data parameters
    mean = (0.5,)
    std = (0.5,)
    size = parser.image_size
    crop_size = parser.crop_size
    batch_size = parser.batch_size
    num_epochs = parser.epoch

    # Create datasets with noise augmentation
    train_dataset = DenoiseDataset(
        img_list=train_img_list,
        img_transform=ImageTransform(size=size, crop_size=crop_size, mean=mean, std=std),
        phase='train',
        noise_levels=parser.noise_levels
    )
    val_dataset = DenoiseDataset(
        img_list=val_img_list,
        img_transform=ImageTransform(size=size, crop_size=crop_size, mean=mean, std=std),
        phase='val',
        noise_levels=parser.noise_levels
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
    print("\n" + "=" * 80)
    print("Training Configuration:")
    print("=" * 80)
    print("  Task: Image Denoising")
    print("  Dataset: BSD400 (train), CBSD68 (test)")
    print(f"  Noise levels: {parser.noise_levels}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {parser.lr}")
    print(f"  Beta warmup epochs: {parser.beta_warmup}")
    print(f"  Pretrain epochs: {parser.pretrain_epochs}")
    print(f"  Perceptual weight: {parser.lambda_perceptual}")
    print(f"  SSIM weight: {parser.lambda_ssim}")
    print(f"  GAN weight: {parser.lambda_gan}")
    print("=" * 80)

    # Train model
    print("\nStarting training...")
    G1 = train_model(
        G1, D1, 
        dataloader=train_dataloader,
        val_dataset=val_dataset,
        num_epochs=num_epochs,
        parser=parser,
        save_model_name='VDTA_IRNet_Denoise'
    )

    print("\n" + "=" * 80)
    print("Training completed!")
    print("Best model saved at: checkpoints/VDTA_IRNet_Denoise_best.pth")
    print("=" * 80)


if __name__ == "__main__":
    if platform.system() == 'Windows':
        torch.multiprocessing.set_start_method('spawn', force=True)
    
    parser = get_parser().parse_args()
    main(parser)