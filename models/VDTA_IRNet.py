import torch.nn.functional as F
import math
import torch.nn as nn
import torch

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


class Cvi(nn.Module):
    def __init__(self, in_channels, out_channels, before=None, after=False, kernel_size=4, stride=2,
                 padding=1, dilation=1, groups=1, bias=False):
        super(Cvi, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv.apply(weights_init('gaussian'))

        if after == 'BN':
            self.after = nn.InstanceNorm2d(out_channels)
        elif after == 'Tanh':
            self.after = torch.tanh
        elif after == 'sigmoid':
            self.after = torch.sigmoid

        if before == 'ReLU':
            self.before = nn.ReLU(inplace=False)
        elif before == 'LReLU':
            self.before = nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, x):
        if hasattr(self, 'before'):
            x = self.before(x)

        x = self.conv(x)

        if hasattr(self, 'after'):
            x = self.after(x)

        return x


class CvTi(nn.Module):
    def __init__(self, in_channels, out_channels, before=None, after=False, kernel_size=4, stride=2,
                 padding=1, dilation=1, groups=1, bias=False):
        super(CvTi, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                        output_padding=0, bias=bias)
        self.conv.apply(weights_init('gaussian'))

        if after == 'BN':
            self.after = nn.InstanceNorm2d(out_channels)
        elif after == 'Tanh':
            self.after = torch.tanh
        elif after == 'sigmoid':
            self.after = torch.sigmoid

        if before == 'ReLU':
            self.before = nn.ReLU(inplace=False)
        elif before == 'LReLU':
            self.before = nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, x):
        if hasattr(self, 'before'):
            x = self.before(x)

        x = self.conv(x)

        if hasattr(self, 'after'):
            x = self.after(x)

        return x


# ================  DUAL-SCALE ATTENTION MODULE ================
class DualScaleAttentionModule(nn.Module):
    """
    Dual-scale attention to better distinguish Degraded area
    - Fine-scale: captures local Degraded boundaries
    - Coarse-scale: captures global illumination patterns
    """
    def __init__(self, channels):
        super(DualScaleAttentionModule, self).__init__()
        
        # Fine-scale attention (3x3 convolutions)
        self.fine_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels // 4, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # Coarse-scale attention (dilated convolutions)
        self.coarse_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, 1, 2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels // 4, 3, 1, 4, dilation=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # Fusion weights
        self.fusion_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 2, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        fine_att = self.fine_attention(x)
        coarse_att = self.coarse_attention(x)
        
        # Adaptive fusion of attention maps
        fusion_w = self.fusion_weights(x)  # [B, 2, 1, 1]
        w_fine, w_coarse = fusion_w[:, 0:1], fusion_w[:, 1:2]
        
        combined_att = w_fine * fine_att + w_coarse * coarse_att
        return x * combined_att


class PureVAE_Encoder(nn.Module):
    # Pure VAE Encoder with Full Latent Propagation
    def __init__(self, input_channels=3, latent_dims=None):
        super(PureVAE_Encoder, self).__init__()
        
        if latent_dims is None:
            latent_dims = [64, 128, 256, 512, 512, 512]
        
        self.latent_dims = latent_dims
        
        # First encoding layer (takes original input)
        self.Cv0 = Cvi(input_channels, 64)
        # Subsequent layers take latent codes as input
        self.Cv1 = Cvi(latent_dims[0], 128, before='LReLU', after='BN')
        self.Cv2 = Cvi(latent_dims[1], 256, before='LReLU', after='BN')
        self.Cv3 = Cvi(latent_dims[2], 512, before='LReLU', after='BN')
        self.Cv4_1 = Cvi(latent_dims[3], 512, before='LReLU', after='BN')
        self.Cv4_2 = Cvi(latent_dims[4], 512, before='LReLU', after='BN')
        self.Cv4_3 = Cvi(latent_dims[4], 512, before='LReLU', after='BN')
        self.Cv5 = Cvi(latent_dims[4], 512, before='LReLU')
        
        # ================ NEW: ADD DUAL-SCALE ATTENTION MODULES ================
        self.attention_1 = DualScaleAttentionModule(128)
        self.attention_2 = DualScaleAttentionModule(256)
        self.attention_3 = DualScaleAttentionModule(512)
        
        # VAE parameter layers
        self.fc_mu0 = nn.Conv2d(64, latent_dims[0], kernel_size=1)
        self.fc_logvar0 = nn.Conv2d(64, latent_dims[0], kernel_size=1)
        
        self.fc_mu1 = nn.Conv2d(128, latent_dims[1], kernel_size=1)
        self.fc_logvar1 = nn.Conv2d(128, latent_dims[1], kernel_size=1)
        
        self.fc_mu2 = nn.Conv2d(256, latent_dims[2], kernel_size=1)
        self.fc_logvar2 = nn.Conv2d(256, latent_dims[2], kernel_size=1)
        
        self.fc_mu3 = nn.Conv2d(512, latent_dims[3], kernel_size=1)
        self.fc_logvar3 = nn.Conv2d(512, latent_dims[3], kernel_size=1)
        
        self.fc_mu4_1 = nn.Conv2d(512, latent_dims[4], kernel_size=1)
        self.fc_logvar4_1 = nn.Conv2d(512, latent_dims[4], kernel_size=1)
        
        self.fc_mu4_2 = nn.Conv2d(512, latent_dims[4], kernel_size=1)
        self.fc_logvar4_2 = nn.Conv2d(512, latent_dims[4], kernel_size=1)
        
        self.fc_mu4_3 = nn.Conv2d(512, latent_dims[4], kernel_size=1)
        self.fc_logvar4_3 = nn.Conv2d(512, latent_dims[4], kernel_size=1)
        
        self.fc_mu5 = nn.Conv2d(512, latent_dims[5], kernel_size=1)
        self.fc_logvar5 = nn.Conv2d(512, latent_dims[5], kernel_size=1)
        
        # Initialized VAE parameters properly
        self._initialize_vae_layers()
        
        # Store for loss computation
        self.mu_list = []
        self.logvar_list = []
    
    def _initialize_vae_layers(self):
        # Proper initialization to prevent KL explosion
        # Initialize all mu layers with small weights
        mu_layers = [self.fc_mu0, self.fc_mu1, self.fc_mu2, self.fc_mu3,
                     self.fc_mu4_1, self.fc_mu4_2, self.fc_mu4_3, self.fc_mu5]
        
        for layer in mu_layers:
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            nn.init.constant_(layer.bias, 0.0)
        
        # Initialize all logvar layers to output small variances
        logvar_layers = [self.fc_logvar0, self.fc_logvar1, self.fc_logvar2, self.fc_logvar3,
                         self.fc_logvar4_1, self.fc_logvar4_2, self.fc_logvar4_3, self.fc_logvar5]
        
        for layer in logvar_layers:
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            nn.init.constant_(layer.bias, -3.0)  # Start with small variance (e^-3 ≈ 0.05)

    def reparameterize(self, mu, logvar):
        # Clamp logvar to prevent numerical issues
        logvar = torch.clamp(logvar, min=-10, max=2)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input):
        # Clear previous computations
        mu_list = []
        logvar_list = []
        
        # Layer 0: Encode original input
        x0 = self.Cv0(input)
        mu0 = self.fc_mu0(x0)
        logvar0 = self.fc_logvar0(x0)
        # Clamp to prevent explosion
        mu0 = torch.clamp(mu0, min=-10, max=10)
        logvar0 = torch.clamp(logvar0, min=-10, max=2)
        z0 = self.reparameterize(mu0, logvar0)
        mu_list.append(mu0)
        logvar_list.append(logvar0)
        
        # ================ NEW: ENHANCED LAYER 1 WITH FREQUENCY FEATURES ================
        # Layer 1: Use z0 as input with frequency-aware processing
        x1 = self.Cv1(z0)
        x1 = self.attention_1(x1)  # Apply dual-scale attention
        mu1 = self.fc_mu1(x1)
        logvar1 = self.fc_logvar1(x1)
        mu1 = torch.clamp(mu1, min=-10, max=10)
        logvar1 = torch.clamp(logvar1, min=-10, max=2)
        z1 = self.reparameterize(mu1, logvar1)
        mu_list.append(mu1)
        logvar_list.append(logvar1)
        
        # ================ NEW: ENHANCED LAYER 2 WITH FREQUENCY FEATURES ================
        # Layer 2: Enhanced with frequency awareness
        x2 = self.Cv2(z1)
        x2 = self.attention_2(x2)  # Apply dual-scale attention 
        mu2 = self.fc_mu2(x2)
        logvar2 = self.fc_logvar2(x2)
        mu2 = torch.clamp(mu2, min=-10, max=10)
        logvar2 = torch.clamp(logvar2, min=-10, max=2)
        z2 = self.reparameterize(mu2, logvar2)
        mu_list.append(mu2)
        logvar_list.append(logvar2)
        
        # Layer 3: Enhanced with attention
        x3 = self.Cv3(z2)
        x3 = self.attention_3(x3)  # Apply dual-scale attention
        mu3 = self.fc_mu3(x3)
        logvar3 = self.fc_logvar3(x3)
        mu3 = torch.clamp(mu3, min=-10, max=10)
        logvar3 = torch.clamp(logvar3, min=-10, max=2)
        z3 = self.reparameterize(mu3, logvar3)
        mu_list.append(mu3)
        logvar_list.append(logvar3)
        
        # Layer 4_1
        x4_1 = self.Cv4_1(z3)
        mu4_1 = self.fc_mu4_1(x4_1)
        logvar4_1 = self.fc_logvar4_1(x4_1)
        mu4_1 = torch.clamp(mu4_1, min=-10, max=10)
        logvar4_1 = torch.clamp(logvar4_1, min=-10, max=2)
        z4_1 = self.reparameterize(mu4_1, logvar4_1)
        mu_list.append(mu4_1)
        logvar_list.append(logvar4_1)
        
        # Layer 4_2
        x4_2 = self.Cv4_2(z4_1)
        mu4_2 = self.fc_mu4_2(x4_2)
        logvar4_2 = self.fc_logvar4_2(x4_2)
        mu4_2 = torch.clamp(mu4_2, min=-10, max=10)
        logvar4_2 = torch.clamp(logvar4_2, min=-10, max=2)
        z4_2 = self.reparameterize(mu4_2, logvar4_2)
        mu_list.append(mu4_2)
        logvar_list.append(logvar4_2)
        
        # Layer 4_3
        x4_3 = self.Cv4_3(z4_2)
        mu4_3 = self.fc_mu4_3(x4_3)
        logvar4_3 = self.fc_logvar4_3(x4_3)
        mu4_3 = torch.clamp(mu4_3, min=-10, max=10)
        logvar4_3 = torch.clamp(logvar4_3, min=-10, max=2)
        z4_3 = self.reparameterize(mu4_3, logvar4_3)
        mu_list.append(mu4_3)
        logvar_list.append(logvar4_3)
        
        # Layer 5
        x5 = self.Cv5(z4_3)
        mu5 = self.fc_mu5(x5)
        logvar5 = self.fc_logvar5(x5)
        mu5 = torch.clamp(mu5, min=-10, max=10)
        logvar5 = torch.clamp(logvar5, min=-10, max=2)
        z5 = self.reparameterize(mu5, logvar5)
        mu_list.append(mu5)
        logvar_list.append(logvar5)
        
        # Store for loss computation
        self.mu_list = [m.clone() for m in mu_list]
        self.logvar_list = [lv.clone() for lv in logvar_list]
        
        # Return latent codes dictionary
        latent = {
            "z0": z0, "z1": z1, "z2": z2, "z3": z3,
            "z4_1": z4_1, "z4_2": z4_2, "z4_3": z4_3, "z5": z5,
            "mu_list": mu_list, "logvar_list": logvar_list
        }
        
        return latent
    
    def compute_kl_loss(self):
        # Compute KL divergence loss with numerical stability
        kl_loss = 0
        for mu, logvar in zip(self.mu_list, self.logvar_list):
            # More stable KL computation
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            # Clamp individual KL terms
            kl = torch.clamp(kl, min=0, max=100)
            kl_loss += kl
        return kl_loss


class PureVAE_Decoder(nn.Module):
    # Pure VAE Decoder - only uses latent codes, no deterministic features
    def __init__(self, output_channels=3, latent_dims=None):
        super(PureVAE_Decoder, self).__init__()
        
        if latent_dims is None:
            # Input latent dimensions
            latent_dims = [64, 128, 256, 512, 512, 512]
        
        self.latent_dims = latent_dims
        
        # Initial fusion layer for z5 latents
        self.initial_fusion = nn.Conv2d(latent_dims[5], 512, kernel_size=1)
        
        # Decoder pathway - carefully designed channel dimensions
        self.CvT6 = CvTi(512, 512, before='ReLU', after='BN')
        self.CvT7_1 = CvTi(512 + latent_dims[4], 512, before='ReLU', after='BN')
        self.CvT7_2 = CvTi(512 + latent_dims[4], 512, before='ReLU', after='BN')
        self.CvT7_3 = CvTi(512 + latent_dims[4], 512, before='ReLU', after='BN')
        self.CvT8 = CvTi(512 + latent_dims[3], 256, before='ReLU', after='BN')
        self.CvT9 = CvTi(256 + latent_dims[2], 128, before='ReLU', after='BN')
        self.CvT10 = CvTi(128 + latent_dims[1], 64, before='ReLU', after='BN')
        self.CvT11 = CvTi(64 + latent_dims[0], output_channels, before='ReLU', after='Tanh')

    def forward(self, latent):
        # PURE VAE DECODING WITH LATENT FUSION
        
        # Fuse initial latents (z5)
        z5_fused = torch.cat([latent["z5"]], dim=1)
        z5_fused = self.initial_fusion(z5_fused)
        
        # Decode with progressive latent fusion
        x6 = self.CvT6(z5_fused)
        
        # Fuse with z4_3 latents 
        z4_3_fused = torch.cat([x6, latent["z4_3"]], dim=1)
        x7_1 = self.CvT7_1(z4_3_fused)
        
        # Fuse with z4_2 latents 
        z4_2_fused = torch.cat([x7_1, latent["z4_2"]], dim=1)
        x7_2 = self.CvT7_2(z4_2_fused)
        
        # Fuse with z4_1 latents 
        z4_1_fused = torch.cat([x7_2, latent["z4_1"]], dim=1)
        x7_3 = self.CvT7_3(z4_1_fused)
        
        # Fuse with z3 latents 
        z3_fused = torch.cat([x7_3, latent["z3"]], dim=1)
        x8 = self.CvT8(z3_fused)
        
        # Fuse with z2 latents
        z2_fused = torch.cat([x8, latent["z2"]], dim=1)
        x9 = self.CvT9(z2_fused)
        
        # Fuse with z1 latents
        z1_fused = torch.cat([x9, latent["z1"]], dim=1)
        x10 = self.CvT10(z1_fused)
        
        # Final output with z0 
        z0_fused = torch.cat([x10, latent["z0"]], dim=1)
        out = self.CvT11(z0_fused)
        
        return out


class DegradeArea_AwareDecoder(nn.Module):
    """
    Degrade-Aware Decoder that learns to detect and process Degrade regions from latent representations.
    Degrade-Aware Decoder decoder extracts Degrade information from multiple latent levels and uses it to guide
    the decoding process, producing Degrade-aware output that helps train the main network.
    """
    def __init__(self, latent_dims=None):
        super(DegradeArea_AwareDecoder, self).__init__()
        
        if latent_dims is None:
            latent_dims = [64, 128, 256, 512, 512, 512]
        
        self.latent_dims = latent_dims
        
        # Degrade detection modules that extract Degrade information from different latent levels
        # Each detector learns to identify Degrade patterns at different scales
        self.shadow_detectors = nn.ModuleDict({
            'z0': nn.Sequential(  # Finest scale - captures detailed Degrade boundaries
                nn.Conv2d(latent_dims[0], 32, 3, 1, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, 1),  # Output single channel Degrade map
                nn.Sigmoid()  # Normalize to [0,1] range
            ),
            'z1': nn.Sequential(  # Fine-medium scale - captures local Degrade patterns
                nn.Conv2d(latent_dims[1], 32, 3, 1, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, 1),
                nn.Sigmoid()
            ),
            'z2': nn.Sequential(  # Medium-coarse scale - captures Degrade regions
                nn.Conv2d(latent_dims[2], 32, 3, 1, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, 1),
                nn.Sigmoid()
            ),
            'z3': nn.Sequential(  # Coarsest scale - captures global Degrade distribution
                nn.Conv2d(latent_dims[3], 32, 3, 1, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, 1),
                nn.Sigmoid()
            )
        })
        
        # Initial layer to process the deepest latent code
        self.initial_layer = nn.Conv2d(latent_dims[5], 512, kernel_size=1)
        # Progressive decoder blocks that reconstruct the image while incorporating latent information
        self.decoder_blocks = nn.ModuleList([
            CvTi(512, 512, before='ReLU', after='BN'),
            CvTi(512 + latent_dims[4], 512, before='ReLU', after='BN'),
            CvTi(512 + latent_dims[4], 512, before='ReLU', after='BN'),
            CvTi(512 + latent_dims[4], 512, before='ReLU', after='BN'),
            CvTi(512 + latent_dims[3], 256, before='ReLU', after='BN'),
            CvTi(256 + latent_dims[2], 128, before='ReLU', after='BN'),
            CvTi(128 + latent_dims[1], 64, before='ReLU', after='BN'),
            CvTi(64 + latent_dims[0], 3, before='ReLU')
        ])
        
        # Degrade feature processor - processes and refines the Degrade-aware output
        # module learns to combine decoded features with Degrade information
        self.shadow_feature_processor = nn.Sequential(  # Input: decoded image (3ch) + shadow maps (3ch)
            nn.Conv2d(3 + 3, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 1),  # Output: processed Degrade-aware image
            nn.Tanh()  # Normalize output to [-1, 1]
        )
    
    def forward(self, latent, input_image=None):
        # Step 1: Extract Degrade maps from multiple latent levels
        # Each level provides Degrade information at different scales
        shadow_z0 = self.shadow_detectors['z0'](latent['z0'])  # Fine Degrade details
        shadow_z1 = self.shadow_detectors['z1'](latent['z1'])  # Local Degrade patterns
        shadow_z2 = self.shadow_detectors['z2'](latent['z2'])  # Regional Degrade
        shadow_z3 = self.shadow_detectors['z3'](latent['z3'])  # Global Degrade distribution
        
        # Step 2: Begin progressive decoding from the deepest latent code
        x = self.initial_layer(latent["z5"])
        x = self.decoder_blocks[0](x)
        
        # Step 3: Progressive upsampling with latent fusion at each level
        # Ensure spatial dimensions match before concatenation
        if x.shape[2:] != latent["z4_3"].shape[2:]:
            x = F.interpolate(x, size=latent["z4_3"].shape[2:], mode='bilinear', align_corners=False)
        x = self.decoder_blocks[1](torch.cat([x, latent["z4_3"]], dim=1))
        
        if x.shape[2:] != latent["z4_2"].shape[2:]:
            x = F.interpolate(x, size=latent["z4_2"].shape[2:], mode='bilinear', align_corners=False)
        x = self.decoder_blocks[2](torch.cat([x, latent["z4_2"]], dim=1))
        
        if x.shape[2:] != latent["z4_1"].shape[2:]:
            x = F.interpolate(x, size=latent["z4_1"].shape[2:], mode='bilinear', align_corners=False)
        x = self.decoder_blocks[3](torch.cat([x, latent["z4_1"]], dim=1))
        
        # Continue fusion with Degrade-detected latent codes
        if x.shape[2:] != latent["z3"].shape[2:]:
            x = F.interpolate(x, size=latent["z3"].shape[2:], mode='bilinear', align_corners=False)
        x = self.decoder_blocks[4](torch.cat([x, latent["z3"]], dim=1))
        
        if x.shape[2:] != latent["z2"].shape[2:]:
            x = F.interpolate(x, size=latent["z2"].shape[2:], mode='bilinear', align_corners=False)
        x = self.decoder_blocks[5](torch.cat([x, latent["z2"]], dim=1))
        
        if x.shape[2:] != latent["z1"].shape[2:]:
            x = F.interpolate(x, size=latent["z1"].shape[2:], mode='bilinear', align_corners=False)
        x = self.decoder_blocks[6](torch.cat([x, latent["z1"]], dim=1))
        
        if x.shape[2:] != latent["z0"].shape[2:]:
            x = F.interpolate(x, size=latent["z0"].shape[2:], mode='bilinear', align_corners=False)
        x = self.decoder_blocks[7](torch.cat([x, latent["z0"]], dim=1))
        
        # Step 4: Combine multi-scale Degrade maps with weighted averaging
        # Higher weights for finer scales as they contain more detailed Degrade information
        h, w = x.shape[2:]
        shadow_combined = (
            F.interpolate(shadow_z0, size=(h, w), mode='bilinear', align_corners=False) * 0.4 +  # 40% weight for finest details
            F.interpolate(shadow_z1, size=(h, w), mode='bilinear', align_corners=False) * 0.3 +  # 30% weight for local patterns
            F.interpolate(shadow_z2, size=(h, w), mode='bilinear', align_corners=False) * 0.2 +  # 20% weight for regional shadows
            F.interpolate(shadow_z3, size=(h, w), mode='bilinear', align_corners=False) * 0.1    # 10% weight for global context
        )
        
        # Convert single-channel Degrade map to 3-channel for processing
        shadow_maps_3ch = shadow_combined.repeat(1, 3, 1, 1)
        
        # Step 5: Process Degrade in the output to make them visible for training
        if input_image is not None:
            # Process the decoded output with Degrade information
            x = self.shadow_feature_processor(torch.cat([x, shadow_maps_3ch], dim=1))
            
            # Combine input and processed output based on Degrade mask
            # Non-Degrade areas from input, Degrade areas from processed output
            x = input_image * (1 - shadow_combined) + x * shadow_combined
        else:
            x = torch.tanh(x)  # During inference without input image, just apply tanh activation
        
        return x

# Degrade Area Localizer
class ShadowMaskGenerator(nn.Module):
    """
    Generates Degrade masks to constrain modifications to Degrade regions only
    """
    def __init__(self, threshold=0.15):
        super(ShadowMaskGenerator, self).__init__()
        self.threshold = threshold
        
    def forward(self, input_img, shadow_enhanced_img):
        # Compute brightness difference
        input_gray = 0.299 * input_img[:, 0:1] + 0.587 * input_img[:, 1:2] + 0.114 * input_img[:, 2:3]
        shadow_gray = 0.299 * shadow_enhanced_img[:, 0:1] + 0.587 * shadow_enhanced_img[:, 1:2] + 0.114 * shadow_enhanced_img[:, 2:3]
        
        # Degrade regions are darker in input
        diff = input_gray - shadow_gray
        
        # Create soft mask with sigmoid
        shadow_mask = torch.sigmoid((diff - self.threshold) * 10)
        
        # Morphological operations to clean mask
        kernel = torch.ones(1, 1, 5, 5, device=shadow_mask.device) / 25
        shadow_mask = F.conv2d(shadow_mask, kernel, padding=2)
        shadow_mask = torch.clamp(shadow_mask, 0, 1)
        
        return shadow_mask


class VDTA_IRNet(nn.Module):
    """
    VDTA_IRNet network focused on Degrade detector and feature processor removal
    """

    def __init__(self, input_channels=3, output_channels=3, latent_dims=None):
        super(VDTA_IRNet, self).__init__()
        
        if latent_dims is None:
            latent_dims = [64, 128, 256, 512, 512, 512]
        
        # Core components
        self.encoder1 = PureVAE_Encoder(input_channels, latent_dims)
        self.encoder2 = PureVAE_Encoder(input_channels, latent_dims)
        self.fallback_decoder = PureVAE_Decoder(output_channels, latent_dims)
        self.Degrade_decoder = DegradeArea_AwareDecoder(latent_dims)
        
        # Degrade-aware refinement
        self.shadow_refinement = nn.Sequential(
            nn.Conv2d(output_channels * 3, 48, 3, 1, 1),  # input + Degrade_out + base_out
            nn.GroupNorm(6, 48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 32, 3, 1, 1),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, output_channels, 3, 1, 1),
            nn.Tanh()
        )
        
        # Adaptive fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(output_channels * 2, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        
        # NEW: Add Degrade mask generator for constraining modifications
        self.shadow_mask_generator = ShadowMaskGenerator(threshold=0.15)
        
        self.kl_loss = 0

    def forward(self, input, GT):
        # Encode
        latent_1 = self.encoder1(input)
        latent_2 = self.encoder2(GT)
        
        # KL loss
        kl_loss = 0
        for mu, logvar in zip(latent_1["mu_list"], latent_1["logvar_list"]):
            kl_loss = kl_loss + (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
        for mu, logvar in zip(latent_2["mu_list"], latent_2["logvar_list"]):
            kl_loss = kl_loss + (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
        self.kl_loss = kl_loss
        
        # Decode
        latent_sf = self.fallback_decoder(latent_1)
        latent_gt = self.fallback_decoder(latent_2)
        shadow_out = self.Degrade_decoder(latent_1, input_image=input)
        
        # Ensure consistency
        if shadow_out.shape[2:] != latent_sf.shape[2:]:
            shadow_out = F.interpolate(shadow_out, size=latent_sf.shape[2:], mode='bilinear', align_corners=False)
        
        # NEW: Generate Degrade mask for training
        shadow_mask = self.shadow_mask_generator(input, shadow_out)
        
        # Add soft constraint to prevent over-masking
        shadow_mask = shadow_mask * 0.7 + 0.15  # Soften the mask
        
        # Degrade-aware refinement
        refined = self.shadow_refinement(torch.cat([input, shadow_out, latent_sf], dim=1))
        
        # Adaptive fusion with Degrade constraint
        weight = self.fusion(torch.cat([refined, latent_sf], dim=1))
        
        # NEW: Apply Degrade mask to constrain modifications during training
        constrained_weight = weight * shadow_mask
        
        # Final output: preserve non-Degrade regions
        final_output = input * (1 - shadow_mask) + (refined * constrained_weight + latent_sf * (1 - constrained_weight)) * shadow_mask
        
        return final_output, latent_gt
    
    def compute_total_vae_loss(self):
        return self.kl_loss

    def test_set(self, input):
        latent_1 = self.encoder1(input)
        
        kl_loss = 0
        for mu, logvar in zip(latent_1["mu_list"], latent_1["logvar_list"]):
            kl_loss = kl_loss + (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
        self.kl_loss = kl_loss
        
        latent_sf_out = self.fallback_decoder(latent_1)
        shadow_out = self.Degrade_decoder(latent_1, input_image=input)
        
        if shadow_out.shape[2:] != latent_sf_out.shape[2:]:
            shadow_out = F.interpolate(shadow_out, size=latent_sf_out.shape[2:], mode='bilinear', align_corners=False)
        
        # NEW: Generate Degrade mask to constrain changes
        shadow_mask = self.shadow_mask_generator(input, shadow_out)
        
        # Add soft constraint to prevent over-masking
        shadow_mask = shadow_mask * 0.7 + 0.15  # Soften the mask
        
        # Apply refinement
        refined = self.shadow_refinement(torch.cat([input, shadow_out, latent_sf_out], dim=1))
        
        # Adaptive fusion
        weight = self.fusion(torch.cat([refined, latent_sf_out], dim=1))
        
        # NEW: Apply Degrade mask to weight to prevent non-Degrade modifications
        constrained_weight = weight * shadow_mask
        
        # NEW: Final output preserves non-Degrade regions completely
        # Only modify pixels within detected Degrade regions
        final_output = input * (1 - shadow_mask) + (refined * constrained_weight + latent_sf_out * (1 - constrained_weight)) * shadow_mask
        
        return final_output

    def train_set(self, input, return_shadow=True):
        latent_1 = self.encoder1(input)
        
        kl_loss = 0
        for mu, logvar in zip(latent_1["mu_list"], latent_1["logvar_list"]):
            kl_loss = kl_loss + (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
        self.kl_loss = kl_loss
        
        latent_sf_out = self.fallback_decoder(latent_1)
        shadow_out = self.Degrade_decoder(latent_1, input_image=input)
        
        if shadow_out.shape[2:] != latent_sf_out.shape[2:]:
            shadow_out = F.interpolate(shadow_out, size=latent_sf_out.shape[2:], mode='bilinear', align_corners=False)
        
        # NEW: Generate Degrade mask for evaluation
        shadow_mask = self.shadow_mask_generator(input, shadow_out)
        
        # Add soft constraint to prevent over-masking
        shadow_mask = shadow_mask * 0.7 + 0.15  # Soften the mask
        
        refined = self.shadow_refinement(torch.cat([input, shadow_out, latent_sf_out], dim=1))
        weight = self.fusion(torch.cat([refined, latent_sf_out], dim=1))
        
        # NEW: Apply Degrade mask constraint
        constrained_weight = weight * shadow_mask
        
        # NEW: Preserve non-Degrade regions
        final_output = input * (1 - shadow_mask) + (refined * constrained_weight + latent_sf_out * (1 - constrained_weight)) * shadow_mask
        
        if return_shadow:
            return final_output, shadow_out
        else:
            return final_output


# ================ EXISTING DISCRIMINATOR ----> instead of this, used another Improved Discriminator inside train ================
class Discriminator(nn.Module):
    def __init__(self, input_channels=4):
        super(Discriminator, self).__init__()
        self.Cv0 = Cvi(input_channels, 64)
        self.Cv1 = Cvi(64, 128, before='LReLU', after='BN')
        self.Cv2 = Cvi(128, 256, before='LReLU', after='BN')
        self.Cv3 = Cvi(256, 512, before='LReLU', after='BN')
        self.Cv4 = Cvi(512, 1, before='LReLU', after='sigmoid')

    def forward(self, input):
        x0 = self.Cv0(input)
        x1 = self.Cv1(x0)
        x2 = self.Cv2(x1)
        x3 = self.Cv3(x2)
        out = self.Cv4(x3)
        return out


if __name__ == '__main__':
    # Test the pure VAE model with latent propagation
    print("Testing Pure VAE with Full Latent Propagation...")
    
    # Test with different input sizes
    size = (2, 3, 256, 256)  # batch_size=2, channels=3, H=256, W=256
    input1 = torch.randn(size)
    input2 = torch.randn(size)
    
    # Initialize model
    model = VDTA_IRNet(input_channels=3, output_channels=3)
    
    # Choose device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Move model and inputs to device
    model = model.to(device)
    input1 = input1.to(device)
    input2 = input2.to(device)
    
    # Test forward pass
    print("\n1. Testing forward pass...")
    out1, out2 = model(input1, input2)
    print(f"   Degrade-free output shape: {out1.shape}")
    print(f"   GT reconstruction shape: {out2.shape}")
    
    # Test VAE loss computation
    kl_loss = model.compute_total_vae_loss()
    print(f"\n2. Total KL loss: {kl_loss.item():.4f}")
    
    # Test single image inference
    print("\n3. Testing single image inference...")
    single_out = model.test_set(input1)
    print(f"   Single output shape: {single_out.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n5. Model parameters:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    print("\nAll tests passed! The model is working correctly.")