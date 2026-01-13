import torch
from thop import profile, clever_format
from models.VDTA_IRNet import VDTA_IRNet

# Initialize model
model = VDTA_IRNet(input_channels=3, output_channels=3)
model.eval()

# Create dummy inputs (batch_size=1, for 256x256 image)
input_tensor = torch.randn(1, 3, 256, 256)
gt_tensor = torch.randn(1, 3, 256, 256)

# Calculate FLOPs and Parameters
flops, params = profile(model, inputs=(input_tensor, gt_tensor))

# Format for readability
flops_formatted, params_formatted = clever_format([flops, params], "%.3f")

print(f"Total Parameters: {params_formatted}")
print(f"Total FLOPs: {flops_formatted}")
print("\nIn standard units:")
print(f"Parameters: {params / 1e6:.2f} M")
print(f"GFLOPs: {flops / 1e9:.2f} G")
