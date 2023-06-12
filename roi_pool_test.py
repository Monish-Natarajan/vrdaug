import torch
import torchvision.ops as ops

# Assuming your input tensor is on CPU
input_tensor = torch.randn(1, 3, 224, 224)
# Assuming your regions of interest (ROIs) tensor is on CPU
rois = torch.tensor([[0, 0.25, 0.25, 0.75, 0.75]])
# Using roi_pool on CPU
output = ops.roi_pool(input_tensor, rois, output_size=(7, 7))
# Printing the output shape
print("Output shape:", output.shape)
