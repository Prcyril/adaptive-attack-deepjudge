import torch
import torchvision.transforms as Transformation
from torchvision.transforms import InterpolationMode

# Custom Gaussian Noise Class
class GaussianNoise:
    def __init__(self, mean_val=0.0, std_dev=0.05):
        self.std_dev = std_dev
        self.mean_val = mean_val
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std_dev + self.mean_val



def transform(x: torch.Tensor, exp_cfg):

    # Define the image transformation pipeline
    image_transform = Transformation.Compose([
        Transformation.RandomAffine(
            degrees=2,
            translate=(0.04, 0.04),
            scale=(1.005, 1.02),
        ),
        Transformation.RandomHorizontalFlip(p=0.9),
        GaussianNoise(0, 0.01),
        
    ])
    cpu_tensor = x.cpu()
    
    augmented_tensor  = image_transform(cpu_tensor)
    
    # Move the augmented tensor back to the original device.
    return augmented_tensor.to(exp_cfg.device)


