import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

class ImageQualityEnhancer:
    @staticmethod
    def enhance_image(image_tensor, sharpness=1.2, contrast=1.1, saturation=1.1):
        # Convert tensor to PIL Image
        if isinstance(image_tensor, torch.Tensor):
            image_array = image_tensor.cpu().numpy()
            image_array = (image_array * 255).astype(np.uint8)
            image = Image.fromarray(image_array[0])
        else:
            image = image_tensor
            
        # Apply enhancements
        if sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(sharpness)
            
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast)
            
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(saturation)
            
        # Convert back to tensor
        image_array = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).unsqueeze(0)
        
        return image_tensor
    
    @staticmethod
    def apply_detail_enhancement(image_tensor, strength=0.2):
        # Apply a subtle detail enhancement using unsharp masking
        if isinstance(image_tensor, torch.Tensor):
            # Apply Gaussian blur
            blurred = F.gaussian_blur(image_tensor, kernel_size=5, sigma=1.0)
            
            # Calculate the difference (details)
            details = image_tensor - blurred
            
            # Add details back with strength control
            enhanced = image_tensor + details * strength
            
            # Clamp values
            enhanced = torch.clamp(enhanced, 0, 1)
            
            return enhanced
        return image_tensor