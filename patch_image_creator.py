import torch
import numpy as np
from PIL import Image
import math

class PatchImageCreator:
    @staticmethod
    def create_patch_image(original_image, patch_size=512, overlap=64):
        # Convert tensor to PIL Image if needed
        if isinstance(original_image, torch.Tensor):
            image_array = original_image.cpu().numpy()
            image_array = (image_array * 255).astype(np.uint8)
            original_pil = Image.fromarray(image_array[0])
        else:
            original_pil = original_image
            
        width, height = original_pil.size
        
        # Calculate number of patches needed
        x_patches = math.ceil((width - overlap) / (patch_size - overlap))
        y_patches = math.ceil((height - overlap) / (patch_size - overlap))
        
        patches = []
        positions = []
        
        for y in range(y_patches):
            for x in range(x_patches):
                # Calculate patch position
                x_pos = min(x * (patch_size - overlap), width - patch_size)
                y_pos = min(y * (patch_size - overlap), height - patch_size)
                
                # Extract patch
                patch = original_pil.crop((x_pos, y_pos, x_pos + patch_size, y_pos + patch_size))
                
                # Convert patch to tensor
                patch_array = np.array(patch).astype(np.float32) / 255.0
                patch_tensor = torch.from_numpy(patch_array).unsqueeze(0)
                
                patches.append(patch_tensor)
                positions.append((x_pos, y_pos))
                
        return patches, positions, (width, height)
    
    @staticmethod
    def reconstruct_image(patches, positions, original_size):
        width, height = original_size
        patch_size = patches[0].shape[1]
        
        # Create empty canvas
        result = torch.zeros((1, height, width, 3), device=patches[0].device)
        count = torch.zeros((1, height, width, 1), device=patches[0].device)
        
        # Place patches on canvas with blending
        for patch, (x_pos, y_pos) in zip(patches, positions):
            x_end = min(x_pos + patch_size, width)
            y_end = min(y_pos + patch_size, height)
            
            # Calculate weights for blending (simple linear blend)
            x_weights = torch.ones((1, patch.shape[1], patch.shape[2], 1), device=patch.device)
            y_weights = torch.ones((1, patch.shape[1], patch.shape[2], 1), device=patch.device)
            
            # Apply blending at edges
            blend_width = 64  # Width of the blending region
            
            # Left edge
            if x_pos > 0:
                for i in range(blend_width):
                    if x_pos + i < x_end:
                        x_weights[:, :, i, :] = (i + 1) / (blend_width + 1)
            
            # Right edge
            if x_pos + patch_size < width:
                for i in range(blend_width):
                    if x_pos + patch_size - i - 1 >= x_pos:
                        x_weights[:, :, patch_size - i - 1, :] = (i + 1) / (blend_width + 1)
            
            # Top edge
            if y_pos > 0:
                for i in range(blend_width):
                    if y_pos + i < y_end:
                        y_weights[:, i, :, :] = (i + 1) / (blend_width + 1)
            
            # Bottom edge
            if y_pos + patch_size < height:
                for i in range(blend_width):
                    if y_pos + patch_size - i - 1 >= y_pos:
                        y_weights[:, patch_size - i - 1, :, :] = (i + 1) / (blend_width + 1)
            
            # Combined weights
            weights = x_weights * y_weights
            
            # Apply patch to canvas
            result[:, y_pos:y_end, x_pos:x_end, :] += patch[:, :y_end-y_pos, :x_end-x_pos, :] * weights[:, :y_end-y_pos, :x_end-x_pos, :]
            count[:, y_pos:y_end, x_pos:x_end, :] += weights[:, :y_end-y_pos, :x_end-x_pos, :]
        
        # Normalize by weights to handle overlapping regions
        result = result / (count + 1e-8)  # Add small epsilon to avoid division by zero
        
        return torch.clamp(result, 0, 1)