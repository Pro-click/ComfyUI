import torch
import math

class PerformanceOptimizer:
    @staticmethod
    def optimize_for_performance():
        # Enable TF32 for better performance on Ampere GPUs
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Set memory efficient attention
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        
        # Optimize PyTorch settings
        torch.set_float32_matmul_precision('high')
        
    @staticmethod
    def get_optimal_batch_size(height, width, max_memory=8):
        # Calculate optimal batch size based on resolution and available memory
        pixels = height * width
        # Approximate memory usage per image (in GB)
        memory_per_image = (pixels * 3 * 4) / (1024 ** 3)  # 3 channels, 4 bytes per pixel
        # Add some overhead for model weights and intermediate activations
        memory_per_image *= 2.5
        
        optimal_batch_size = max(1, math.floor(max_memory / memory_per_image))
        return min(optimal_batch_size, 4)  # Cap at 4 to prevent excessive memory usage