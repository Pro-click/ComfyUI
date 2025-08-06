import os
import subprocess
import sys

def install_dependencies():
    print("Checking and installing dependencies...")
    
    # Install PyTorch with CUDA support
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "torch==2.5.0", "torchvision==0.18.1", "torchaudio==2.5.0",
        "-f", "https://download.pytorch.org/whl/cu118/torch_stable.html"
    ])
    
    # Install xformers
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "xformers==0.0.28.post2"
    ])
    
    # Install remaining dependencies
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
    ])
    
    print("Dependencies installed successfully!")

if __name__ == "__main__":
    install_dependencies()