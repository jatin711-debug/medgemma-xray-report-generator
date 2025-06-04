import torch

def get_device():
    """Get the device to run PyTorch models on."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    
    
print(f"Using device: {get_device()}")