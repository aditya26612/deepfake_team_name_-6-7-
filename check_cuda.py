import torch

print("🔥 PyTorch CUDA Check 🔥\n")

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

# If CUDA is available, print GPU details
if cuda_available:
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    print(f"Current Device Index: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Device Capability: {torch.cuda.get_device_capability(0)}")
    print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"Memory Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
else:
    print("⚠️ CUDA is not available. The model will run on CPU instead.")
