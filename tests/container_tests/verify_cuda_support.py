import sys
import torch

def check_pytorch_cuda_support():
    """
    Verifies PyTorch was built with CUDA support, regardless of whether
    a GPU is currently available.
    """
    # Print PyTorch version for reference
    print(f"PyTorch version: {torch.__version__}")
    
    # Check if CUDA is available (will be False without a GPU)
    cuda_available = torch.cuda.is_available()
    print(f"CUDA is available: {cuda_available}")
    
    # Check if PyTorch was compiled with CUDA support
    cuda_version = torch.version.cuda
    
    if cuda_version is not None:
        print(f"PyTorch was compiled with CUDA support (CUDA version: {cuda_version})")
        
        # Get more detailed CUDA capabilities
        print("\nCUDA Capabilities:")
        try:
            device_count = torch.cuda.device_count()
            print(f"Device count: {device_count}")
            
            # List CUDA compute capabilities
            print("CUDA compute capabilities: ", end="")
            # Get this info even if no GPU is attached
            arch_list = torch._C._cuda_getArchFlags()
            if arch_list:
                print(arch_list)
            else:
                print("Not available, but PyTorch has CUDA support")
        except Exception as e:
            # This is expected without a GPU
            print(f"Could not query CUDA capabilities (expected without GPU): {e}")
        
        return True
    else:
        print("ERROR: PyTorch was NOT compiled with CUDA support")
        return False

if __name__ == "__main__":
    print("Checking PyTorch CUDA support...")
    has_cuda_support = check_pytorch_cuda_support()
    
    # Exit with appropriate status code
    if has_cuda_support:
        print("\nSUCCESS: PyTorch has CUDA support")
        sys.exit(0)
    else:
        print("\nFAILURE: PyTorch does not have CUDA support")
        sys.exit(1)