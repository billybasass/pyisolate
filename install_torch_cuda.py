#!/usr/bin/env python3
"""
Helper script to detect CUDA version and install appropriate PyTorch version.
"""

import importlib.util
import platform
import re
import subprocess
import sys
from typing import Optional


def run_command(cmd: list[str], capture_output: bool = True) -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    try:
        if capture_output:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
            return result.returncode, result.stdout, result.stderr
        else:
            result = subprocess.run(cmd, check=False)  # noqa: S603
            return result.returncode, "", ""
    except Exception as e:
        return 1, "", str(e)


def get_cuda_version() -> Optional[str]:
    """Detect CUDA version from nvidia-smi."""
    # Try nvidia-smi first
    returncode, stdout, stderr = run_command(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    if returncode != 0:
        print("nvidia-smi not found or failed. CUDA not available.")
        return None

    # Get CUDA version
    returncode, stdout, stderr = run_command(["nvidia-smi"])
    if returncode == 0:
        # Look for CUDA Version in nvidia-smi output
        match = re.search(r"CUDA Version:\s*(\d+\.\d+)", stdout)
        if match:
            cuda_version = match.group(1)
            print(f"Detected CUDA version: {cuda_version}")
            return cuda_version

    # Try nvcc as fallback
    returncode, stdout, stderr = run_command(["nvcc", "--version"])
    if returncode == 0:
        match = re.search(r"release (\d+\.\d+)", stdout)
        if match:
            cuda_version = match.group(1)
            print(f"Detected CUDA version from nvcc: {cuda_version}")
            return cuda_version

    print("Could not detect CUDA version")
    return None


def get_torch_cuda_version(cuda_version: str) -> str:
    """Map CUDA version to PyTorch CUDA version."""
    major, minor = cuda_version.split(".")
    major = int(major)
    minor = int(minor)

    # PyTorch CUDA version mapping (as of late 2024)
    if major >= 12:
        if minor >= 1:
            return "cu121"
        else:
            return "cu118"  # PyTorch typically supports CUDA 11.8 for compatibility
    elif major == 11:
        if minor >= 8:
            return "cu118"
        elif minor >= 7:
            return "cu117"
        else:
            return "cu116"
    elif major == 10:
        return "cu102"
    else:
        print(f"CUDA {cuda_version} is quite old. Using CPU version.")
        return "cpu"


def install_torch(cuda_version: Optional[str] = None):
    """Install PyTorch with appropriate CUDA support."""
    system = platform.system()

    if cuda_version and cuda_version != "cpu":
        torch_cuda = get_torch_cuda_version(cuda_version)

        if torch_cuda == "cpu":
            # CPU only
            print("\nInstalling PyTorch (CPU only)...")
            cmd = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]
        else:
            # CUDA version
            print(f"\nInstalling PyTorch with CUDA {cuda_version} support (torch_{torch_cuda})...")

            # Different index URLs for different CUDA versions
            if torch_cuda in ["cu118", "cu121"]:
                index_url = f"https://download.pytorch.org/whl/{torch_cuda}"
            else:
                index_url = f"https://download.pytorch.org/whl/{torch_cuda}"

            cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                index_url,
            ]
    else:
        # CPU only
        print("\nInstalling PyTorch (CPU only)...")
        if system == "Darwin":  # macOS
            cmd = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]
        else:
            cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ]

    print(f"Running: {' '.join(cmd)}")
    returncode, _, _ = run_command(cmd, capture_output=False)

    if returncode != 0:
        print("\nERROR: Failed to install PyTorch")
        print("You may need to install it manually.")
        print("Visit https://pytorch.org/get-started/locally/ for installation instructions.")
        return False

    print("\nPyTorch installed successfully!")
    return True


def verify_torch_installation():
    """Verify PyTorch installation and CUDA availability."""
    try:
        import torch

        print(f"\n✓ PyTorch version: {torch.__version__}")

        if torch.cuda.is_available():
            print("✓ CUDA available: Yes")
            print(f"✓ CUDA version used by PyTorch: {torch.version.cuda}")
            print(f"✓ Number of GPUs: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

                # Get memory info
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / (1024**3)  # Convert to GB
                print(f"    Memory: {total_memory:.1f} GB")
        else:
            print("✗ CUDA available: No (CPU only)")

        return True
    except ImportError:
        print("\n✗ PyTorch is not installed")
        return False
    except Exception as e:
        print(f"\n✗ Error verifying PyTorch: {e}")
        return False


def main():
    """Main function."""
    print("=" * 60)
    print("PyTorch Installation Helper for pyisolate Benchmarks")
    print("=" * 60)

    # Check if torch is already installed
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is not None:
        print("\nPyTorch is already installed.")
        verify_torch_installation()

        # Ask if user wants to reinstall
        response = input("\nDo you want to reinstall PyTorch? (y/N): ").strip().lower()
        if response != "y":
            print("Keeping existing PyTorch installation.")
            return 0
    else:
        print("\nPyTorch is not installed.")

    # Detect CUDA
    cuda_version = get_cuda_version()

    if cuda_version:
        print(f"\nCUDA {cuda_version} detected.")
        response = input("Do you want to install PyTorch with CUDA support? (Y/n): ").strip().lower()
        if response == "n":
            cuda_version = None

    # Install PyTorch
    if install_torch(cuda_version):
        # Verify installation
        if verify_torch_installation():
            print("\n✓ PyTorch installation complete and verified!")
            return 0
        else:
            print("\n✗ PyTorch installation verification failed.")
            return 1
    else:
        print("\n✗ PyTorch installation failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
