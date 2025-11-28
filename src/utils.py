import gc
from pathlib import Path

import torch


def get_project_root() -> Path:
    """Get the project root directory by finding pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: assume utils is located 2 levels down at root/src/utils.py
    return current.parent.parent


def clean_up():
    """
    Util function to clean up memory
    """
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def check_memory():
    """
    Check memory usage for torch across different backends
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"CUDA - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    if torch.backends.mps.is_available():
        allocated = torch.mps.current_allocated_memory() / 1024**3
        print(f"MPS - Allocated: {allocated:.2f} GB")
    print(f"CPU - Memory: {torch.get_num_threads()} threads")
