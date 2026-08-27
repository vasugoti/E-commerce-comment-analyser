"""
Reproducibility utilities.

Ensures deterministic behavior across Python, NumPy, and PyTorch by setting
consistent random seeds. All experiments use seeds 13, 42, 2024 as defined
in the research plan §9.
"""

import os
import random
import numpy as np

# Default seeds for all experiments (§9)
DEFAULT_SEEDS = [13, 42, 2024]


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Integer seed value.
        deterministic: If True, enable PyTorch deterministic mode (slower but reproducible).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # PyTorch 1.8+ deterministic algorithms
            try:
                torch.use_deterministic_algorithms(True)
            except AttributeError:
                pass  # Older PyTorch version
    except ImportError:
        pass  # PyTorch not installed


def get_device() -> str:
    """
    Get the best available compute device.

    Returns:
        'cuda' if GPU available, else 'cpu'.
    """
    try:
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            print(f"  Using GPU: {gpu_name} ({gpu_mem:.1f} GB)")
            return device
    except ImportError:
        pass
    print("  Using CPU")
    return 'cpu'


def log_environment() -> dict:
    """
    Log the current environment for reproducibility.

    Returns:
        Dict of library versions and hardware info.
    """
    import sys
    import platform

    env = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'processor': platform.processor(),
    }

    # Library versions
    libs = ['numpy', 'pandas', 'scipy', 'sklearn', 'torch', 'transformers',
            'datasets', 'gensim', 'optuna', 'nltk']
    for lib in libs:
        try:
            mod = __import__(lib)
            env[f'{lib}_version'] = getattr(mod, '__version__', 'unknown')
        except ImportError:
            env[f'{lib}_version'] = 'not installed'

    # GPU info
    try:
        import torch
        env['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env['gpu_name'] = torch.cuda.get_device_name(0)
            env['gpu_count'] = torch.cuda.device_count()
            env['cuda_version'] = torch.version.cuda
    except ImportError:
        env['cuda_available'] = False

    return env
