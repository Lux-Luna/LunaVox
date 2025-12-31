"""
Model version detection for GPT-SoVITS models (v2, v2Pro, v2ProPlus).
"""
import logging
import sys
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


def ensure_torch() -> None:
    """
    Ensure PyTorch is installed. If not, prompt the user and install the CPU version.
    """
    try:
        import torch
    except ImportError:
        print("\n" + "="*60)
        print("Dependency Missing: PyTorch is required for model conversion.")
        print("LunaVox will now attempt to install the CPU version of PyTorch.")
        print("This is a one-time setup and will take a few minutes (~200MB).")
        print("="*60 + "\n")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "torch", "--index-url", "https://download.pytorch.org/whl/cpu"
            ])
            print("\n" + "-"*60)
            print("PyTorch (CPU) installed successfully!")
            print("-"*60 + "\n")
        except Exception as e:
            logger.error(f"Failed to install PyTorch automatically: {e}")
            print("\n" + "!"*60)
            print("Automatic installation failed.")
            print("Please install PyTorch manually: pip install torch")
            print("!"*60 + "\n")
            raise ImportError("PyTorch is required but could not be installed.")


def detect_version(pth_path: str) -> str:
    """
    Detect GPT-SoVITS model version from .pth config.
    
    Args:
        pth_path: Path to the SoVITS .pth model file
        
    Returns:
        'v2', 'v2Pro', or 'v2ProPlus'
        
    Version detection logic:
    - v2: gin_channels=512, upsample_initial_channel=512
    - v2Pro: gin_channels=1024, upsample_initial_channel=512
    - v2ProPlus: gin_channels=1024, upsample_initial_channel=768
    """
    ensure_torch()
    import torch
    from io import BytesIO
    
    try:
        f = open(pth_path, "rb")
        meta = f.read(2)
        if meta != b"PK":
            data = b"PK" + f.read()
            bio = BytesIO()
            bio.write(data)
            bio.seek(0)
            state = torch.load(bio, map_location='cpu', weights_only=False)
        else:
            f.close()
            state = torch.load(pth_path, map_location='cpu', weights_only=False)
        
        config = state.get('config', {})
        
        if hasattr(config, 'model'):
            model_config = config.model if hasattr(config, 'model') else {}
            gin_channels = getattr(model_config, 'gin_channels', 512)
            upsample_initial = getattr(model_config, 'upsample_initial_channel', 512)
            upsample_kernels = getattr(model_config, 'upsample_kernel_sizes', [])
        else:
            model_config = config.get('model', {})
            gin_channels = model_config.get('gin_channels', 512)
            upsample_initial = model_config.get('upsample_initial_channel', 512)
            upsample_kernels = model_config.get('upsample_kernel_sizes', [])
        
        logger.info(f"Detected config: gin_channels={gin_channels}, "
                   f"upsample_initial_channel={upsample_initial}, "
                   f"upsample_kernel_sizes={upsample_kernels}")
        
        if gin_channels == 512:
            version = 'v2'
        elif upsample_initial == 768:
            version = 'v2ProPlus'
        elif gin_channels == 1024:
            version = 'v2Pro'
        else:
            if upsample_kernels and len(upsample_kernels) > 0:
                if upsample_kernels[0] == 20:
                    version = 'v2ProPlus'
                else:
                    version = 'v2Pro' if gin_channels > 512 else 'v2'
            else:
                logger.warning("Could not definitively determine version, defaulting to v2")
                version = 'v2'
        
        logger.info(f"✓ Detected model version: {version}")
        return version
        
    except Exception as e:
        logger.error(f"Failed to detect version from {pth_path}: {e}")
        logger.warning("Defaulting to v2")
        return 'v2'


def is_v2pro_variant(version: str) -> bool:
    """Check if version is v2Pro or v2ProPlus (requires SV embedding)."""
    return version in ['v2Pro', 'v2ProPlus']


class HParams:
    """Hyperparameters container with attribute access."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()
