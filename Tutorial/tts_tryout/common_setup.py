import sys
from pathlib import Path

def configure_paths():
    """
    Configures sys.path to include the local 'src' directory.
    This allows running the tutorial scripts without installing the package via pip.
    """
    SCRIPT_DIR = Path(__file__).parent
    REPO_ROOT = SCRIPT_DIR.parent.parent
    src_path = str(REPO_ROOT / "src")
    
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

if __name__ == "__main__":
    configure_paths()
