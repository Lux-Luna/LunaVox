#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path

# Add the project root to sys.path so we can import the build_manager package
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from tools.build_manager.main import main

if __name__ == "__main__":
    main()
