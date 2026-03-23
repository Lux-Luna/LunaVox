from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import time
import sys

@dataclass
class BuildContext:
    root: Path
    env: dict[str, str]
    toolchain_name: str = "unknown"
    toolchain_args: list[str] = field(default_factory=list)
    lib_dir: Path = field(init=False)
    ort_root: Path = field(init=False)
    build_dir: Path = field(init=False)
    log_file: Path = field(init=False)

    def __post_init__(self):
        self.lib_dir = self.root / "lib" / "llama"
        self.ort_root = self.root / "lib" / "onnx"
        self.build_dir = self.root / "build"
        log_dir = self.root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = log_dir / "latest.log"
