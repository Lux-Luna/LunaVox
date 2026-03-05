"""
Build script for LunaVox C++ Accelerator Plugin.

Usage:
    conda run -n lunavox python setup.py build_ext --inplace

This will produce lunavox_accelerator.pyd in the current directory.
"""

import os
import sys
import platform
from setuptools import setup, Extension
import pybind11

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ORT_SDK_DIR = os.path.join(SCRIPT_DIR, "deps", "ort_sdk", "onnxruntime-win-x64-gpu-1.20.1")
ORT_INCLUDE = os.path.join(ORT_SDK_DIR, "include")
ORT_LIB_DIR = os.path.join(ORT_SDK_DIR, "lib")
SRC_DIR = os.path.join(SCRIPT_DIR, "src")

# Validate paths
if not os.path.exists(ORT_INCLUDE):
    raise FileNotFoundError(f"ORT include dir not found: {ORT_INCLUDE}")
if not os.path.exists(ORT_LIB_DIR):
    raise FileNotFoundError(f"ORT lib dir not found: {ORT_LIB_DIR}")

ext_modules = [
    Extension(
        "lunavox_accelerator",
        sources=[os.path.join(SRC_DIR, "lunavox_accelerator.cpp")],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            ORT_INCLUDE,
        ],
        library_dirs=[ORT_LIB_DIR],
        libraries=["onnxruntime"],
        language="c++",
        extra_compile_args=[
            "/std:c++17",
            "/O2",         # Optimize for speed
            "/EHsc",       # Exception handling
            "/DWIN32",
            "/D_WINDOWS",
            "/DNOMINMAX",  # Prevent windows.h min/max macros
        ],
        extra_link_args=[],
    )
]

setup(
    name="lunavox_accelerator",
    version="0.1.0",
    description="LunaVox C++ Accelerator - Native T2S inference loop",
    ext_modules=ext_modules,
    python_requires=">=3.9",
)
