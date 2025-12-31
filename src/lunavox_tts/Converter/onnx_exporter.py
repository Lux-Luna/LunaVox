import os
from os import PathLike
from typing import Union

def convert_to_onnx(
        torch_ckpt_path: Union[str, PathLike],
        torch_pth_path: Union[str, PathLike],
        output_dir: Union[str, PathLike],
) -> None:
    """
    Converts PyTorch model checkpoints to the ONNX format.

    This function requires PyTorch to be installed.

    Args:
        torch_ckpt_path (str | PathLike): The path to the T2S model (.ckpt) file.
        torch_pth_path (str | PathLike): The path to the VITS model (.pth) file.
        output_dir (str | PathLike): The directory where the ONNX models will be saved.
    """
    from ..Converter.version_detector import ensure_torch
    ensure_torch()
    from ..Converter.v2.Converter import convert

    torch_ckpt_path = os.fspath(torch_ckpt_path)
    torch_pth_path = os.fspath(torch_pth_path)
    output_dir = os.fspath(output_dir)

    convert(
        torch_pth_path=torch_pth_path,
        torch_ckpt_path=torch_ckpt_path,
        output_dir=output_dir,
    )
