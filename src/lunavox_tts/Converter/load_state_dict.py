import sys
import os

sys.path.append(os.path.dirname(__file__))

import torch
from io import BytesIO
import utils


def load_sovits_model(pth_path: str, device: str = 'cpu'):
    f = open(pth_path, "rb")
    meta = f.read(2)
    if meta != b"PK":
        # noinspection PyTypeChecker
        data = b"PK" + f.read()
        bio = BytesIO()
        # noinspection PyTypeChecker
        bio.write(data)
        bio.seek(0)
        return torch.load(bio, map_location=device, weights_only=False)
    return torch.load(pth_path, map_location=device, weights_only=False)


def load_gpt_model(ckpt_path: str, device: str = 'cpu'):
    return torch.load(ckpt_path, map_location=device, weights_only=True)
