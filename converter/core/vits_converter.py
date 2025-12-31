"""
VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech) converter.

Handles conversion of the VITS vocoder model.
"""
import onnx
import json
import os
import logging
from collections import OrderedDict
from typing import Dict, Any

logger = logging.getLogger(__name__)


def load_sovits_model(pth_path: str) -> Dict[str, Any]:
    """Load SoVITS model from .pth file."""
    import torch
    from io import BytesIO
    
    with open(pth_path, "rb") as f:
        meta = f.read(2)
        if meta != b"PK":
            data = b"PK" + f.read()
            bio = BytesIO()
            bio.write(data)
            bio.seek(0)
            return torch.load(bio, map_location='cpu', weights_only=False)
        else:
            f.seek(0)
            return torch.load(f, map_location='cpu', weights_only=False)


class VITSConverter:
    """
    Converter for VITS vocoder model.
    Creates FP16 binary weights + FP32-linked ONNX skeleton.
    
    Supports v2, v2Pro, v2ProPlus versions.
    """
    
    def __init__(
        self,
        torch_pth_path: str,
        vits_onnx_path: str,
        key_list_path: str,
        output_dir: str,
        cache_dir: str,
        model_version: str = 'v2',
    ):
        self.torch_pth_path = torch_pth_path
        self.vits_onnx_path = vits_onnx_path
        self.key_list_path = key_list_path
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.model_version = model_version
        
        # Output paths
        self.fp16_bin_path = os.path.join(output_dir, "vits_fp16.bin")
        self.index_table_path = os.path.join(cache_dir, "vits_weights_index_fp32.json")
        self.output_onnx_path = os.path.join(output_dir, "vits_fp32.onnx")
        self.virtual_fp32_bin_name = "vits_fp32.bin"
        
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        if not os.path.exists(key_list_path):
            raise FileNotFoundError(f"Key list not found: {key_list_path}")
            
    def _create_fp16_bin_and_index(self) -> None:
        """Create FP16 binary and FP32 index table from .pth."""
        import torch
        
        with open(self.key_list_path, 'r') as f:
            onnx_keys = [line.strip() for line in f.readlines() if line.strip()]
            
        torch_state_dict = load_sovits_model(self.torch_pth_path)['weight']
        index_table = OrderedDict()
        current_fp32_offset = 0
        
        with open(self.fp16_bin_path, 'wb') as f_bin:
            for onnx_key in onnx_keys:
                # Map ONNX key to PyTorch key
                torch_key = onnx_key[len("vq_model."):] if onnx_key.startswith("vq_model.") else onnx_key
                
                tensor = torch_state_dict.get(torch_key)
                if tensor is None:
                    raise ValueError(f"Key '{torch_key}' not found in PyTorch weights")
                    
                fp16_array = tensor.to(torch.float16).cpu().numpy()
                f_bin.write(fp16_array.tobytes())
                
                # FP32 length = FP16 length * 2
                tensor_length_fp32 = fp16_array.nbytes * 2
                index_table[onnx_key] = {'offset': current_fp32_offset, 'length': tensor_length_fp32}
                current_fp32_offset += tensor_length_fp32
                
        with open(self.index_table_path, 'w') as f:
            json.dump(index_table, f, indent=4)
            
        logger.info(f"✓ Created VITS FP16 binary: {self.fp16_bin_path}")
        
    def _relink_onnx_to_fp32(self) -> None:
        """Relink ONNX model to use virtual FP32 binary (for CPU FP16->FP32 patching)."""
        with open(self.index_table_path, 'r') as f:
            index_table = json.load(f)
            
        model = onnx.load_model(self.vits_onnx_path, load_external_data=False)
        
        for tensor in model.graph.initializer:
            if tensor.name in index_table:
                tensor.ClearField('raw_data')
                tensor.data_location = onnx.TensorProto.EXTERNAL
                info = index_table[tensor.name]
                del tensor.external_data[:]
                
                for k, v in zip(["location", "offset", "length"],
                               [self.virtual_fp32_bin_name, str(info['offset']), str(info['length'])]):
                    entry = tensor.external_data.add()
                    entry.key = k
                    entry.value = v
                    
        onnx.save(model, self.output_onnx_path)
        logger.info(f"✓ Created VITS FP32 ONNX: {self.output_onnx_path}")

    def _relink_onnx_to_fp16(self) -> None:
        """Relink ONNX model to use native FP16 binary (for GPU direct loading)."""
        with open(self.index_table_path, 'r') as f:
            index_table = json.load(f)
            
        model = onnx.load_model(self.vits_onnx_path, load_external_data=False)
        bin_name = os.path.basename(self.fp16_bin_path)
        output_path = os.path.join(self.output_dir, "vits_fp16.onnx")
        
        for tensor in model.graph.initializer:
            if tensor.name in index_table:
                info = index_table[tensor.name]
                tensor.ClearField('raw_data')
                tensor.data_location = onnx.TensorProto.EXTERNAL
                tensor.data_type = onnx.TensorProto.FLOAT16
                del tensor.external_data[:]
                
                # FP16 offset = FP32 offset / 2, FP16 length = FP32 length / 2
                fp16_offset = info['offset'] // 2
                fp16_length = info['length'] // 2
                
                for k, v in zip(["location", "offset", "length"],
                               [bin_name, str(fp16_offset), str(fp16_length)]):
                    entry = tensor.external_data.add()
                    entry.key = k
                    entry.value = v
                    
        onnx.save(model, output_path)
        logger.info(f"✓ Created VITS FP16 ONNX: {output_path}")
        
    def convert(self, format: str = "fp16") -> None:
        """Run VITS conversion (Classic FP32)."""
        self._create_fp16_bin_and_index()
        # Always export FP32 Skeletons (Classic)
        self._relink_onnx_to_fp32()

