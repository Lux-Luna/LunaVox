"""
Prompt Encoder converter for v2ProPlus models.
"""
import json
import os
import logging
from collections import OrderedDict
from typing import Dict, Any

import onnx

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


class PromptEncoderConverter:
    """
    Converter for Prompt Encoder (v2ProPlus only).
    Creates FP16 binary weights + FP32-linked ONNX skeleton.
    """
    
    def __init__(
        self,
        torch_pth_path: str,
        onnx_template_path: str,
        key_list_path: str,
        output_dir: str,
        cache_dir: str,
    ):
        self.torch_pth_path = torch_pth_path
        self.onnx_template_path = onnx_template_path
        self.key_list_path = key_list_path
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        
        self.fp16_bin_path = os.path.join(output_dir, "prompt_encoder_fp16.bin")
        self.index_table_path = os.path.join(cache_dir, "prompt_encoder_index_fp32.json")
        self.output_onnx_path = os.path.join(output_dir, "prompt_encoder_fp32.onnx")
        self.virtual_fp32_bin_name = "prompt_encoder_fp32.bin"
        
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
    def _create_fp16_bin_and_index(self) -> None:
        """Create FP16 binary and FP32 index table."""
        import torch
        
        with open(self.key_list_path, 'r') as f:
            onnx_keys = [line.strip() for line in f.readlines() if line.strip()]
            
        state_dict = load_sovits_model(self.torch_pth_path)
        torch_state_dict = state_dict.get('weight', state_dict)
        
        index_table = OrderedDict()
        current_fp32_offset = 0
        
        with open(self.fp16_bin_path, 'wb') as f_bin:
            for onnx_key in onnx_keys:
                torch_key = onnx_key[9:] if onnx_key.startswith("vq_model.") else onnx_key
                
                tensor = torch_state_dict.get(torch_key)
                if tensor is None:
                    tensor = torch_state_dict.get(onnx_key)
                
                if tensor is None:
                    raise ValueError(f"Key '{torch_key}' not found in weights")
                    
                fp16_array = tensor.to(torch.float16).cpu().numpy()
                f_bin.write(fp16_array.tobytes())
                
                tensor_length_fp32 = fp16_array.nbytes * 2
                index_table[onnx_key] = {'offset': current_fp32_offset, 'length': tensor_length_fp32}
                current_fp32_offset += tensor_length_fp32
                
        with open(self.index_table_path, 'w') as f:
            json.dump(index_table, f, indent=4)
            
    def _relink_onnx_to_fp32(self) -> None:
        """Relink ONNX model to use virtual FP32 binary (for CPU FP16->FP32 patching)."""
        with open(self.index_table_path, 'r') as f:
            index_table = json.load(f)
            
        model = onnx.load_model(self.onnx_template_path, load_external_data=False)
        
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
        logger.info(f"✓ Created PromptEncoder FP32 ONNX: {self.output_onnx_path}")

    def _relink_onnx_to_fp16(self) -> None:
        """Relink ONNX model to use native FP16 binary (for GPU direct loading)."""
        with open(self.index_table_path, 'r') as f:
            index_table = json.load(f)
            
        model = onnx.load_model(self.onnx_template_path, load_external_data=False)
        bin_name = os.path.basename(self.fp16_bin_path)
        output_path = os.path.join(self.output_dir, "prompt_encoder_fp16.onnx")
        
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
        logger.info(f"✓ Created PromptEncoder FP16 ONNX: {output_path}")
        
    def convert(self, format: str = "fp16") -> None:
        """Run Prompt Encoder conversion for specific format."""
        logger.info(f"Starting PromptEncoder conversion ({format})...")
        self._create_fp16_bin_and_index()
        if format == "fp16":
            self._relink_onnx_to_fp32()
        else:
            self._relink_onnx_to_fp16()

