"""
T2S (Text-to-Semantic) model converter.

Handles conversion of:
- T2S Encoder
- T2S First Stage Decoder
- T2S Stage Decoder
"""
import onnx
import numpy as np
import json
import os
import logging
from collections import OrderedDict
from typing import Dict, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


def load_gpt_model(ckpt_path: str) -> Dict[str, Any]:
    """Load GPT model from .ckpt file."""
    import torch
    from io import BytesIO
    
    with open(ckpt_path, "rb") as f:
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


class T2SConverter:
    """
    Converter for T2S Stage Decoder and First Stage Decoder.
    Creates FP16 binary weights + FP32-linked ONNX skeleton.
    """
    
    def __init__(
        self,
        torch_ckpt_path: str,
        stage_decoder_onnx_path: str,
        first_stage_decoder_onnx_path: str,
        key_list_path: str,
        output_dir: str,
        cache_dir: str,
    ):
        self.torch_ckpt_path = torch_ckpt_path
        self.stage_decoder_onnx_path = stage_decoder_onnx_path
        self.first_stage_decoder_onnx_path = first_stage_decoder_onnx_path
        self.key_list_path = key_list_path
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Output paths
        self.fp16_bin_path = os.path.join(output_dir, "t2s_shared_fp16.bin")
        self.index_table_path = os.path.join(cache_dir, "t2s_weights_index_fp32.json")
        self.virtual_fp32_bin_name = "t2s_shared_fp32.bin"
        
    def _create_fp16_bin_with_mapping(self) -> None:
        """Create FP16 binary and FP32 index table from .ckpt."""
        import torch
        
        if not os.path.exists(self.key_list_path):
            raise FileNotFoundError(f"Key list not found: {self.key_list_path}")
            
        with open(self.key_list_path, 'r') as f:
            onnx_keys = [line.strip() for line in f.readlines()]
            
        ckpt_data = load_gpt_model(self.torch_ckpt_path)
        if 'weight' not in ckpt_data:
            raise KeyError(f"'weight' key not found in .ckpt. Keys: {list(ckpt_data.keys())}")
            
        torch_state_dict = ckpt_data['weight']
        index_table = OrderedDict()
        current_fp32_offset = 0
        
        with open(self.fp16_bin_path, 'wb') as f_bin:
            for onnx_key in onnx_keys:
                transformed_key = onnx_key.replace('transformer_encoder', 'h')
                torch_key = f"model.{transformed_key}"
                tensor = torch_state_dict.get(torch_key)
                
                fp16_array = tensor.to(torch.float16).cpu().numpy()
                f_bin.write(fp16_array.tobytes())
                
                # FP32 length = FP16 length * 2
                tensor_length_fp32 = fp16_array.nbytes * 2
                index_table[onnx_key] = {'offset': current_fp32_offset, 'length': tensor_length_fp32}
                current_fp32_offset += tensor_length_fp32
                
        with open(self.index_table_path, 'w') as f:
            json.dump(index_table, f, indent=4)
            
        logger.info(f"✓ Created FP16 binary: {self.fp16_bin_path}")
        
    def _relink_onnx_to_fp32(self, input_onnx: str, output_onnx: str) -> None:
        """Relink ONNX model to use virtual FP32 binary (for CPU FP16->FP32 patching)."""
        with open(self.index_table_path, 'r') as f:
            index_table = json.load(f)
            
        model = onnx.load_model(input_onnx, load_external_data=False)
        
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
                    
        onnx.save(model, output_onnx)
        logger.info(f"✓ Created relinked FP32 ONNX: {output_onnx}")
        
    def _relink_onnx_to_fp16(self, input_onnx: str, output_onnx: str) -> None:
        """Relink ONNX model to use native FP16 binary (for GPU direct loading)."""
        # For FP16, we use the actual FP16 bin file and offsets
        # Since t2s_shared_fp16.bin is purely packed FP16, 
        # offset is current_fp16_offset, length is tensor_size * 2
        with open(self.index_table_path, 'r') as f:
            index_table = json.load(f)
            
        model = onnx.load_model(input_onnx, load_external_data=False)
        bin_name = os.path.basename(self.fp16_bin_path)
        
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
        
        # Optionally convert inputs/outputs to float16 for full FP16 graph
        # For now, keeping weights as FP16 is the most important for memory
        onnx.save(model, output_onnx)
        logger.info(f"✓ Created relinked FP16 ONNX: {output_onnx}")
        
    def convert(self, format: str = "fp16") -> None:
        """Run T2S decoder conversion (Classic FP32 Skeleton)."""
        self._create_fp16_bin_with_mapping()
        
        # Always export FP32 Skeletons (Classic)
        self._relink_onnx_to_fp32(self.stage_decoder_onnx_path, os.path.join(self.output_dir, "t2s_stage_decoder_fp32.onnx"))
        self._relink_onnx_to_fp32(self.first_stage_decoder_onnx_path, os.path.join(self.output_dir, "t2s_first_stage_decoder_fp32.onnx"))


class EncoderConverter:
    """
    Converter for T2S Encoder.
    Merges weights from .ckpt and .pth into a single FP32 binary.
    """
    
    def __init__(
        self,
        ckpt_path: str,
        pth_path: str,
        onnx_template_path: str,
        output_dir: str,
    ):
        self.ckpt_path = ckpt_path
        self.pth_path = pth_path
        self.onnx_template_path = onnx_template_path
        self.output_dir = output_dir
        
        self.output_bin_path = os.path.join(output_dir, "t2s_encoder_fp32.bin")
        self.output_onnx_path = os.path.join(output_dir, "t2s_encoder_fp32.onnx")
        
        os.makedirs(output_dir, exist_ok=True)
        
    def convert(self, format: str = "fp16") -> None:
        """Convert encoder (Classic FP32)."""
        import torch
        
        # Fixed ONNX weight keys (order determines .bin layout)
        onnx_keys = [
            "encoder.ar_text_embedding.word_embeddings.weight",
            "encoder.bert_proj.weight",
            "encoder.bert_proj.bias",
            "encoder.ar_text_position.alpha",
            "vits.ssl_proj.weight",
            "vits.ssl_proj.bias",
            "vits.quantizer.vq.layers.0._codebook.embed"
        ]
        
        ckpt_state = load_gpt_model(self.ckpt_path)['weight']
        pth_state = load_sovits_model(self.pth_path)['weight']
        
        # Always export FP32 Version (Classic)
        self._export_version(onnx_keys, ckpt_state, pth_state, is_fp16=False)

    def _export_version(self, onnx_keys, ckpt_state, pth_state, is_fp16=False) -> None:
        import torch
        suffix = "fp16" if is_fp16 else "fp32"
        bin_path = os.path.join(self.output_dir, f"t2s_encoder_{suffix}.bin")
        onnx_path = os.path.join(self.output_dir, f"t2s_encoder_{suffix}.onnx")
        bin_filename = os.path.basename(bin_path)
        
        model = onnx.load(self.onnx_template_path, load_external_data=False)
        init_map = {init.name: init for init in model.graph.initializer}
        
        current_offset = 0
        dtype = torch.float16 if is_fp16 else torch.float32
        
        with open(bin_path, 'wb') as f_bin:
            for onnx_key in onnx_keys:
                if onnx_key.startswith("encoder."):
                    source_key = "model." + onnx_key[len("encoder."):]
                    source_dict = ckpt_state
                elif onnx_key.startswith("vits."):
                    source_key = onnx_key[len("vits."):]
                    source_dict = pth_state
                else:
                    raise ValueError(f"Unknown key prefix: {onnx_key}")
                    
                tensor = source_dict.get(source_key)
                if tensor is None:
                    raise ValueError(f"Key '{source_key}' not found in source")
                    
                array = tensor.to(dtype).cpu().numpy()
                tensor_bytes = array.tobytes()
                f_bin.write(tensor_bytes)
                
                if onnx_key in init_map:
                    proto = init_map[onnx_key]
                    proto.ClearField('raw_data')
                    proto.data_location = onnx.TensorProto.EXTERNAL
                    if is_fp16:
                        proto.data_type = onnx.TensorProto.FLOAT16
                    del proto.external_data[:]
                    
                    for k, v in zip(["location", "offset", "length"],
                                   [bin_filename, str(current_offset), str(len(tensor_bytes))]):
                        entry = proto.external_data.add()
                        entry.key = k
                        entry.value = v
                        
                current_offset += len(tensor_bytes)
                
        onnx.save(model, onnx_path)
        logger.info(f"✓ Created {suffix.upper()} encoder: {onnx_path}")

