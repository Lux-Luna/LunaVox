import json
import os
from collections import OrderedDict
import logging

import onnx

from ..load_state_dict import load_sovits_model

logger = logging.getLogger(__name__)

class PromptEncoderConverter:
    """
    Converter for Prompt Encoder (v2ProPlus).
    Creates:
    1. FP16 .bin weights file
    2. ONNX model patched to point to the weights
    """

    def __init__(self,
                 torch_pth_path: str,
                 prompt_encoder_onnx_path: str,
                 key_list_file: str,
                 output_dir: str,
                 cache_dir: str,
                 ):
        self.torch_pth_path: str = torch_pth_path
        self.vits_onnx_path: str = prompt_encoder_onnx_path
        self.key_list_file: str = key_list_file
        self.output_dir: str = output_dir
        self.cache_dir: str = cache_dir
        
        # Output paths
        self.fp16_bin_path: str = os.path.join(self.output_dir, "prompt_encoder_fp16.bin")
        self.index_table_path: str = os.path.join(self.cache_dir, "prompt_encoder_weights_index_fp32.json")
        self.relinked_fp32_onnx_path: str = os.path.join(self.output_dir, "prompt_encoder_fp32.onnx")
        # 虚拟文件名，运行时由 ModelManager 拦截并使用 prompt_encoder_fp16.bin 补丁
        self.virtual_fp32_bin_name: str = "prompt_encoder_fp32.bin" 

        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        if not os.path.exists(self.key_list_file):
            raise FileNotFoundError(f"Error: Key list file not found! Path: {self.key_list_file}")

    def step1_create_fp16_bin_and_fp32_index(self):
        """
        Create FP16 bin file and FP32 index table.
        """
        import torch
        with open(self.key_list_file, 'r') as f:
            onnx_keys = [line.strip() for line in f.readlines() if line.strip()]

        # Load PyTorch weights
        state_dict_container = load_sovits_model(self.torch_pth_path)
        if 'weight' in state_dict_container:
            torch_state_dict = state_dict_container['weight']
        else:
            torch_state_dict = state_dict_container # Fallback if direct dict

        index_table = OrderedDict()
        current_fp32_offset = 0

        with open(self.fp16_bin_path, 'wb') as f_bin:
            for onnx_key in onnx_keys:
                # v2pp keys often start with "vq_model." in the onnx_keys list
                # But in the PTH file they might be "enc_p..." or "vq_model.enc_p..."
                
                torch_key = onnx_key
                if onnx_key.startswith("vq_model."):
                    torch_key = onnx_key[9:] # len("vq_model.") == 9

                torch_tensor = torch_state_dict.get(torch_key)
                if torch_tensor is None:
                    # Try raw key just in case
                    torch_tensor = torch_state_dict.get(onnx_key)
                
                if torch_tensor is None:
                    raise ValueError(f"❌ Critical error: Key '{torch_key}' (or '{onnx_key}') not found in PyTorch weights.")

                # Convert to FP16 and write
                torch_tensor_fp16 = torch_tensor.to(torch.float16)
                numpy_array_fp16 = torch_tensor_fp16.cpu().numpy()
                tensor_bytes_fp16 = numpy_array_fp16.tobytes()
                f_bin.write(tensor_bytes_fp16)

                # Record FP32 length/offset for the ONNX link
                tensor_length_fp32 = len(tensor_bytes_fp16) * 2

                index_table[onnx_key] = {
                    'offset': current_fp32_offset,
                    'length': tensor_length_fp32
                }

                current_fp32_offset += tensor_length_fp32

        with open(self.index_table_path, 'w') as f_json:
            json.dump(index_table, f_json, indent=4)

    def step2_relink_onnx_for_fp32(self):
        """
        Patch the ONNX model to point to the virtual external .bin file.
        """
        with open(self.index_table_path, 'r') as f:
            index_table = json.load(f)

        model = onnx.load_model(self.vits_onnx_path, load_external_data=False)
        
        # We point to the virtual filename. 
        # At runtime, LunaVox will intercept this and load from "prompt_encoder_fp16.bin" with patching.
        reconstructed_bin_filename = self.virtual_fp32_bin_name

        for tensor in model.graph.initializer:
            if tensor.name in index_table:
                tensor.ClearField('raw_data')
                tensor.data_location = onnx.TensorProto.EXTERNAL
                info = index_table[tensor.name]

                del tensor.external_data[:]

                keys = ["location", "offset", "length"]
                values = [reconstructed_bin_filename, str(info['offset']), str(info['length'])]

                for k, v in zip(keys, values):
                    entry = tensor.external_data.add()
                    entry.key = k
                    entry.value = v

        onnx.save(model, self.relinked_fp32_onnx_path)

    def run_full_process(self):
        logger.info("Starting PromptEncoder conversion...")
        self.step1_create_fp16_bin_and_fp32_index()
        self.step2_relink_onnx_for_fp32()
        logger.info("PromptEncoder conversion completed.")
