import onnx
import numpy as np
import json
import os
from collections import OrderedDict

from ..load_state_dict import load_sovits_model


class VITSConverter:
    """
    一个转换器，用于从 PyTorch 模型创建：
    1. 一个用于分发的半精度 (fp16) .bin 权重文件。
    2. 一个与全精度 (fp32) 布局兼容的 ONNX 模型。
    
    支持 v2, v2Pro, v2ProPlus 三个版本。
    """

    def __init__(self,
                 torch_pth_path: str,
                 vits_onnx_path: str,
                 key_list_file: str,
                 output_dir: str,
                 cache_dir: str,
                 model_version: str = 'v2',
                 ):
        self.torch_pth_path: str = torch_pth_path
        self.vits_onnx_path: str = vits_onnx_path
        self.key_list_file: str = key_list_file
        self.output_dir: str = output_dir
        self.cache_dir: str = cache_dir
        self.model_version: str = model_version
        # 定义输出文件路径
        self.fp16_bin_path: str = os.path.join(self.output_dir, "vits_fp16.bin")
        self.index_table_path: str = os.path.join(self.cache_dir, "vits_weights_index_fp32.json")
        self.relinked_fp32_onnx_path: str = os.path.join(self.output_dir, "vits_fp32.onnx")
        # 虚拟文件名，运行时由 ModelManager 拦截并使用 vits_fp16.bin 补丁
        self.virtual_fp32_bin_name: str = "vits_fp32.bin"

        # 确保输出目录存在
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        if not os.path.exists(self.key_list_file):
            raise FileNotFoundError(f"Error: Key list file not found! Path: {self.key_list_file}")

    def step1_create_fp16_bin_and_fp32_index(self):
        """
        (1) 创建一个半精度 (fp16) 的 .bin 文件，但生成一个
            描述全精度 (fp32) 布局的索引表。
        """
        import torch
        # 加载 key 列表（过滤空行）
        with open(self.key_list_file, 'r') as f:
            onnx_keys = [line.strip() for line in f.readlines() if line.strip()]

        # 加载 PyTorch 模型权重
        torch_state_dict = load_sovits_model(self.torch_pth_path)['weight']

        index_table = OrderedDict()
        current_fp32_offset = 0

        with open(self.fp16_bin_path, 'wb') as f_bin:
            for onnx_key in onnx_keys:
                torch_key = onnx_key[len("vq_model."):] if onnx_key.startswith("vq_model.") else onnx_key

                torch_tensor = torch_state_dict.get(torch_key)
                if torch_tensor is None:
                    raise ValueError(f"❌ Critical error: Key '{torch_key}' not found in the PyTorch weights")

                # 转换为 fp16 并写入文件
                torch_tensor_fp16 = torch_tensor.to(torch.float16)
                numpy_array_fp16 = torch_tensor_fp16.cpu().numpy()
                tensor_bytes_fp16 = numpy_array_fp16.tobytes()
                f_bin.write(tensor_bytes_fp16)
                tensor_length_fp32 = len(tensor_bytes_fp16) * 2
                index_table[onnx_key] = {
                    'offset': current_fp32_offset,
                    'length': tensor_length_fp32
                }
                current_fp32_offset += tensor_length_fp32

        # 保存描述 fp32 布局的索引表
        with open(self.index_table_path, 'w') as f_json:
            json.dump(index_table, f_json, indent=4)  # type: ignore

    def step2_relink_onnx_for_fp32(self):
        """
        (2) 根据 fp32 索引表，修改 ONNX 模型，使其链接到一个
            虚拟的全精度 .bin 文件名。
        """
        # 加载描述 fp32 布局的索引表
        with open(self.index_table_path, 'r') as f:
            index_table = json.load(f)

        model = onnx.load_model(self.vits_onnx_path, load_external_data=False)
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

        # 保存修改后的、链接到虚拟 fp32 权重的 ONNX 模型
        onnx.save(model, self.relinked_fp32_onnx_path)

    def run_full_process(self):
        self.step1_create_fp16_bin_and_fp32_index()
        self.step2_relink_onnx_for_fp32()
