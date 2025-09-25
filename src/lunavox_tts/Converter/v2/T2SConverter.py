import torch
import onnx
import numpy as np
import json
import os
from collections import OrderedDict

from ..load_state_dict import load_gpt_model


class T2SModelConverter:
    """
    一个专门的转换器，用于处理 t2s (Text-to-Speech) 模型。
    - PyTorch 模型: .ckpt 文件
    - ONNX 模型: t2s_stage_decoder_fp32.onnx
    - 遵循特定的键名映射规则。
    """

    def __init__(self,
                 torch_ckpt_path: str,
                 stage_decoder_onnx_path: str,
                 first_stage_decoder_onnx_path: str,
                 key_list_file: str,
                 output_dir: str,
                 cache_dir: str,
                 ):
        self.torch_ckpt_path: str = torch_ckpt_path
        self.stage_decoder_onnx_path: str = stage_decoder_onnx_path
        self.first_stage_decoder_onnx_path: str = first_stage_decoder_onnx_path
        self.key_list_file: str = key_list_file
        self.output_dir: str = output_dir
        self.cache_dir: str = cache_dir

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # 定义输出文件路径
        self.fp16_bin_path: str = os.path.join(self.output_dir, "t2s_shared_fp16.bin")
        self.index_table_path: str = os.path.join(self.cache_dir, "t2s_weights_index_fp32.json")
        self.relinked_encoder_path: str = os.path.join(self.output_dir, "t2s_encoder_fp32.onnx")
        self.relinked_stage_decoder_path: str = os.path.join(self.output_dir, "t2s_stage_decoder_fp32.onnx")
        self.relinked_first_stage_decoder_path: str = os.path.join(self.output_dir, "t2s_first_stage_decoder_fp32.onnx")
        self.reconstructed_fp32_bin_path = os.path.join(self.output_dir, "t2s_shared_fp32.bin")

    def step1_create_fp16_bin_with_key_mapping(self):
        """
        (1) 根据特定的键映射规则，从 .ckpt 创建 fp16 .bin 和 fp32 索引。
            (已根据用户验证脚本的正确逻辑进行最终修正)
        """
        if not os.path.exists(self.key_list_file):
            raise FileNotFoundError(
                f"Error: Stage 1 requires the key list file, but it was not found: {self.key_list_file}")

        with open(self.key_list_file, 'r') as f:
            onnx_keys = [line.strip() for line in f.readlines()]

        ckpt_data = load_gpt_model(self.torch_ckpt_path)
        if 'weight' not in ckpt_data:
            raise KeyError(
                f"❌ Error: 'weight' key not found in the .ckpt file. Top-level keys in the file are: {list(ckpt_data.keys())}")

        torch_state_dict = ckpt_data['weight']

        index_table = OrderedDict()
        current_fp32_offset = 0

        with open(self.fp16_bin_path, 'wb') as f_bin:
            for onnx_key in onnx_keys:
                transformed_onnx_key = onnx_key.replace('transformer_encoder', 'h')
                torch_lookup_key = f"model.{transformed_onnx_key}"
                torch_tensor = torch_state_dict.get(torch_lookup_key)
                numpy_array_fp16 = torch_tensor.to(torch.float16).cpu().numpy()
                f_bin.write(numpy_array_fp16.tobytes())
                tensor_length_fp32 = numpy_array_fp16.nbytes * 2
                index_table[onnx_key] = {'offset': current_fp32_offset, 'length': tensor_length_fp32}
                current_fp32_offset += tensor_length_fp32

        with open(self.index_table_path, 'w') as f_json:
            json.dump(index_table, f_json, indent=4)  # type: ignore

    def step2_relink_onnx_for_fp32(self, old_model: str, new_model: str):
        """
        (2) 根据 fp32 索引表，修改 ONNX 模型，使其链接到未来的全精度 .bin。
            (使用与第一个脚本相同的、更稳定的底层方法)
        """
        if not os.path.exists(self.index_table_path):
            raise FileNotFoundError(
                f"Error: Stage 2 requires the index file, but it was not found: {self.index_table_path}")

        # 加载描述 fp32 布局的索引表
        with open(self.index_table_path, 'r') as f:
            index_table = json.load(f)

        model = onnx.load_model(old_model, load_external_data=False)
        reconstructed_bin_filename = os.path.basename(self.reconstructed_fp32_bin_path)

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

        onnx.save(model, new_model)

    @staticmethod
    def step3_reconstruct_fp32_bin_from_fp16(fp16_bin_path: str, output_fp32_bin_path: str):
        """
        (3) 静态工具函数：从半精度 .bin 文件还原出全精度 .bin 文件。
        """
        fp16_array = np.fromfile(fp16_bin_path, dtype=np.float16)
        fp32_array = fp16_array.astype(np.float32)
        fp32_array.tofile(output_fp32_bin_path)

    def run_full_process(self):
        self.step1_create_fp16_bin_with_key_mapping()
        self.step2_relink_onnx_for_fp32(self.stage_decoder_onnx_path, self.relinked_stage_decoder_path)
        self.step2_relink_onnx_for_fp32(self.first_stage_decoder_onnx_path, self.relinked_first_stage_decoder_path)
