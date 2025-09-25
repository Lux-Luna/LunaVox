import torch
import onnx
import os

from ..load_state_dict import load_gpt_model, load_sovits_model


class EncoderConverter:
    """
    一个转换器，用于为 t2s_encoder 模型创建：
    1. 一个从 .ckpt 和 .pth 文件中合并而来的全精度 (fp32) .bin 权重文件。
    2. 一个链接到该 .bin 文件的 ONNX 模型。
    """

    def __init__(self,
                 ckpt_path: str,
                 pth_path: str,
                 onnx_input_path: str,
                 output_dir: str,
                 ):
        self.ckpt_path: str = ckpt_path
        self.pth_path: str = pth_path
        self.onnx_input_path: str = onnx_input_path
        self.output_dir: str = output_dir

        # 定义最终输出文件的路径
        self.output_bin_path: str = os.path.join(self.output_dir, "t2s_encoder_fp32.bin")
        self.output_onnx_path: str = os.path.join(self.output_dir, "t2s_encoder_fp32.onnx")

        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

        # 检查所有输入文件是否存在
        for path in [self.ckpt_path, self.pth_path, self.onnx_input_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Error: Input file not found! Path: {path}")

    def convert(self):
        # 1. 定义固定的 ONNX 权重键列表 (此顺序决定了 .bin 文件的布局)
        onnx_keys = [
            "encoder.ar_text_embedding.word_embeddings.weight",
            "encoder.bert_proj.weight",
            "encoder.bert_proj.bias",
            "encoder.ar_text_position.alpha",
            "vits.ssl_proj.weight",
            "vits.ssl_proj.bias",
            "vits.quantizer.vq.layers.0._codebook.embed"
        ]

        # 2. 加载所有必要的模型和权重
        ckpt_state_dict = load_gpt_model(self.ckpt_path)['weight']
        pth_state_dict = load_sovits_model(self.pth_path)['weight']
        model = onnx.load(self.onnx_input_path, load_external_data=False)
        initializer_map = {init.name: init for init in model.graph.initializer}
        current_offset = 0
        bin_filename = os.path.basename(self.output_bin_path)

        # 3. 生成 .bin 文件并同步修改 ONNX 模型
        with open(self.output_bin_path, 'wb') as f_bin:
            for onnx_key in onnx_keys:
                source_key = ""
                source_dict = None

                if onnx_key.startswith("encoder."):
                    source_key = "model." + onnx_key[len("encoder."):]
                    source_dict = ckpt_state_dict
                elif onnx_key.startswith("vits."):
                    source_key = onnx_key[len("vits."):]
                    source_dict = pth_state_dict

                if source_dict is None:
                    raise ValueError(
                        f"❌ Critical error: Unable to determine the weight source for ONNX key '{onnx_key}'.")
                # 从源文件中提取张量
                tensor = source_dict.get(source_key)
                if tensor is None:
                    raise ValueError(
                        f"❌ Critical error: Key '{source_key}' (corresponding to ONNX key '{onnx_key}') not found in the source file.")

                # 转换为 fp32 numpy 数组并获取字节
                numpy_array_fp32 = tensor.to(torch.float32).cpu().numpy()
                tensor_bytes = numpy_array_fp32.tobytes()
                tensor_length = len(tensor_bytes)
                f_bin.write(tensor_bytes)

                # 在 ONNX 模型中找到对应的 initializer 并修改它
                if onnx_key in initializer_map:
                    tensor_proto = initializer_map[onnx_key]

                    tensor_proto.ClearField('raw_data')
                    tensor_proto.data_location = onnx.TensorProto.EXTERNAL
                    del tensor_proto.external_data[:]

                    keys_to_set = ["location", "offset", "length"]
                    values_to_set = [bin_filename, str(current_offset), str(tensor_length)]

                    for k, v in zip(keys_to_set, values_to_set):
                        entry = tensor_proto.external_data.add()
                        entry.key = k
                        entry.value = v

                # 更新下一个权重的偏移量
                current_offset += tensor_length

        # 4. 保存修改后的 ONNX 模型
        onnx.save(model, self.output_onnx_path)
