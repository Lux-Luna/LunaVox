import os
from pathlib import Path
from typing import Tuple

import gradio as gr

# Prefer local package source like in webui.py
import sys
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
REPO_SRC = REPO_ROOT / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))
import lunavox_tts as lunavox


def _convert_models(ckpt_file, pth_file, out_dir_text: str) -> Tuple[str, str]:
    if ckpt_file is None or pth_file is None:
        return "", "请同时选择 .ckpt 与 .pth 文件。"

    try:
        ckpt_path = ckpt_file.name if hasattr(ckpt_file, "name") else str(ckpt_file)
        pth_path = pth_file.name if hasattr(pth_file, "name") else str(pth_file)
        out_dir = out_dir_text.strip()
        if not out_dir:
            return "", "请指定输出目录。"
        os.makedirs(out_dir, exist_ok=True)

        lunavox.convert_to_onnx(
            torch_ckpt_path=ckpt_path,
            torch_pth_path=pth_path,
            output_dir=out_dir,
        )

        return "转换完成。", f"已输出到：{os.path.abspath(out_dir)}"
    except Exception as e:
        return "", f"转换失败：{e}"


def render_converter_ui() -> None:
    with gr.Accordion("GPT-SoVITS v2 模型转换器", open=False):
        gr.Markdown(
            "选择 GPT/T2S 的 .ckpt 与 VITS 的 .pth 文件，设置输出目录，点击转换。\n\n"
            "注意：本功能仅支持 v2 版本模型。"
        )
        with gr.Row():
            with gr.Column():
                in_ckpt = gr.File(label="选择 .ckpt (GPT/T2S)", file_types=[".ckpt"], type="filepath")
                in_pth = gr.File(label="选择 .pth (VITS)", file_types=[".pth"], type="filepath")
                out_dir = gr.Textbox(label="输出目录", value=str(REPO_ROOT / "Output" / "converted"))
                btn_convert = gr.Button("开始转换", variant="primary")
            with gr.Column():
                out_title = gr.Markdown("准备就绪。")
                out_msg = gr.Markdown("")

        btn_convert.click(_convert_models, inputs=[in_ckpt, in_pth, out_dir], outputs=[out_title, out_msg])


