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
from i18n_texts import ui_text


def _convert_models(ckpt_file, pth_file, out_dir_text: str, version: str = "v2") -> Tuple[str, str]:
    """转换模型到 ONNX
    
    Args:
        ckpt_file: .ckpt 文件
        pth_file: .pth 文件
        out_dir_text: 输出目录
        version: 模型版本 ("v2" 或 "v2_pro_plus")
    """
    if ckpt_file is None or pth_file is None:
        return "", "请同时选择 .ckpt 与 .pth 文件。"

    try:
        ckpt_path = ckpt_file.name if hasattr(ckpt_file, "name") else str(ckpt_file)
        pth_path = pth_file.name if hasattr(pth_file, "name") else str(pth_file)
        out_dir = out_dir_text.strip()
        if not out_dir:
            return "", "请指定输出目录。"
        os.makedirs(out_dir, exist_ok=True)

        # 根据版本调用不同的转换函数
        if version == "v2":
            lunavox.convert_to_onnx(
                torch_ckpt_path=ckpt_path,
                torch_pth_path=pth_path,
                output_dir=out_dir,
            )
        elif version == "v2_pro_plus":
            # v2_pro_plus 使用相同的转换函数，因为底层实现相同
            lunavox.convert_to_onnx(
                torch_ckpt_path=ckpt_path,
                torch_pth_path=pth_path,
                output_dir=out_dir,
            )
        else:
            return "", f"不支持的模型版本: {version}"

        version_display = "v2 Pro Plus" if version == "v2_pro_plus" else "v2"
        return "转换完成。", f"{version_display} 模型已输出到：{os.path.abspath(out_dir)}"
    except Exception as e:
        return "", f"转换失败：{e}"


def render_converter_ui():
    with gr.Row():
        with gr.Column():
            # 版本选择（labels will be updated by i18n from webui)
            conv_version = gr.Dropdown(
                choices=["v2", "v2_pro_plus"],
                value="v2",
                label=ui_text("en", "converter", "version_label"),
                interactive=True,
                info=""
            )
            in_ckpt = gr.File(label=ui_text("en", "converter", "in_ckpt_label"), file_types=[".ckpt"], type="filepath")
            in_pth = gr.File(label=ui_text("en", "converter", "in_pth_label"), file_types=[".pth"], type="filepath")
            out_dir = gr.Textbox(label=ui_text("en", "converter", "out_dir_label"), value=str(REPO_ROOT / "Output" / "converted"))
            btn_convert = gr.Button(ui_text("en", "converter", "btn_convert"), variant="primary")
        with gr.Column():
            out_title = gr.Markdown(ui_text("en", "converter", "ready"))
            out_msg = gr.Markdown("")

    btn_convert.click(_convert_models, inputs=[in_ckpt, in_pth, out_dir, conv_version], outputs=[out_title, out_msg])

    # Return components to allow external i18n updates
    return {
        "conv_version": conv_version,
        "in_ckpt": in_ckpt,
        "in_pth": in_pth,
        "out_dir": out_dir,
        "btn_convert": btn_convert,
        "out_title": out_title,
        "out_msg": out_msg,
    }


