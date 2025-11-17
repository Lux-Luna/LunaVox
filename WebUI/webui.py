import logging
import os
import json
import time
from pathlib import Path
import socket
import sys
import tempfile
from typing import List, Optional, Tuple

# Import LunaVox TTS public APIs (support running from repo without installation)
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent  # Go up one level from WebUI to repo root
REPO_SRC = REPO_ROOT / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))
import lunavox_tts as lunavox
from lunavox_tts import unload_character
from lunavox_tts.ModelManager import model_manager
import gradio as gr
import numpy as np
import soundfile as sf
from converter import render_converter_ui
from i18n_texts import (
    get_guide_markdown,
    get_supported_language_display,
    display_to_code,
    code_to_display,
    ui_text,
    get_prompt_language_choices,
    get_output_language_choices,
)

logger = logging.getLogger(__name__)

# ------------------------------
# Paths and environment setup
# ------------------------------
DATA_DIR = REPO_ROOT / "Data"
CHAR_MODEL_DIR_V2 = DATA_DIR / "character_model" / "v2"
CHAR_MODEL_DIR_V2_PRO_PLUS = DATA_DIR / "character_model" / "v2_pro_plus"
AUDIO_RESOURCES_DIR = DATA_DIR / "audio_resources"
OUTPUT_DIR = REPO_ROOT / "Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Assets
ASSETS_DIR = SCRIPT_DIR / "assets"
LANGUAGE_SVG_PATH = ASSETS_DIR / "language.svg"
DEFAULT_WEBUI_PORT = 7860
LANG_CODE_TO_AUDIO_DIR = {
    "ja": AUDIO_RESOURCES_DIR / "Japanese",
    "zh": AUDIO_RESOURCES_DIR / "Chinese",
    "en": AUDIO_RESOURCES_DIR / "English",
}

# Prefer local dependencies to avoid downloads
os.environ.setdefault("HUBERT_MODEL_PATH", str(DATA_DIR / "chinese-hubert-base.onnx"))
os.environ.setdefault("OPEN_JTALK_DICT_DIR", str(DATA_DIR / "open_jtalk_dic_utf_8-1.11"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


# ------------------------------
# Utilities
# ------------------------------
def _read_text_file(path: Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception:
        return ""


def _is_port_available(port: int) -> bool:
    if port < 1 or port > 65535:
        return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _pick_server_port(default_port: int = DEFAULT_WEBUI_PORT, scan_span: int = 50) -> int:
    candidates: list[int] = []
    env_value = os.getenv("GRADIO_SERVER_PORT")
    if env_value:
        try:
            env_port = int(env_value)
            candidates.append(env_port)
        except ValueError:
            logger.warning("Invalid GRADIO_SERVER_PORT value '%s'. Ignoring.", env_value)
    if default_port not in candidates:
        candidates.append(default_port)
    for offset in range(scan_span):
        candidate = default_port + offset
        if candidate not in candidates:
            candidates.append(candidate)
    for port in candidates:
        if _is_port_available(port):
            if port != default_port:
                logger.info("Selected alternative WebUI port %s (default %s busy).", port, default_port)
            return port
    raise RuntimeError("Unable to find a free TCP port for the WebUI.")
def _is_valid_character_dir(path: Path, version: str = "v2") -> bool:
    """验证角色模型目录是否有效
    
    Args:
        path: 模型目录路径
        version: 模型版本 ("v2" 或 "v2_pro_plus")
    """
    if not path.is_dir():
        return False
    
    names = {p.name for p in path.iterdir() if p.is_file()}
    
    if version == "v2":
        required_onnx = {
            't2s_encoder_fp32.onnx',
            't2s_first_stage_decoder_fp32.onnx',
            't2s_stage_decoder_fp32.onnx',
            'vits_fp32.onnx',
        }
        # 需要 fp16 权重来生成 fp32（加载前会自动生成）
        required_fp16_bin = {
            't2s_shared_fp16.bin',
            'vits_fp16.bin',
        }
        
        if not required_onnx.issubset(names):
            return False
        if not required_fp16_bin.issubset(names):
            return False
        return True
    
    elif version == "v2_pro_plus":
        # v2 pro plus 使用相同的文件结构
        required_onnx = {
            't2s_encoder_fp32.onnx',
            't2s_first_stage_decoder_fp32.onnx',
            't2s_stage_decoder_fp32.onnx',
            'vits_fp32.onnx',
        }
        required_fp16_bin = {
            't2s_shared_fp16.bin',
            'vits_fp16.bin',
        }
        
        if not required_onnx.issubset(names):
            return False
        if not required_fp16_bin.issubset(names):
            return False
        return True
    
    return False


def _to_lang_code(display: str) -> str:
    """将下拉显示文本映射为语言代码。"""
    mapping = {
        "中文": "zh",
        "英语": "en",
        "日语": "ja",
        # English display
        "Chinese": "zh",
        "English": "en",
        "Japanese": "ja",
        # 容错：若直接传入代码
        "zh": "zh",
        "en": "en",
        "ja": "ja",
    }
    return mapping.get((display or "").strip(), "ja")


def list_character_folders(version: str = "v2") -> List[str]:
    """列出指定版本的角色模型文件夹
    
    Args:
        version: 模型版本 ("v2" 或 "v2_pro_plus")
    """
    if version == "v2":
        base_dir = CHAR_MODEL_DIR_V2
    elif version == "v2_pro_plus":
        base_dir = CHAR_MODEL_DIR_V2_PRO_PLUS
    else:
        return []
    
    if not base_dir.exists():
        return []
    
    folders = [p.name for p in base_dir.iterdir() if _is_valid_character_dir(p, version)]
    folders.sort()
    return folders


def get_model_dir(character_name: str, version: str = "v2") -> Path:
    """获取角色模型目录路径
    
    Args:
        character_name: 角色名称
        version: 模型版本 ("v2" 或 "v2_pro_plus")
    """
    if version == "v2":
        return CHAR_MODEL_DIR_V2 / character_name
    elif version == "v2_pro_plus":
        return CHAR_MODEL_DIR_V2_PRO_PLUS / character_name
    else:
        return CHAR_MODEL_DIR_V2 / character_name


def list_language_audio_resources(language: str) -> List[Tuple[str, str]]:
    """
    根据语言代码列出 audio_resources/<Language> 下的 wav 文件。
    """
    lang_code = (language or "ja").lower()
    lang_code = lang_code if lang_code in LANG_CODE_TO_AUDIO_DIR else "ja"
    folder = LANG_CODE_TO_AUDIO_DIR.get(lang_code)
    if not folder or not folder.exists():
        return []
    audio_files: List[Tuple[str, str]] = []
    for audio_file in folder.glob("*.wav"):
        if audio_file.is_file():
            audio_files.append((audio_file.stem, str(audio_file)))
    audio_files.sort(key=lambda x: x[0].lower())
    return audio_files


def load_default_prompt(character_name: str) -> Tuple[Optional[str], Optional[str], str]:
    # Default prompt loading is disabled per user requirement.
    return None, None, ""


def ensure_character_loaded(character_name: str, version: str = "v2", language: str = "ja") -> Tuple[str, List[Tuple[str, str]]]:
    """加载角色模型
    
    Args:
        character_name: 角色名称
        version: 模型版本 ("v2" 或 "v2_pro_plus")
    """
    try:
        model_dir = get_model_dir(character_name, version)
        
        # 检查模型目录是否存在
        if not model_dir.exists():
            version_display = "v2 Pro Plus" if version == "v2_pro_plus" else "v2"
            return f"错误：角色 {character_name} 的 {version_display} 模型目录不存在：{model_dir}", []
        
        # 先卸载同名角色（如果存在）以避免冲突
        try:
            lunavox.unload_character(character_name)
        except Exception as e:
            # 卸载失败是正常的，可能角色本来就不存在
            print(f"卸载角色时出错（可忽略）: {e}")
        
        # 加载新模型
        lunavox.load_character(character_name, str(model_dir))
        
        audio_resources = list_language_audio_resources(language)
        
        version_display = "v2 Pro Plus" if version == "v2_pro_plus" else "v2"
        if audio_resources:
            return f"角色 {character_name} ({version_display}) 模型已加载，找到 {len(audio_resources)} 个预设音频。", audio_resources
        else:
            return f"角色 {character_name} ({version_display}) 模型已加载。", []
            
    except Exception as e:
        version_display = "v2 Pro Plus" if version == "v2_pro_plus" else "v2"
        return f"加载角色 {character_name} ({version_display}) 模型时出错：{str(e)}", []


def set_reference(character_name: str, audio_path: str, audio_text: str, audio_lang: str = "ja") -> str:
    if not audio_path:
        return "请先上传参考音频。"
    if not audio_text:
        return "请填写参考音频对应的文本。"
    
    try:
        # 检查文件是否存在且可读
        if not os.path.exists(audio_path):
            return f"音频文件不存在: {audio_path}"
        
        # 尝试复制文件到输出目录以避免权限问题
        import shutil
        from pathlib import Path
        
        # 创建临时文件以避免权限问题
        temp_dir = OUTPUT_DIR / "temp_audio"
        temp_dir.mkdir(exist_ok=True)
        
        temp_audio_path = temp_dir / f"ref_{character_name}_{int(time.time())}.wav"
        
        try:
            shutil.copy2(audio_path, temp_audio_path)
            # 使用临时文件路径设置参考音频
            lunavox.set_reference_audio(character_name, str(temp_audio_path), audio_text, audio_lang)
            return "参考音频设置成功。"
        except PermissionError:
            # 如果复制失败，直接使用原文件路径
            lunavox.set_reference_audio(character_name, audio_path, audio_text, audio_lang)
            return "参考音频设置成功（使用原始文件）。"
            
    except Exception as e:
        return f"设置参考音频时出错: {str(e)}"


def synthesize(character_name: str, text: str, language: str) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
    if not text or not text.strip():
        return None, "请输入要合成的文本。"

    tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = Path(tmp_file.name)
    tmp_file.close()

    try:
        lunavox.tts(
            character_name=character_name,
            text=text.strip(),
            play=False,
            split_sentence=True,
            save_path=str(tmp_path),
            language=language,
        )
        if not tmp_path.exists():
            return None, "合成失败，请检查日志。"
        audio_data, sample_rate = sf.read(tmp_path, dtype="float32")
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        return (sample_rate, audio_data), "合成完成。"
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


# ------------------------------
# Gradio UI
# ------------------------------
def build_ui() -> gr.Blocks:
    with gr.Blocks(css="footer {visibility: hidden} .boxed {border: 1px solid #e5e7eb; padding: 12px; border-radius: 6px; margin-bottom: 8px;}") as demo:
        gr.Markdown("""
        **LunaVox 本地 WebUI**  
        - 支持 v2 和 v2 Pro Plus 模型版本
        - 自动扫描对应版本目录下的角色模型  
        - 上传参考音频与文本，一键合成语音  
        - 生成音频保存在 `Output` 目录下，并可在线试听
        """)

        # Global language selector (default English)
        with gr.Row():
            svg_html = _read_text_file(LANGUAGE_SVG_PATH)
            if svg_html:
                gr.HTML(f"<div style='display:flex;align-items:center;gap:8px'>{svg_html}<span style='font-weight:600'>Language</span></div>")
            else:
                gr.Markdown("🌐 Language")
            ui_lang_dd = gr.Dropdown(
                choices=["English", "中文"],
                value="English",
                label=None,
                interactive=True,
            )

        # 使用 Tab 切换模块，默认显示“语音合成”（第一个 Tab）
        with gr.Tabs():
            with gr.TabItem("语音合成"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # 版本选择
                        dd_version = gr.Dropdown(
                            choices=["v2", "v2_pro_plus"],
                            value="v2",
                            label=ui_text("en", "webui", "version_label"),
                            interactive=True,
                        )
                        
                        character_list = list_character_folders("v2")
                        dd_character = gr.Dropdown(
                            choices=character_list,
                            value=None,
                            label=ui_text("en", "webui", "character_label"),
                            interactive=True,
                            info=ui_text("en", "webui", "character_info"),
                        )
                        btn_load_character = gr.Button(ui_text("en", "webui", "btn_load"), variant="primary")
                        btn_unload_character = gr.Button("Unload Character & Cleanup", variant="secondary")
                        status = gr.Markdown(ui_text("en", "webui", "status_ready"))

                        # States
                        st_version = gr.State("v2")
                        st_character = gr.State("")
                        st_loaded_character = gr.State("")
                        st_ref_audio_path = gr.State("")
                        st_ref_audio_text = gr.State("")
                        st_ui_lang = gr.State("en")

                    with gr.Column(scale=2):
                        ref_section_title_md = gr.Markdown(ui_text("en", "webui", "ref_section_title"))
                        
                        # 参考音频资源下拉选择器（独立框）
                        with gr.Group(elem_classes=["boxed"]):
                            ref_audio_dropdown = gr.Dropdown(
                                label=ui_text("en", "webui", "preset_ref_label"),
                                choices=list_language_audio_resources("ja"),
                                value=None,
                                interactive=True,
                                allow_custom_value=False,
                                info=ui_text("en", "webui", "preset_ref_info"),
                            )

                        # 参考音频语言（独立框，位于预设参考音频下方）
                        with gr.Group(elem_classes=["boxed"]):
                            ref_lang_dd = gr.Dropdown(
                                choices=get_prompt_language_choices("en"),
                                value="Japanese",
                                label=ui_text("en", "webui", "ref_lang_label"),
                                interactive=True,
                            )
                        
                        or_md = gr.Markdown(ui_text("en", "webui", "or"))
                        
                        ref_audio = gr.Audio(
                            label=ui_text("en", "webui", "upload_ref_label"),
                            sources=["upload"],
                            type="filepath",
                        )
                        auto_filename = gr.Checkbox(
                            label=ui_text("en", "webui", "auto_filename_label"),
                            value=True,
                            info=ui_text("en", "webui", "auto_filename_info"),
                        )
                        ref_text = gr.Textbox(label=ui_text("en", "webui", "ref_text_label"), lines=2, placeholder=ui_text("en", "webui", "ref_text_placeholder"))

                        synth_section_title_md = gr.Markdown(ui_text("en", "webui", "synth_section_title"))
                        lang_dd = gr.Dropdown(
                            choices=get_output_language_choices("en"),
                            value=get_output_language_choices("en")[0],
                            label=ui_text("en", "webui", "output_lang_label"),
                            interactive=True,
                        )
                        input_text = gr.Textbox(label=ui_text("en", "webui", "input_text_label"), lines=4, placeholder=ui_text("en", "webui", "input_text_placeholder"))
                        btn_tts = gr.Button(ui_text("en", "webui", "btn_tts"))
                        out_audio = gr.Audio(label=ui_text("en", "webui", "out_audio_label"), type="numpy")
                        out_msg = gr.Markdown()

            with gr.TabItem("模型转换"):
                # 将模型转换界面放入独立的标签页
                conv_ui = render_converter_ui()

            with gr.TabItem("使用指引 / Guide"):
                with gr.Column():
                    svg_html = _read_text_file(LANGUAGE_SVG_PATH)
                    if svg_html:
                        gr.HTML(f"<div style='display:flex;align-items:center;gap:8px'>{svg_html}<span style='font-weight:600'>Guide Language</span></div>")
                    else:
                        gr.Markdown("🌐 Guide Language")

                    guide_lang_dd = gr.Dropdown(
                        choices=get_supported_language_display(),
                        value=code_to_display("en"),
                        label="Language",
                        interactive=True,
                    )
                    guide_md = gr.Markdown(value=get_guide_markdown("en"))

        # ------------------------------
        # Event handlers
        # ------------------------------
        def on_app_load() -> tuple:
            version = "v2"
            characters = list_character_folders(version)
            if not characters:
                return (
                    "No characters found. Please put models under Data/character_model/v2.",
                    version,
                    "",
                    "",
                    "",
                    "",
                    gr.update(choices=list_language_audio_resources("ja")),
                )

            # 不自动加载模型，让用户手动选择
            return (
                ui_text("en", "webui", "status_ready"),
                version,
                "",
                "",
                "",
                "",
                gr.update(choices=list_language_audio_resources("ja")),
            )

        demo.load(on_app_load, outputs=[status, st_version, st_character, st_loaded_character, st_ref_audio_path, st_ref_audio_text, ref_audio_dropdown])

        def on_version_change(current_character: str, loaded_character: str, new_version: str, ref_lang: str):
            """处理版本切换"""
            if not new_version:
                return (
                    "Please select version.",
                    new_version,
                    gr.update(choices=[]),
                    "",
                    "",
                    "",
                    "",
                    gr.update(choices=list_language_audio_resources(_to_lang_code(ref_lang))),
                )
            
            # 如果有当前角色，先卸载以确保重新加载（即使新版本有同名角色）
            # 优先使用已加载角色进行卸载和清理
            target_to_unload = loaded_character or current_character
            if target_to_unload:
                try:
                    lunavox.unload_character(target_to_unload)
                except Exception as e:
                    print(f"卸载角色时出错（可忽略）: {e}")
                try:
                    # 清理临时 fp32 权重文件
                    model_manager.clean_cache()
                except Exception as e:
                    print(f"清理临时权重时出错（可忽略）: {e}")
            
            characters = list_character_folders(new_version)
            if not characters:
                version_dir = "Data/character_model/v2_pro_plus" if new_version == "v2_pro_plus" else "Data/character_model/v2"
                return (
                    f"No {new_version} characters found. Put models under {version_dir}.",
                    new_version,
                    gr.update(choices=[]),
                    "",
                    "",
                    "",
                    "",
                    gr.update(choices=list_language_audio_resources(_to_lang_code(ref_lang))),
                )
            
            # 不自动加载第一个角色，让用户手动选择
            version_display = "v2 Pro Plus" if new_version == "v2_pro_plus" else "v2"
            return (
                f"Switched to {version_display}. Select a character to load.",
                new_version,
                gr.update(choices=characters, value=None),
                "",
                "",
                "",
                "",
                gr.update(choices=list_language_audio_resources(_to_lang_code(ref_lang))),
            )

        dd_version.change(
            on_version_change,
            inputs=[st_character, st_loaded_character, dd_version, ref_lang_dd],
            outputs=[status, st_version, dd_character, st_character, st_loaded_character, st_ref_audio_path, st_ref_audio_text, ref_audio_dropdown],
        )

        def on_guide_lang_change(display_lang: str):
            code = display_to_code(display_lang)
            return code, gr.update(value=get_guide_markdown(code))

        guide_lang_dd.change(
            on_guide_lang_change,
            inputs=[guide_lang_dd],
            outputs=[st_ui_lang, guide_md],
        )

        def on_load_character_click(version: str, character: str, loaded_character: str, ref_lang: str):
            """处理加载角色按钮点击"""
            if not character:
                return (
                    "请先选择一个角色。",
                    "",
                    "",
                    "",
                    gr.update(choices=list_language_audio_resources(_to_lang_code(ref_lang))),
                    loaded_character or "",
                )

            # 如果已加载的角色与将要加载的不同，先卸载并清理
            if loaded_character and loaded_character != character:
                try:
                    lunavox.unload_character(loaded_character)
                except Exception as e:
                    print(f"卸载角色时出错（可忽略）: {e}")
                try:
                    model_manager.clean_cache()
                except Exception as e:
                    print(f"清理临时权重时出错（可忽略）: {e}")
            
            msg, audio_resources = ensure_character_loaded(character, version, _to_lang_code(ref_lang))
            return msg, character, "", "", gr.update(choices=audio_resources, value=None), character

        btn_load_character.click(
            on_load_character_click,
            inputs=[st_version, dd_character, st_loaded_character, ref_lang_dd],
            outputs=[status, st_character, st_ref_audio_path, st_ref_audio_text, ref_audio_dropdown, st_loaded_character],
        )

        # 卸载当前角色并清理临时文件
        def on_unload_character_click(loaded_character: str, ref_lang: str):
            if not loaded_character:
                return (
                    "No character is currently loaded.",
                    "",
                    "",
                    "",
                    "",
                    gr.update(choices=list_language_audio_resources(_to_lang_code(ref_lang)), value=None),
                )
            try:
                lunavox.unload_character(loaded_character)
            except Exception as e:
                print(f"卸载角色时出错（可忽略）: {e}")
            try:
                model_manager.clean_cache()
            except Exception as e:
                print(f"清理临时权重时出错（可忽略）: {e}")
            return (
                "Character unloaded and temporary weights cleaned.",
                "",
                "",
                "",
                "",
                gr.update(choices=list_language_audio_resources(_to_lang_code(ref_lang)), value=None),
            )

        btn_unload_character.click(
            on_unload_character_click,
            inputs=[st_loaded_character, ref_lang_dd],
            outputs=[status, st_character, st_loaded_character, st_ref_audio_path, st_ref_audio_text, ref_audio_dropdown],
        )

        def on_character_change(current_character: str, version: str, new_char: str, ref_lang: str):
            """处理角色选择变更（仅更新状态，不自动加载）"""
            if not new_char:
                return (
                    "Please select a character.",
                    gr.update(),
                    gr.update(),
                    new_char,
                    "",
                    "",
                    gr.update(choices=list_language_audio_resources(_to_lang_code(ref_lang)), value=None),
                )
            
            audio_resources = list_language_audio_resources(_to_lang_code(ref_lang))
            
            return (
                f"Selected {new_char}. Click 'Load Character' to proceed.",
                gr.update(value=None),
                gr.update(value=""),
                new_char,
                "",
                "",
                gr.update(choices=audio_resources, value=None),
            )

        dd_character.change(
            on_character_change,
            inputs=[st_character, st_version, dd_character, ref_lang_dd],
            outputs=[status, ref_audio, ref_text, st_character, st_ref_audio_path, st_ref_audio_text, ref_audio_dropdown],
        )

        def on_ref_language_change(ref_lang: str):
            return gr.update(choices=list_language_audio_resources(_to_lang_code(ref_lang)), value=None)

        ref_lang_dd.change(
            on_ref_language_change,
            inputs=[ref_lang_dd],
            outputs=[ref_audio_dropdown],
        )
        
        # 处理参考音频下拉选择器选择
        def on_ref_audio_dropdown_change(character: str, selected_audio: Optional[str], ref_lang: str):
            if not character or not selected_audio:
                return "Please select character and reference audio.", character, "", "", gr.update(value=None), gr.update(value="")
            
            # selected_audio 现在直接是文件路径
            file_path = selected_audio
            display_name = Path(file_path).stem
            
            try:
                msg = set_reference(character, file_path, display_name, _to_lang_code(ref_lang))
                return msg, character, file_path, display_name, gr.update(value=file_path), gr.update(value=display_name)
            except Exception as e:
                return f"Failed to set reference: {e}", character, "", "", gr.update(value=None), gr.update(value="")
        
        ref_audio_dropdown.change(
            on_ref_audio_dropdown_change,
            inputs=[st_character, ref_audio_dropdown, ref_lang_dd],
            outputs=[status, st_character, st_ref_audio_path, st_ref_audio_text, ref_audio, ref_text],
        )

        # Auto set reference when audio or text changes (set only when both present)
        def on_ref_audio_change(character: str, audio_fp: Optional[str], audio_tx: str, auto_filename_enabled: bool, ref_lang: str):
            if not character:
                return "Please select a character.", character, audio_fp or "", audio_tx or "", audio_tx or ""
            
            # 如果启用了自动文件名功能且有音频文件，自动提取文件名作为文本
            if auto_filename_enabled and audio_fp and not (audio_tx or "").strip():
                try:
                    from pathlib import Path
                    audio_filename = Path(audio_fp).stem  # 获取不带扩展名的文件名
                    audio_tx = audio_filename
                except Exception as e:
                    print(f"提取文件名时出错: {e}")
            
            if audio_fp and (audio_tx or "").strip():
                try:
                    msg = set_reference(character, audio_fp, (audio_tx or "").strip(), _to_lang_code(ref_lang))
                except Exception as e:
                    msg = f"设置参考音频时出错: {e}"
            else:
                if audio_fp and not (audio_tx or "").strip():
                    msg = "Reference audio uploaded. Please enter its transcript."
                elif (audio_tx or "").strip() and not audio_fp:
                    msg = "Transcript entered. Please upload reference audio."
                else:
                    msg = "Upload reference audio and enter its transcript."
            return msg, character, audio_fp or "", audio_tx or "", audio_tx or ""

        ref_audio.change(
            on_ref_audio_change,
            inputs=[st_character, ref_audio, ref_text, auto_filename, ref_lang_dd],
            outputs=[status, st_character, st_ref_audio_path, st_ref_audio_text, ref_text],
        )

        def on_ref_text_change(character: str, audio_fp: Optional[str], audio_tx: str, ref_lang: str):
            if not character:
                return "Please select a character.", character, audio_fp or "", audio_tx or ""
            if (audio_tx or "").strip() and audio_fp:
                msg = set_reference(character, audio_fp, (audio_tx or "").strip(), _to_lang_code(ref_lang))
            else:
                if (audio_tx or "").strip() and not audio_fp:
                    msg = "Transcript entered. Please upload reference audio."
                elif audio_fp and not (audio_tx or "").strip():
                    msg = "Reference audio uploaded. Please enter its transcript."
                else:
                    msg = "Upload reference audio and enter its transcript."
            return msg, character, audio_fp or "", audio_tx or ""

        ref_text.change(
            on_ref_text_change,
            inputs=[st_character, ref_audio, ref_text, ref_lang_dd],
            outputs=[status, st_character, st_ref_audio_path, st_ref_audio_text],
        )

        def on_tts(character: str, text_val: str, lang: str):
            if not character:
                return None, "Please select a character."
            try:
                import lunavox_tts as lv
                # set current language
                from lunavox_tts.Utils.Shared import context
                lang_code = _to_lang_code(lang)
                context.current_language = lang_code
            except Exception:
                pass
            audio_path, msg = synthesize(character, text_val, lang_code)
            return audio_path, msg

        btn_tts.click(
            on_tts,
            inputs=[st_character, input_text, lang_dd],
            outputs=[out_audio, out_msg],
        )

        # i18n: apply language change across the UI (webui + converter)
        def on_ui_language_change(display_lang: str):
            code = display_to_code(display_lang)
            prompt_lang_choices = get_prompt_language_choices(code)
            prompt_lang_value = prompt_lang_choices[-1] if prompt_lang_choices else ""
            prompt_lang_code = _to_lang_code(prompt_lang_value or "ja")

            output_lang_choices = get_output_language_choices(code)
            output_lang_value = output_lang_choices[0] if output_lang_choices else ""

            audio_choices = list_language_audio_resources(prompt_lang_code)

            return [
                gr.update(label=ui_text(code, "webui", "version_label")),  # dd_version
                gr.update(label=ui_text(code, "webui", "character_label"), info=ui_text(code, "webui", "character_info")),  # dd_character
                gr.update(value=ui_text(code, "webui", "btn_load")),  # btn_load_character
                gr.update(value=ui_text(code, "webui", "status_ready")),  # status
                gr.update(value=ui_text(code, "webui", "ref_section_title")),  # ref_section_title_md
                gr.update(  # ref_audio_dropdown
                    label=ui_text(code, "webui", "preset_ref_label"),
                    info=ui_text(code, "webui", "preset_ref_info"),
                    choices=audio_choices,
                    value=None,
                ),
                gr.update(  # ref_lang_dd
                    label=ui_text(code, "webui", "ref_lang_label"),
                    choices=prompt_lang_choices,
                    value=prompt_lang_value,
                ),
                gr.update(value=ui_text(code, "webui", "or")),  # or_md
                gr.update(  # ref_audio
                    label=ui_text(code, "webui", "upload_ref_label"),
                    value=None,
                ),
                gr.update(label=ui_text(code, "webui", "auto_filename_label"), info=ui_text(code, "webui", "auto_filename_info")),  # auto_filename
                gr.update(  # ref_text
                    label=ui_text(code, "webui", "ref_text_label"),
                    placeholder=ui_text(code, "webui", "ref_text_placeholder"),
                    value="",
                ),
                gr.update(value=ui_text(code, "webui", "synth_section_title")),  # synth_section_title_md
                gr.update(  # lang_dd
                    label=ui_text(code, "webui", "output_lang_label"),
                    choices=output_lang_choices,
                    value=output_lang_value,
                ),
                gr.update(  # input_text
                    label=ui_text(code, "webui", "input_text_label"),
                    placeholder=ui_text(code, "webui", "input_text_placeholder"),
                    value="",
                ),
                gr.update(value=ui_text(code, "webui", "btn_tts")),  # btn_tts
                gr.update(  # out_audio
                    label=ui_text(code, "webui", "out_audio_label"),
                    value=None,
                ),
                gr.update(label=ui_text(code, "converter", "version_label")),  # conv_version
                gr.update(label=ui_text(code, "converter", "in_ckpt_label")),  # in_ckpt
                gr.update(label=ui_text(code, "converter", "in_pth_label")),  # in_pth
                gr.update(label=ui_text(code, "converter", "out_dir_label")),  # out_dir
                gr.update(value=ui_text(code, "converter", "btn_convert")),  # btn_convert
                gr.update(value=ui_text(code, "converter", "ready")),  # out_title
            ]

        ui_lang_dd.change(
            on_ui_language_change,
            inputs=[ui_lang_dd],
            outputs=[
                dd_version, dd_character, btn_load_character, status,
                ref_section_title_md, ref_audio_dropdown, ref_lang_dd, or_md,
                ref_audio, auto_filename, ref_text, synth_section_title_md,
                lang_dd, input_text, btn_tts, out_audio,
                conv_ui["conv_version"], conv_ui["in_ckpt"], conv_ui["in_pth"],
                conv_ui["out_dir"], conv_ui["btn_convert"], conv_ui["out_title"],
            ],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    try:
        server_port = _pick_server_port()
    except RuntimeError as exc:
        logger.error("Failed to start WebUI: %s", exc)
        raise SystemExit(1) from exc
    app.launch(
        server_name="127.0.0.1",
        server_port=server_port,
        inbrowser=True,
        show_api=False,
        allowed_paths=[str(OUTPUT_DIR), str(AUDIO_RESOURCES_DIR)],
    )


