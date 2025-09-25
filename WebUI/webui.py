import os
import json
import time
from pathlib import Path
import sys
from typing import List, Optional, Tuple

# Import LunaVox TTS public APIs (support running from repo without installation)
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent  # Go up one level from WebUI to repo root
REPO_SRC = REPO_ROOT / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))
import lunavox_tts as lunavox
import gradio as gr


# ------------------------------
# Paths and environment setup
# ------------------------------
DATA_DIR = REPO_ROOT / "Data"
CHAR_MODEL_DIR = DATA_DIR / "character_model"
AUDIO_RESOURCES_DIR = DATA_DIR / "audio_resources"
OUTPUT_DIR = REPO_ROOT / "Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Prefer local dependencies to avoid downloads
os.environ.setdefault("HUBERT_MODEL_PATH", str(DATA_DIR / "chinese-hubert-base.onnx"))
os.environ.setdefault("OPEN_JTALK_DICT_DIR", str(DATA_DIR / "open_jtalk_dic_utf_8-1.11"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


# ------------------------------
# Utilities
# ------------------------------
def list_character_folders() -> List[str]:
    if not CHAR_MODEL_DIR.exists():
        return []
    folders = [p.name for p in CHAR_MODEL_DIR.iterdir() if p.is_dir()]
    folders.sort()
    return folders


def get_model_dir(character_name: str) -> Path:
    return CHAR_MODEL_DIR / character_name


def list_reference_audio_resources(character_name: str) -> List[Tuple[str, str]]:
    """
    搜索指定角色的参考音频资源
    
    Args:
        character_name: 角色名称
        
    Returns:
        List of (display_name, choice_value) tuples for Gradio dropdown
        display_name: 只显示文件名（不含扩展名）
        choice_value: 内部使用的完整路径
    """
    if not AUDIO_RESOURCES_DIR.exists():
        return []
    
    # 转换为小写进行搜索，避免大小写问题
    character_name_lower = character_name.lower()
    
    # 查找匹配的文件夹
    matching_folders = []
    for folder in AUDIO_RESOURCES_DIR.iterdir():
        if folder.is_dir() and folder.name.lower() == character_name_lower:
            matching_folders.append(folder)
    
    if not matching_folders:
        return []
    
    # 收集所有wav文件
    audio_files = []
    for folder in matching_folders:
        for audio_file in folder.glob("*.wav"):
            # 使用文件名（不含扩展名）作为显示名称
            display_name = audio_file.stem
            # 选择值就是文件路径
            choice_value = str(audio_file)
            audio_files.append((display_name, choice_value))
    
    # 按文件名排序
    audio_files.sort(key=lambda x: x[0])
    return audio_files


def load_default_prompt(character_name: str) -> Tuple[Optional[str], Optional[str], str]:
    # Default prompt loading is disabled per user requirement.
    return None, None, ""


def ensure_character_loaded(character_name: str) -> Tuple[str, List[Tuple[str, str]]]:
    model_dir = get_model_dir(character_name)
    lunavox.load_character(character_name, str(model_dir))
    
    # 同时搜索参考音频资源
    audio_resources = list_reference_audio_resources(character_name)
    
    if audio_resources:
        return f"角色 {character_name} 模型已加载，找到 {len(audio_resources)} 个参考音频资源。", audio_resources
    else:
        return f"角色 {character_name} 模型已加载。", []


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


def synthesize(character_name: str, text: str, language: str) -> Tuple[Optional[str], str]:
    if not text or not text.strip():
        return None, "请输入要合成的文本。"

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = OUTPUT_DIR / f"{character_name}_{timestamp}.wav"

    # Do not play on host machine (web-only playback)
    lunavox.tts(
        character_name=character_name,
        text=text.strip(),
        play=False,
        split_sentence=True,
        save_path=str(save_path),
        language=language,
    )
    if save_path.exists():
        return str(save_path), f"合成完成：{save_path}"
    return None, "合成失败，请检查日志。"


# ------------------------------
# Gradio UI
# ------------------------------
def build_ui() -> gr.Blocks:
    with gr.Blocks(css="footer {visibility: hidden}") as demo:
        gr.Markdown("""
        **LunaVox 本地 WebUI**  
        - 自动扫描 `Data/character_model` 下的角色模型  
        - 上传参考音频与文本，一键合成日文语音  
        - 生成音频保存在 `Output` 目录下，并可在线试听
        """)

        with gr.Row():
            with gr.Column(scale=1):
                character_list = list_character_folders()
                default_character = character_list[0] if character_list else ""
                dd_character = gr.Dropdown(
                    choices=character_list,
                    value=default_character,
                    label="角色选择",
                    interactive=True,
                )
                status = gr.Markdown("准备就绪。")

                # States
                st_character = gr.State(default_character)
                st_ref_audio_path = gr.State("")
                st_ref_audio_text = gr.State("")

            with gr.Column(scale=2):
                gr.Markdown("### 参考音频")
                ref_lang_dd = gr.Dropdown(
                    choices=["auto", "ja", "en", "zh"],
                    value="ja",
                    label="参考音频语言 / Prompt Language",
                    interactive=True,
                )
                
                # 参考音频资源下拉选择器
                ref_audio_dropdown = gr.Dropdown(
                    label="预设参考音频",
                    choices=[],
                    value=None,
                    interactive=True,
                    allow_custom_value=False,
                    info="从Data/audio_resources中选择预设的参考音频"
                )
                
                gr.Markdown("**或**")
                
                ref_audio = gr.Audio(
                    label="上传参考音频",
                    sources=["upload"],
                    type="filepath",
                )
                auto_filename = gr.Checkbox(
                    label="自动使用文件名作为参考文本",
                    value=True,
                    info="勾选后，上传音频文件时会自动将文件名（去除后缀）作为参考文本"
                )
                ref_text = gr.Textbox(label="参考音频文本", lines=2, placeholder="请输入与参考音频匹配的日文文本")

                gr.Markdown("### 文本合成")
                lang_dd = gr.Dropdown(choices=["ja", "en", "zh"], value="ja", label="输出语言 / Output Language", interactive=True)
                input_text = gr.Textbox(label="输入文本", lines=4, placeholder="请输入要合成的文本（ja/en）")
                btn_tts = gr.Button("开始合成")
                out_audio = gr.Audio(label="合成结果试听", type="filepath")
                out_msg = gr.Markdown()

        # ------------------------------
        # Event handlers
        # ------------------------------
        def on_app_load() -> tuple:
            characters = list_character_folders()
            if not characters:
                return "未找到任何角色模型，请将模型放入 Data/character_model 下。", "", "", "", gr.update(choices=[])

            character = characters[0]
            msg_load, audio_resources = ensure_character_loaded(character)
            return (msg_load, character, "", "", gr.update(choices=audio_resources))

        demo.load(on_app_load, outputs=[status, st_character, st_ref_audio_path, st_ref_audio_text, ref_audio_dropdown])

        def on_character_change(new_char: str):
            if not new_char:
                return "请选择角色。", gr.update(), gr.update(), new_char, "", "", gr.update(choices=[])
            msg, audio_resources = ensure_character_loaded(new_char)
            return (
                msg,
                gr.update(value=None),
                gr.update(value=""),
                new_char,
                "",
                "",
                gr.update(choices=audio_resources),
            )

        dd_character.change(
            on_character_change,
            inputs=[dd_character],
            outputs=[status, ref_audio, ref_text, st_character, st_ref_audio_path, st_ref_audio_text, ref_audio_dropdown],
        )
        
        # 处理参考音频下拉选择器选择
        def on_ref_audio_dropdown_change(character: str, selected_audio: Optional[str], ref_lang: str):
            if not character or not selected_audio:
                return "请选择角色和参考音频。", character, "", "", gr.update(value=None), gr.update(value="")
            
            # selected_audio 现在直接是文件路径
            file_path = selected_audio
            display_name = Path(file_path).stem
            
            try:
                msg = set_reference(character, file_path, display_name, ref_lang)
                return msg, character, file_path, display_name, gr.update(value=file_path), gr.update(value=display_name)
            except Exception as e:
                return f"设置参考音频时出错: {e}", character, "", "", gr.update(value=None), gr.update(value="")
        
        ref_audio_dropdown.change(
            on_ref_audio_dropdown_change,
            inputs=[st_character, ref_audio_dropdown, ref_lang_dd],
            outputs=[status, st_character, st_ref_audio_path, st_ref_audio_text, ref_audio, ref_text],
        )

        # Auto set reference when audio or text changes (set only when both present)
        def on_ref_audio_change(character: str, audio_fp: Optional[str], audio_tx: str, auto_filename_enabled: bool, ref_lang: str):
            if not character:
                return "请选择角色。", character, audio_fp or "", audio_tx or "", audio_tx or ""
            
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
                    msg = set_reference(character, audio_fp, (audio_tx or "").strip(), ref_lang)
                except Exception as e:
                    msg = f"设置参考音频时出错: {e}"
            else:
                if audio_fp and not (audio_tx or "").strip():
                    msg = "已上传参考音频，请填写对应文本以完成设置。"
                elif (audio_tx or "").strip() and not audio_fp:
                    msg = "已填写参考文本，请上传参考音频以完成设置。"
                else:
                    msg = "请上传参考音频并填写对应文本。"
            return msg, character, audio_fp or "", audio_tx or "", audio_tx or ""

        ref_audio.change(
            on_ref_audio_change,
            inputs=[st_character, ref_audio, ref_text, auto_filename, ref_lang_dd],
            outputs=[status, st_character, st_ref_audio_path, st_ref_audio_text, ref_text],
        )

        def on_ref_text_change(character: str, audio_fp: Optional[str], audio_tx: str, ref_lang: str):
            if not character:
                return "请选择角色。", character, audio_fp or "", audio_tx or ""
            if (audio_tx or "").strip() and audio_fp:
                msg = set_reference(character, audio_fp, (audio_tx or "").strip(), ref_lang)
            else:
                if (audio_tx or "").strip() and not audio_fp:
                    msg = "已填写参考文本，请上传参考音频以完成设置。"
                elif audio_fp and not (audio_tx or "").strip():
                    msg = "已上传参考音频，请填写对应文本以完成设置。"
                else:
                    msg = "请上传参考音频并填写对应文本。"
            return msg, character, audio_fp or "", audio_tx or ""

        ref_text.change(
            on_ref_text_change,
            inputs=[st_character, ref_audio, ref_text, ref_lang_dd],
            outputs=[status, st_character, st_ref_audio_path, st_ref_audio_text],
        )

        def on_tts(character: str, text_val: str, lang: str):
            if not character:
                return None, "请选择角色。"
            try:
                import lunavox_tts as lv
                # set current language
                from lunavox_tts.Utils.Shared import context
                context.current_language = lang if lang in ["ja", "en", "zh"] else "ja"
            except Exception:
                pass
            audio_path, msg = synthesize(character, text_val, lang)
            return audio_path, msg

        btn_tts.click(
            on_tts,
            inputs=[st_character, input_text, lang_dd],
            outputs=[out_audio, out_msg],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, show_api=False)


