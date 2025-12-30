#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
转换 v2pp_pretrained 模型为 ONNX 和 BIN 格式
"""
import os
import sys
from pathlib import Path

# 添加 src 到路径
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR
PARENT_DIR = REPO_ROOT.parent  # E:\Lunavox
sys.path.insert(0, str(REPO_ROOT / "src"))

import lunavox_tts as lunavox

# 输入路径（Base_model 在父目录）
BASE_MODEL_DIR = PARENT_DIR / "Base_model" / "v2pp_pretrained"
CKPT_FILE = BASE_MODEL_DIR / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
PTH_FILE = BASE_MODEL_DIR / "s2Gv2ProPlus.pth"

# 输出路径
OUTPUT_DIR = REPO_ROOT / "Data" / "character_model" / "v2_pro_plus" / "pretrained"

def main():
    print("=" * 60)
    print("开始转换 v2pp_pretrained 模型")
    print("=" * 60)
    
    # 检查输入文件
    if not CKPT_FILE.exists():
        print(f"[ERROR] 错误: 找不到 .ckpt 文件: {CKPT_FILE}")
        sys.exit(1)
    
    if not PTH_FILE.exists():
        print(f"[ERROR] 错误: 找不到 .pth 文件: {PTH_FILE}")
        sys.exit(1)
    
    print(f"[OK] 找到 .ckpt 文件: {CKPT_FILE}")
    print(f"[OK] 找到 .pth 文件: {PTH_FILE}")
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] 输出目录: {OUTPUT_DIR}")
    
    # 执行转换
    print("\n开始转换...")
    try:
        lunavox.convert_to_onnx(
            torch_ckpt_path=str(CKPT_FILE),
            torch_pth_path=str(PTH_FILE),
            output_dir=str(OUTPUT_DIR),
        )
        print("\n" + "=" * 60)
        print("[SUCCESS] 转换完成!")
        print(f"模型已保存到: {OUTPUT_DIR}")
        print("=" * 60)
    except Exception as e:
        print(f"\n[ERROR] 转换失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

