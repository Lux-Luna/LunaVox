import sys, os
from pathlib import Path

# Add 'src' to path to access lunavox_tts
sys.path.append(str(Path(__file__).parents[1] / "src"))

try:
    from lunavox_tts import convert_to_onnx
    from lunavox_tts.Converter.v2.Converter import find_ckpt_and_pth
except ImportError:
    print("Error: lunavox_tts not found. Run from the project root.")
    sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python pytorch_to_onnx_bin_demo.py <input_dir> <output_root> <character_name>")
        sys.exit(1)

    input_dir, output_root, name = sys.argv[1:4]
    ckpt, pth = find_ckpt_and_pth(input_dir)
    
    if ckpt and pth:
        # Note: The converter automatically detects the model version (v2 or v2ProPlus)
        # based on the configuration stored inside the .pth file.
        print(f"Converting '{name}' from {input_dir}...")
        convert_to_onnx(ckpt, pth, os.path.join(output_root, name))
    else:
        print("Error: Required .ckpt and .pth files not found in the input directory.")
