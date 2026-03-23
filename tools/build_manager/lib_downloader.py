import os
import shutil
import zipfile
import tarfile
import urllib.request
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn
from rich.console import Console

console = Console()

# URLs
LIB_URLS = {
    "onnx": {
        "win_cpu": "https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/onnxruntime-win-x64-1.24.4.zip",
        "win_cuda": "https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/onnxruntime-win-x64-gpu_cuda13-1.24.4.zip",
        "macos": "https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/onnxruntime-osx-arm64-1.24.4.tgz"
    },
    "llama": {
        "win_cpu": "https://github.com/ggml-org/llama.cpp/releases/download/b8470/llama-b8470-bin-win-cpu-x64.zip",
        "win_vulkan": "https://github.com/ggml-org/llama.cpp/releases/download/b8470/llama-b8470-bin-win-vulkan-x64.zip",
        "win_cuda": "https://github.com/ggml-org/llama.cpp/releases/download/b8470/llama-b8470-bin-win-cuda-13.1-x64.zip",
        "macos": "https://github.com/ggml-org/llama.cpp/releases/download/b8470/llama-b8470-bin-macos-arm64.tar.gz",
        "ios": "https://github.com/ggml-org/llama.cpp/releases/download/b8470/llama-b8470-xcframework.zip"
    }
}

def download_file(url, target_path):
    with Progress(
        TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        DownloadColumn(),
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
    ) as progress:
        filename = url.split("/")[-1]
        task_id = progress.add_task("download", filename=filename, total=None)
        
        def report(count, block_size, total_size):
            if progress.tasks[task_id].total is None and total_size > 0:
                progress.update(task_id, total=total_size)
            progress.update(task_id, completed=count * block_size)
            
        urllib.request.urlretrieve(url, target_path, reporthook=report)

def extract_archive(archive_path, extract_dir):
    with console.status(f"[bold cyan]Extracting {archive_path.name}...") as status:
        if str(archive_path).endswith(".zip"):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif str(archive_path).endswith((".tar.gz", ".tgz")):
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_dir)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path}")

def update_library(lib_name, backend, root_dir):
    if lib_name not in LIB_URLS:
        raise ValueError(f"Unknown library: {lib_name}")
    
    if backend not in LIB_URLS[lib_name]:
        backends = ", ".join(LIB_URLS[lib_name].keys())
        raise ValueError(f"Unknown backend '{backend}' for {lib_name}. Available: {backends}")
    
    url = LIB_URLS[lib_name][backend]
    lib_root = Path(root_dir) / "lib"
    target_dir = lib_root / lib_name
    temp_dir = lib_root / f"temp_{lib_name}"
    archive_name = url.split("/")[-1]
    archive_path = lib_root / archive_name
    
    # Ensure lib root exists
    lib_root.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Download
        download_file(url, archive_path)
        
        # 2. Extract
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        extract_archive(archive_path, temp_dir)
        
        # 3. Handle structure (many archives have a root folder)
        extracted_items = list(temp_dir.glob("*"))
        if len(extracted_items) == 1 and extracted_items[0].is_dir():
            source_dir = extracted_items[0]
        else:
            source_dir = temp_dir
            
        console.print(f"[bold green]Installing {lib_name} to {target_dir}...")
        
        # 4. Remove old files
        if target_dir.exists():
            shutil.rmtree(target_dir)
            
        # 5. Move/Copy to target
        shutil.copytree(source_dir, target_dir)
        
        console.print(f"[bold green]✔[/] [white]Lib {lib_name} updated successfully at[/] [cyan]{target_dir}[/]")
        
    finally:
        # Cleanup
        if archive_path.exists():
            archive_path.unlink()
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python lib_downloader.py <onnx|llama> <backend> [root_dir]")
        sys.exit(1)
    
    lib = sys.argv[1]
    backend = sys.argv[2]
    root = sys.argv[3] if len(sys.argv) > 3 else "."
    
    update_library(lib, backend, root)
