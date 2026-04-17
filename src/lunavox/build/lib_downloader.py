import os
import shutil
import zipfile
import tarfile
import urllib.request
import platform
import time
import json
from pathlib import Path
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn

from lunavox.core.ui import console

# Load URLs from libs.json
LIBS_CONFIG_PATH = Path(__file__).parent / "libs.json"

def load_libs_config():
    with open(LIBS_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

LIBS_CONFIG = load_libs_config()
LIB_URLS = LIBS_CONFIG["libraries"]

def download_file(url, target_path):
    with Progress(
        TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "|",
        DownloadColumn(),
        "|",
        TransferSpeedColumn(),
        "|",
        TimeRemainingColumn(),
    ) as progress:
        filename = url.split("/")[-1]
        task_id = progress.add_task("download", filename=filename, total=None)
        
        def report(count, block_size, total_size):
            if progress.tasks[task_id].total is None and total_size > 0:
                progress.update(task_id, total=total_size)
            progress.update(task_id, completed=count * block_size)
            
        urllib.request.urlretrieve(url, target_path, reporthook=report)

def flatten_nuget_structure(temp_dir: Path):
    """
    NuGet (nupkg) structure is typically:
    - build/native/include/ -> include/
    - runtimes/win-x64/native/ -> lib/
    We want to flatten it to match the standard expected structure.
    """
    console.print("[dim]Flattening NuGet package structure...[/]")
    
    # 1. Look for include folders
    potential_includes = list(temp_dir.glob("**/include"))
    if potential_includes:
        target_inc = temp_dir / "include"
        if target_inc != potential_includes[0]:
             target_inc.mkdir(exist_ok=True, parents=True)
             for item in potential_includes[0].glob("*"):
                 # Using copy/remove or shutil.move; move is better
                 dest = target_inc / item.name
                 if dest.exists() and dest.is_dir(): shutil.rmtree(dest)
                 shutil.move(str(item), str(target_inc))

    # 2. Look for DLLs and LIBS
    target_lib = temp_dir / "lib"
    target_lib.mkdir(exist_ok=True, parents=True)
    
    # Common runtime paths in NuGet for ONNX Runtime / DirectML
    # Note: we prioritize win-x64 for this specific project
    runtime_paths = [
        temp_dir / "runtimes" / "win-x64" / "native",
        temp_dir / "build" / "native" / "lib" / "x64",
        temp_dir / "build" / "native" / "bin" / "x64",
    ]
    
    found_binaries = False
    for rp in runtime_paths:
        if rp.exists():
            for item in rp.glob("*"):
                if item.suffix.lower() in (".dll", ".lib", ".pdb", ".so", ".dylib", ".a"):
                    dest = target_lib / item.name
                    if dest.exists(): dest.unlink()
                    shutil.move(str(item), str(target_lib))
                    found_binaries = True
    
    # 3. Cleanup original NuGet metadata/folders to keep target clean
    folders_to_clean = ["build", "runtimes", "_rels", "package"]
    for folder in folders_to_clean:
        p = temp_dir / folder
        if p.exists():
            shutil.rmtree(p)
    
    for p in temp_dir.glob("*.nuspec"): p.unlink()
    for p in temp_dir.glob("[Content_Types].xml"): p.unlink()
    
    console.print("[dim]NuGet structure flattened successfully.[/]")

def extract_archive(archive_path, extract_dir):
    with console.status(f"[bold cyan]Extracting {archive_path.name}...") as status:
        if str(archive_path).endswith((".zip", ".nupkg")):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif str(archive_path).endswith((".tar.gz", ".tgz")):
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_dir)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path}")

def install_local_archive(lib_name, archive_path, root_dir, *, label=None, version="custom"):
    """Install a backend from an archive that's already on disk.

    Used by the GUI's drag-and-drop / browse path so users can stage a
    locally-built or differently-versioned llama.cpp / onnxruntime tree
    without going through the URL table in libs.json.

    ``label`` is what gets written into ``lib/metadata.json`` (e.g.
    ``"vulkan"`` or ``"DmlExecutionProvider"``); when omitted we record
    the archive filename as the label so the source is at least visible.
    """
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    suffix = archive_path.suffix.lower()
    if suffix not in {".zip", ".nupkg", ".tgz"} and not str(archive_path).endswith(".tar.gz"):
        raise ValueError(f"Unsupported archive format: {archive_path.name}")

    lib_root = Path(root_dir) / "lib"
    target_dir = lib_root / lib_name
    temp_dir = lib_root / f"temp_{lib_name}_local"
    lib_root.mkdir(parents=True, exist_ok=True)

    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        extract_archive(archive_path, temp_dir)

        if suffix == ".nupkg":
            flatten_nuget_structure(temp_dir)

        extracted_items = list(temp_dir.glob("*"))
        if len(extracted_items) == 1 and extracted_items[0].is_dir():
            source_dir = extracted_items[0]
        else:
            source_dir = temp_dir

        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)

        metadata_file = lib_root / "metadata.json"
        all_metadata = {}
        if metadata_file.exists():
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    all_metadata = json.load(f)
            except Exception:
                pass

        resolved_label = label or archive_path.stem
        if lib_name == "onnx":
            # Locally-staged archives don't carry a structured capability list;
            # treat the single label as the primary EP and universally keep
            # CPU as the guaranteed fallback.
            caps = [resolved_label]
            if resolved_label != "CPUExecutionProvider":
                caps.append("CPUExecutionProvider")
            all_metadata[lib_name] = {
                "provider": caps[0],
                "build_capabilities": caps,
                "version": version,
                "platform": platform.system().lower(),
                "installed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        else:
            all_metadata[lib_name] = {
                "backend": resolved_label,
                "version": version,
                "platform": platform.system().lower(),
                "installed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, indent=2)

        console.print(
            f"[bold green]✔[/] [white]Lib {lib_name} installed from local archive at[/] "
            f"[cyan]{target_dir}[/]"
        )

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def update_library(lib_name, backend, root_dir):
    if lib_name not in LIB_URLS:
        raise ValueError(f"Unknown library: {lib_name}")
    
    if backend not in LIB_URLS[lib_name]["backends"]:
        backends = ", ".join(LIB_URLS[lib_name]["backends"].keys())
        raise ValueError(f"Unknown backend '{backend}' for {lib_name}. Available: {backends}")
    
    url = LIB_URLS[lib_name]["backends"][backend]
    lib_root = Path(root_dir) / "lib"
    target_dir = lib_root / lib_name
    temp_dir = lib_root / f"temp_{lib_name}"
    
    # Improved filename extraction
    archive_name = url.split("/")[-1].split("?")[0]
    if not any(archive_name.endswith(ext) for ext in [".zip", ".nupkg", ".tgz", ".tar.gz"]):
        # Fallback if URL doesn't have extension (e.g. NuGet v2 API)
        if "nuget.org" in url or "package" in url:
            archive_name += ".nupkg"
        else:
            archive_name += ".zip"
            
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
        
        # 3. Flatten NuGet if applicable
        if archive_path.suffix.lower() == ".nupkg":
            flatten_nuget_structure(temp_dir)
        
        # 4. Handle structure (many archives have a root folder)
        extracted_items = list(temp_dir.glob("*"))
        if len(extracted_items) == 1 and extracted_items[0].is_dir():
            source_dir = extracted_items[0]
        else:
            source_dir = temp_dir
            
        console.print(f"[bold green]Installing {lib_name} to {target_dir}...")
        
        # 5. Remove old files
        if target_dir.exists():
            shutil.rmtree(target_dir)
            
        # 6. Move/Copy to target
        shutil.copytree(source_dir, target_dir)
        
        # 7. Update Global Metadata
        metadata_file = lib_root / "metadata.json"
        all_metadata = {}
        if metadata_file.exists():
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    all_metadata = json.load(f)
            except Exception:
                pass
        
        # Determine explicit provider/backend for C++ engine.
        # For ONNX we now emit an ordered capability list: the primary EP
        # plus a CPU fallback so the runtime can gracefully degrade when
        # an accelerated EP fails to initialize (e.g. CoreML partitioner
        # bug on certain fp16 attention graphs). GUI code still reads the
        # single "provider" field, so we preserve it as the list's head.
        if lib_name == "onnx":
            onnx_caps_mapping = {
                "win_cpu":       ["CPUExecutionProvider"],
                "win_cuda12":    ["CUDAExecutionProvider", "CPUExecutionProvider"],
                "win_cuda13":    ["CUDAExecutionProvider", "CPUExecutionProvider"],
                "win_directml":  ["DmlExecutionProvider", "CPUExecutionProvider"],
                "linux_cpu":     ["CPUExecutionProvider"],
                "linux_cuda12":  ["CUDAExecutionProvider", "CPUExecutionProvider"],
                "macos_arm64":   ["CoreMLExecutionProvider", "CPUExecutionProvider"],
            }
            caps = onnx_caps_mapping.get(backend, ["CPUExecutionProvider"])
            all_metadata[lib_name] = {
                "provider": caps[0],
                "build_capabilities": caps,
                "version": LIB_URLS[lib_name].get("version", "unknown"),
                "platform": platform.system().lower(),
                "installed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        else:
            llama_mapping = {
                "win_cpu": "cpu",
                "win_cuda12": "cuda",
                "win_cuda13": "cuda",
                "win_vulkan": "vulkan",
                "linux_cpu": "cpu",
                "linux_cuda12": "cuda",
                "macos_arm64": "metal",
            }
            backend_name = llama_mapping.get(backend, "cpu")
            all_metadata[lib_name] = {
                "backend": backend_name,
                "version": LIB_URLS[lib_name].get("version", "unknown"),
                "platform": platform.system().lower(),
                "installed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, indent=2)
        
        console.print(f"[bold green]✔[/] [white]Lib {lib_name} updated successfully at[/] [cyan]{target_dir}[/]")
        console.print(f"[dim]Updated global metadata at {metadata_file}[/]")
        
    finally:
        # Cleanup
        if archive_path.exists():
            archive_path.unlink()
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

def download_platform_libs(platform_key, root_dir):
    config = LIBS_CONFIG
    platforms = config.get("platforms", {})
    if platform_key not in platforms:
        raise ValueError(f"Unknown platform: {platform_key}. Available: {', '.join(platforms.keys())}")
    
    platform = platforms[platform_key]
    console.print(f"[bold blue]Starting download for platform: {platform['display_name']}[/]")
    
    for lib_name, backend in platform["components"].items():
        console.print(f"\n[bold cyan]=== Updating {lib_name} ({backend}) ===[/]")
        update_library(lib_name, backend, root_dir)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python lib_downloader.py <onnx|llama> <backend> [root_dir]")
        sys.exit(1)
    
    lib = sys.argv[1]
    backend = sys.argv[2]
    root = sys.argv[3] if len(sys.argv) > 3 else "."
    
    update_library(lib, backend, root)
