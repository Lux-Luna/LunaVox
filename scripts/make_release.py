"""Build the three release zips for a LunaVox tagged release.

Subcommands
-----------
zip-backend --platform {win_cpu|win_vulkan}
    Rebuild the native engine + backend DLLs for the given platform key
    and package them as ``LunaVox-v<version>-win-x64-<tag>.zip``.

zip-bundle
    Download python-build-standalone, pip-install the locally built
    ``lunavox`` wheel into it, and package as
    ``lunavox-v<version>-cli-gui-bundle.zip``.

The script does NOT call ``gh``. Publish manually with:

    gh release create v<version> dist/*.zip --title "..." --notes-file ...
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Pins — update here when bumping python-build-standalone.
# ---------------------------------------------------------------------------

PBS_TAG = "20260414"
PBS_PY = "3.11.15"
PBS_ASSET = f"cpython-{PBS_PY}+{PBS_TAG}-x86_64-pc-windows-msvc-install_only.tar.gz"
PBS_URL = (
    f"https://github.com/astral-sh/python-build-standalone/releases/download/"
    f"{PBS_TAG}/{PBS_ASSET}"
)

# ``lib/metadata.json`` strings written by ``lunavox build libs``.
PLATFORM_META = {
    "win_cpu": {"onnx_provider": "CPUExecutionProvider", "llama_backend": "cpu"},
    "win_vulkan": {"onnx_provider": "DmlExecutionProvider", "llama_backend": "vulkan"},
}
PLATFORM_TAG = {"win_cpu": "cpu", "win_vulkan": "vulkan-dml"}

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
STAGE = DIST / "stage"
CACHE = DIST / "_cache"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("version"):
            return s.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("Could not find version in pyproject.toml")


def run(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=cwd, env=env)


def rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def already_matches(platform_key: str) -> bool:
    """Return True if lib/metadata.json + build/lunavox.dll already reflect
    a complete build for ``platform_key`` — lets us skip win_vulkan rebuild
    when the dev checkout is already on that backend."""
    meta_path = ROOT / "lib" / "metadata.json"
    dll_path = ROOT / "build" / "lunavox.dll"
    if not meta_path.exists() or not dll_path.exists():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = PLATFORM_META[platform_key]
    onnx = (meta.get("onnx") or {}).get("provider")
    llama = (meta.get("llama") or {}).get("backend")
    return onnx == expected["onnx_provider"] and llama == expected["llama_backend"]


def run_lunavox(*args: str) -> None:
    """Invoke the lunavox CLI using whatever Python is running this script."""
    env = os.environ.copy()
    # Force UTF-8 so rich's unicode glyphs (e.g. the checkmark it prints on
    # a successful lib download) don't crash on a GBK-default Windows cmd.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    run([sys.executable, "-m", "lunavox", *args], cwd=ROOT, env=env)


def make_zip(src: Path, out: Path) -> None:
    """Zip ``src`` (a directory) to ``out.zip`` (no extra outer folder)."""
    if out.exists():
        out.unlink()
    print(f"$ zip {src} -> {out}")
    base = out.with_suffix("")
    shutil.make_archive(str(base), "zip", root_dir=src.parent, base_dir=src.name)


# ---------------------------------------------------------------------------
# Zip A / Zip B — compiled backend
# ---------------------------------------------------------------------------


# Extensions / names we strip from build/ when packaging.
BUILD_EXCLUDE_SUFFIXES = {".exp", ".lib", ".ilk", ".pdb"}
BUILD_EXCLUDE_NAMES = {
    "build.ninja",
    ".ninja_deps",
    ".ninja_log",
    "cmake_install.cmake",
    "CMakeCache.txt",
    "CTestTestfile.cmake",
    "DartConfiguration.tcl",
}
BUILD_EXCLUDE_DIRS = {"CMakeFiles", "Testing", "bdist.win-amd64", "lib"}


def _copy_build_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        if entry.is_dir():
            if entry.name in BUILD_EXCLUDE_DIRS:
                continue
            _copy_build_tree(entry, dst / entry.name)
        else:
            if entry.name in BUILD_EXCLUDE_NAMES:
                continue
            if entry.suffix in BUILD_EXCLUDE_SUFFIXES:
                continue
            shutil.copy2(entry, dst / entry.name)


def _backend_readme(platform_key: str, ver: str) -> str:
    label = "CPU" if platform_key == "win_cpu" else "Vulkan + DirectML"
    return f"""# LunaVox v{ver} — Windows x64 ({label}) build

Pre-compiled C++ engine (`lunavox.dll`, `lunavox-cli.exe`) plus the matching
{label} backend DLLs. You still need the Python CLI/GUI to drive it — either:

- `pip install lunavox`, then `cd` into this folder and run
  `lunavox synth "hello" -o out.wav`, **or**
- Download `lunavox-v{ver}-cli-gui-bundle.zip` and extract **this** folder's
  contents into its root so `build/` and `lib/` land side-by-side with
  `python/` and `lunavox.bat`.

## Quick smoke test (no Python needed)

    build\\lunavox-cli.exe --help

## Next steps

- `lunavox model pull` to fetch the default Qwen3-TTS model.
- `lunavox gui` for the desktop app, or `lunavox serve` for HTTP/WebSocket.
"""


def zip_backend(platform_key: str, *, force_rebuild: bool = False) -> Path:
    if platform_key not in PLATFORM_META:
        raise SystemExit(f"Unknown --platform {platform_key!r}; pick one of {list(PLATFORM_META)}")

    ver = version()
    tag = PLATFORM_TAG[platform_key]

    need_rebuild = force_rebuild or not already_matches(platform_key)
    if need_rebuild:
        print(f"[backend] rebuilding for {platform_key} …")
        run_lunavox("build", "libs", "--platform", platform_key)
        run_lunavox("build", "--clean", "--j", str(os.cpu_count() or 4))
    else:
        print(f"[backend] existing build/ already matches {platform_key}, skipping rebuild")

    # ------ stage ------
    stage_name = f"LunaVox-v{ver}-win-x64-{tag}"
    stage_dir = STAGE / stage_name
    rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    # build/ — trimmed
    _copy_build_tree(ROOT / "build", stage_dir / "build")

    # lib/ — raw
    shutil.copytree(ROOT / "lib", stage_dir / "lib")

    # marker + readme
    (stage_dir / ".lunavox-root").write_text(f"lunavox v{ver} {tag}\n", encoding="utf-8")
    write(stage_dir / "README.md", _backend_readme(platform_key, ver))

    out_zip = DIST / f"{stage_name}.zip"
    make_zip(stage_dir, out_zip)
    rmtree(stage_dir)
    size_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"[backend] {out_zip}  ({size_mb:.1f} MB)")
    return out_zip


# ---------------------------------------------------------------------------
# Zip C — embedded-Python CLI+GUI bundle
# ---------------------------------------------------------------------------


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[bundle] cached: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[bundle] downloading {url}")
    # curl is always on Windows 10+/11.
    run(["curl", "-L", "--fail", "--progress-bar", "-o", str(dest), url])


def _extract_pbs(tarball: Path, dest: Path) -> None:
    rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[bundle] extracting {tarball.name} -> {dest}")
    with tarfile.open(tarball, "r:gz") as tar:
        # python-build-standalone tarballs contain a top-level "python/" dir.
        # Extract into dest.parent so the top folder lands as "python".
        # If the tarball's top dir isn't exactly "python", rename.
        tar.extractall(path=dest.parent)
    # find the extracted dir (usually "python")
    candidates = [p for p in dest.parent.iterdir() if p.is_dir() and p.name != dest.name]
    if not (dest.parent / "python").exists():
        # python-build-standalone ships as "python/" inside — so (dest.parent/"python") is it
        # If it was named something else, rename it.
        extracted = None
        for p in candidates:
            if (p / "python.exe").exists():
                extracted = p
                break
        if extracted is not None and extracted != dest:
            extracted.rename(dest)
    else:
        (dest.parent / "python").rename(dest) if dest.parent / "python" != dest else None


def _prune_site_packages(site_packages: Path) -> None:
    """Drop tests and __pycache__ to shrink the bundle."""
    removed = 0
    for path in list(site_packages.rglob("__pycache__")):
        shutil.rmtree(path, ignore_errors=True)
        removed += 1
    for pattern in ("tests", "test"):
        for path in site_packages.iterdir():
            if not path.is_dir():
                continue
            nested = path / pattern
            if nested.is_dir():
                shutil.rmtree(nested, ignore_errors=True)
                removed += 1
    print(f"[bundle] pruned {removed} cache/test dirs from site-packages")


def _bundle_readme(ver: str) -> str:
    return f"""# LunaVox v{ver} — CLI + GUI bundle (no Python install required)

Self-contained Python 3.11 with LunaVox and all GUI / server dependencies
pre-installed. You still need **one of** these backend zips, extracted
**into this folder**:

- `LunaVox-v{ver}-win-x64-cpu.zip`         (CPU only — universal, slower)
- `LunaVox-v{ver}-win-x64-vulkan-dml.zip`  (GPU — recommended; NVIDIA, AMD, Intel)

After extracting a backend zip here, you should see `build\\lunavox.dll`
and `lib\\llama\\`, `lib\\onnx\\` sitting next to `python\\` and
`lunavox.bat`.

## First run

    lunavox.bat model pull                # fetch the default Qwen3-TTS model
    lunavox.bat gui                       # or double-click lunavox-gui.bat

## Commands

    lunavox.bat synth "Hello, world." -o hello.wav
    lunavox.bat serve --host 0.0.0.0 --port 8000
    lunavox.bat doctor                    # sanity check

Conversion from source weights requires the heavy torch stack and is **not**
bundled. Install it separately with `pip install "lunavox[convert]"` on a
full Python install if you need `lunavox model convert`.
"""


LAUNCHER_CLI = """@echo off
setlocal
set "LUNAVOX_PROJECT_ROOT=%~dp0"
"%~dp0python\\python.exe" -m lunavox %*
"""

LAUNCHER_GUI = """@echo off
setlocal
set "LUNAVOX_PROJECT_ROOT=%~dp0"
"%~dp0python\\python.exe" -m lunavox gui %*
"""


def zip_bundle() -> Path:
    ver = version()

    # find the wheel we built earlier
    wheel = next(DIST.glob(f"lunavox-{ver}-*.whl"), None)
    if wheel is None:
        raise SystemExit(
            f"No wheel at dist/lunavox-{ver}-*.whl — run `python -m build --wheel` first."
        )

    stage_name = f"lunavox-v{ver}-cli-gui-bundle"
    stage_dir = STAGE / stage_name
    rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    # 1) python
    tarball = CACHE / PBS_ASSET
    _download(PBS_URL, tarball)
    _extract_pbs(tarball, stage_dir / "python")

    py_exe = stage_dir / "python" / "python.exe"
    if not py_exe.exists():
        raise SystemExit(f"Extracted Python missing: {py_exe}")

    # 2) pip install the wheel into the bundled Python's site-packages
    print("[bundle] pip install lunavox wheel into bundled Python")
    run([str(py_exe), "-m", "pip", "install", "--upgrade", "pip"])
    run([str(py_exe), "-m", "pip", "install", "--no-warn-script-location", str(wheel)])

    # 3) prune
    # python-build-standalone layout: python/Lib/site-packages
    site_pkgs = stage_dir / "python" / "Lib" / "site-packages"
    if site_pkgs.exists():
        _prune_site_packages(site_pkgs)

    # 4) empty skeleton dirs
    for sub in ("build", "lib", "models"):
        d = stage_dir / sub
        d.mkdir(exist_ok=True)
        (d / ".gitkeep").write_text("", encoding="utf-8")

    # 5) marker + launchers + readme
    (stage_dir / ".lunavox-root").write_text(
        f"lunavox v{ver} cli+gui bundle\n", encoding="utf-8"
    )
    write(stage_dir / "lunavox.bat", LAUNCHER_CLI)
    write(stage_dir / "lunavox-gui.bat", LAUNCHER_GUI)
    write(stage_dir / "README.md", _bundle_readme(ver))

    out_zip = DIST / f"{stage_name}.zip"
    make_zip(stage_dir, out_zip)
    rmtree(stage_dir)
    size_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"[bundle] {out_zip}  ({size_mb:.1f} MB)")
    return out_zip


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Package LunaVox release artefacts.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_back = sub.add_parser("zip-backend", help="Build + zip a Windows backend bundle")
    p_back.add_argument("--platform", required=True, choices=list(PLATFORM_META))
    p_back.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild even if lib/ + build/ already match the target platform",
    )

    sub.add_parser("zip-bundle", help="Build + zip the CLI+GUI embedded-Python bundle")

    args = parser.parse_args()

    DIST.mkdir(parents=True, exist_ok=True)
    STAGE.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    if args.cmd == "zip-backend":
        zip_backend(args.platform, force_rebuild=args.force_rebuild)
    elif args.cmd == "zip-bundle":
        zip_bundle()
    else:
        parser.error(f"unknown command {args.cmd!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
