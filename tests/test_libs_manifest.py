"""Schema integrity of ``src/lunavox/build/libs.json`` — the manifest
``lunavox download-libs`` reads to know which runtime libraries to
fetch. A broken manifest breaks first-time setup on every platform, so
we lock in the shape here."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

import pytest

MANIFEST = Path(__file__).resolve().parent.parent / "src" / "lunavox" / "build" / "libs.json"


@pytest.fixture(scope="module")
def manifest() -> dict:
    with open(MANIFEST, encoding="utf-8") as f:
        return json.load(f)


def test_top_level_shape(manifest):
    assert set(manifest.keys()) >= {"libraries", "platforms"}
    assert isinstance(manifest["libraries"], dict)
    assert isinstance(manifest["platforms"], dict)


def test_libraries_have_onnx_and_llama(manifest):
    libs = manifest["libraries"]
    assert "onnx" in libs
    assert "llama" in libs
    for name, lib in libs.items():
        assert "version" in lib, f"library {name} missing version"
        assert "backends" in lib, f"library {name} missing backends"
        assert isinstance(lib["backends"], dict)
        assert lib["backends"], f"library {name} has no backends"


def test_all_backend_urls_parse(manifest):
    for lib_name, lib in manifest["libraries"].items():
        for backend, url in lib["backends"].items():
            parsed = urlparse(url)
            assert parsed.scheme in {"http", "https"}, (
                f"{lib_name}/{backend} has non-http URL: {url}"
            )
            assert parsed.netloc, f"{lib_name}/{backend} has empty host: {url}"


def test_platforms_reference_valid_backends(manifest):
    libs = manifest["libraries"]
    for plat_name, plat in manifest["platforms"].items():
        assert "components" in plat, f"platform {plat_name} missing components"
        for lib_name, backend_key in plat["components"].items():
            assert lib_name in libs, f"platform {plat_name} references unknown library {lib_name}"
            assert backend_key in libs[lib_name]["backends"], (
                f"platform {plat_name}/{lib_name} references unknown backend {backend_key}"
            )


def test_every_platform_has_onnx_and_llama(manifest):
    for plat_name, plat in manifest["platforms"].items():
        comps = plat["components"]
        assert "onnx" in comps, f"platform {plat_name} missing onnx"
        assert "llama" in comps, f"platform {plat_name} missing llama"
