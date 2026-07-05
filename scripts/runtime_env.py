from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys
from importlib import metadata
from pathlib import Path
from typing import Any


DEFAULT_PACKAGES = [
    "faiss",
    "numpy",
    "pandas",
    "pdfplumber",
    "pyarrow",
    "pymupdf",
    "sentence_transformers",
    "tiktoken",
    "torch",
    "transformers",
]

DEFAULT_COMMANDS = ["pdftoppm", "tesseract"]

DIST_NAME_OVERRIDES = {
    "faiss": "faiss-cpu",
    "pymupdf": "PyMuPDF",
    "sentence_transformers": "sentence-transformers",
}


def _distribution_name(module_name: str) -> str:
    return DIST_NAME_OVERRIDES.get(module_name, module_name)


def module_status(module_name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(module_name)
    installed = spec is not None
    version = None
    error = None
    if installed:
        dist_name = _distribution_name(module_name)
        try:
            version = metadata.version(dist_name)
        except metadata.PackageNotFoundError:
            version = None
        except Exception as exc:  # pragma: no cover - defensive metadata fallback
            error = str(exc)
    return {
        "installed": installed,
        "version": version,
        "module": module_name,
        "distribution": _distribution_name(module_name),
        "error": error,
    }


def command_status(command_name: str) -> dict[str, Any]:
    path = shutil.which(command_name)
    return {
        "available": path is not None,
        "path": path,
    }


def dependency_report(
    packages: list[str] | None = None,
    commands: list[str] | None = None,
) -> dict[str, Any]:
    package_names = packages or list(DEFAULT_PACKAGES)
    command_names = commands or list(DEFAULT_COMMANDS)
    modules = {name: module_status(name) for name in package_names}
    binaries = {name: command_status(name) for name in command_names}
    return {
        "modules": modules,
        "commands": binaries,
    }


def collect_runtime_provenance(
    packages: list[str] | None = None,
    commands: list[str] | None = None,
) -> dict[str, Any]:
    report = dependency_report(packages=packages, commands=commands)
    return {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "cwd": str(Path.cwd()),
        "conda_prefix": os.getenv("CONDA_PREFIX"),
        "conda_default_env": os.getenv("CONDA_DEFAULT_ENV"),
        "virtual_env": os.getenv("VIRTUAL_ENV"),
        "dependency_report": report,
    }


def _tiktoken_cl100k_base_available() -> tuple[bool, str]:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return True, f"cl100k_base available ({enc.name})."
    except Exception as exc:
        return False, f"cl100k_base unavailable: {type(exc).__name__}: {exc}"


def critical_environment_checks() -> list[dict[str, Any]]:
    report = dependency_report()
    modules = report["modules"]
    tiktoken_encoding_ok, tiktoken_encoding_detail = _tiktoken_cl100k_base_available()
    checks = [
        {
            "name": "tiktoken_available",
            "ok": bool(modules["tiktoken"]["installed"]),
            "detail": "Required for exact preprocessing token counts.",
        },
        {
            "name": "tiktoken_cl100k_base_available",
            "ok": bool(tiktoken_encoding_ok),
            "detail": tiktoken_encoding_detail,
        },
        {
            "name": "faiss_available",
            "ok": bool(modules["faiss"]["installed"]),
            "detail": "Required for dense index build and retrieval.",
        },
        {
            "name": "sentence_transformers_available",
            "ok": bool(modules["sentence_transformers"]["installed"]),
            "detail": "Required for MiniLM embeddings.",
        },
        {
            "name": "transformers_available",
            "ok": bool(modules["transformers"]["installed"]),
            "detail": "Required by sentence-transformers and index building.",
        },
        {
            "name": "torch_available",
            "ok": bool(modules["torch"]["installed"]),
            "detail": "Required backend for sentence-transformers.",
        },
    ]
    return checks


def pinned_requirements_status(requirements_path: Path) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    if not requirements_path.exists():
        return statuses
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        dist_name, expected_version = [part.strip() for part in line.split("==", 1)]
        try:
            installed_version = metadata.version(dist_name)
            installed = True
        except metadata.PackageNotFoundError:
            installed_version = None
            installed = False
        statuses.append(
            {
                "distribution": dist_name,
                "expected_version": expected_version,
                "installed": installed,
                "installed_version": installed_version,
                "matches": installed and installed_version == expected_version,
            }
        )
    return statuses
