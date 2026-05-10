"""Helpers for building PufferLib's one-env-at-a-time native backend."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from puffer_llm_sweeper.runner import load_run_config


DEFAULT_PUFFER_SOURCE_DIR = Path(".deps/PufferLib")


@dataclass(frozen=True)
class BackendInfo:
    env_name: str
    gpu: int | None
    precision_bytes: int | None


def backend_info() -> BackendInfo | None:
    code = (
        "import pufferlib._C as c; "
        "print(getattr(c, 'env_name', ''), getattr(c, 'gpu', ''), "
        "getattr(c, 'precision_bytes', ''))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    parts = result.stdout.strip().split()
    if not parts:
        return None
    return BackendInfo(
        env_name=parts[0],
        gpu=_parse_optional_int(parts, 1),
        precision_bytes=_parse_optional_int(parts, 2),
    )


def ensure_backend_for_config(
    config_path: Path,
    source_dir: Path = DEFAULT_PUFFER_SOURCE_DIR,
    force: bool = False,
    cpu: bool = False,
    arch: str | None = None,
) -> BackendInfo:
    config = load_run_config(config_path)
    return ensure_backend(
        config.env_name,
        source_dir=source_dir,
        force=force,
        cpu=cpu,
        arch=arch,
    )


def ensure_backend(
    env_name: str,
    source_dir: Path = DEFAULT_PUFFER_SOURCE_DIR,
    force: bool = False,
    cpu: bool = False,
    arch: str | None = None,
) -> BackendInfo:
    current = backend_info()
    if not force and current and current.env_name == env_name:
        return current

    build_backend(env_name, source_dir=source_dir, cpu=cpu, arch=arch)
    updated = backend_info()
    if updated is None:
        raise RuntimeError("Built PufferLib backend, but pufferlib._C still cannot be imported.")
    if updated.env_name != env_name:
        raise RuntimeError(
            f"Built PufferLib backend for {env_name}, but imported backend is {updated.env_name}."
        )
    return updated


def build_backend(
    env_name: str,
    source_dir: Path = DEFAULT_PUFFER_SOURCE_DIR,
    cpu: bool = False,
    arch: str | None = None,
) -> None:
    source_dir = source_dir.resolve()
    build_script = source_dir / "build.sh"
    if not build_script.exists():
        raise FileNotFoundError(
            f"Could not find PufferLib build script at {build_script}. "
            "Clone PufferLib into .deps/PufferLib or pass --source-dir."
        )

    command = ["bash", "build.sh", env_name, "--float"]
    if cpu:
        command.append("--cpu")

    env = build_environment(source_dir=source_dir, arch=arch, cpu=cpu)
    subprocess.run(command, cwd=source_dir, env=env, check=True)


def build_environment(source_dir: Path, arch: str | None = None, cpu: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("CC", _default_cc())
    env.setdefault("CXX", "g++")

    cuda_home = env.get("CUDA_HOME") or env.get("CUDA_PATH")
    if not cuda_home and Path("/opt/cuda").exists():
        cuda_home = "/opt/cuda"
    if cuda_home:
        env["CUDA_HOME"] = cuda_home
        env["PATH"] = f"{Path(cuda_home) / 'bin'}:{env.get('PATH', '')}"

    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
    env.setdefault("CCACHE_BASEDIR", str(source_dir))
    if not cpu:
        detected_arch = arch or detect_cuda_arch()
        if detected_arch:
            env.setdefault("NVCC_ARCH", detected_arch)
    return env


def detect_cuda_arch() -> str | None:
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"


def _default_cc() -> str:
    if shutil.which("ccache") and shutil.which("gcc"):
        return "ccache gcc"
    if shutil.which("gcc"):
        return "gcc"
    if shutil.which("clang"):
        return "clang"
    return "cc"


def _parse_optional_int(parts: list[str], index: int) -> int | None:
    try:
        return int(parts[index])
    except (IndexError, ValueError):
        return None
