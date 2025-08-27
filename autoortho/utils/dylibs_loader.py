from __future__ import annotations
import os, sys, platform
from pathlib import Path

ispc_names = {
    "linux": "libispc_texcomp.so",
    "windows": "ispc_texcomp.dll",
    "darwin": "libispc_texcomp.dylib",
}

stb_names = {
    "linux": "lib_stb_dxt.so",
    "windows": "stb_dxt.dll",
    "darwin": None,
}

current_system = platform.system().lower()


def _app_frameworks_dir() -> Path | None:
    """If running inside AutoOrtho.app, return .../Contents/Frameworks, else None."""
    if current_system != "darwin":
        return None
    exe = Path(sys.argv[0]).resolve()
    # .../AutoOrtho.app/Contents/MacOS/AutoOrtho
    # parents[0]=.../MacOS, [1]=.../Contents, [2]=.../AutoOrtho.app
    try:
        if exe.parents[2].suffix == ".app":
            return exe.parents[1] / "Frameworks"
    except IndexError:
        pass
    return None


def compressor_lib_path(compressor: str) -> str:
    if compressor not in ["ISPC", "STB"]:
        raise ValueError(f"Invalid compressor: {compressor}")
    if current_system not in ["linux", "windows", "darwin"]:
        raise ValueError(f"Invalid system: {current_system}")

    here = Path(__file__).resolve().parent.parent  # .../autoortho/

    if current_system == "linux":
        return str(here / "lib" / "linux" / stb_names[current_system] if compressor == "STB" else ispc_names[current_system])
    if current_system == "windows":
        return str(here / "lib" / "windows" / stb_names[current_system] if compressor == "STB" else ispc_names[current_system])
    if current_system == "darwin":
        if compressor == "STB":
            return None  # STB is not supported on macOS
        fw = _app_frameworks_dir()
        if fw is not None:
            cand = fw / ispc_names[current_system]
            if cand.exists():
                return str(cand)
        # dev run fallback (repo layout)
        return str(here / "lib" / "macos" / ispc_names[current_system])
    raise RuntimeError("Unsupported OS")


def aoimage_lib_path() -> str:
    if current_system not in ["linux", "windows", "darwin"]:
        raise ValueError(f"Invalid system: {current_system}")

    here = Path(__file__).resolve().parent.parent  # .../autoortho/

    if current_system == "linux":
        return str(here / "aoimage" / "aoimage.so")
    if current_system == "windows":
        return str(here / "aoimage" / "aoimage.dll")
    if current_system == "darwin":
        return str(here / "aoimage" / "aoimage.dylib")
    raise RuntimeError("Unsupported OS")

