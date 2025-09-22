import os
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def cleanup_mountpoint(mountpoint):
    placeholder_path = os.path.join(mountpoint, ".AO_PLACEHOLDER")
    try:
        if os.path.ismount(mountpoint):
            log.debug(f"Skipping cleanup: still mounted: {mountpoint}")
            return
        # Ensure the directory exists and contains placeholder structure
        os.makedirs(mountpoint, exist_ok=True)
        for d in ('Earth nav data', 'terrain', 'textures'):
            os.makedirs(os.path.join(mountpoint, d), exist_ok=True)
        Path(placeholder_path).touch()
        log.info(f"Prepared placeholder content at: {mountpoint}")
    except Exception as e:
        log.warning(f"cleanup_mountpoint failed for {mountpoint}: {e}")


def _is_nuitka_compiled() -> bool:
    # Nuitka exposes __compiled__ on the module that was compiled.
    import sys
    m = sys.modules.get("__main__")
    return bool(getattr(m, "__compiled__", False))