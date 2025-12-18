import os
import sys
import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)


def safe_ismount(path) -> bool:
    """
    Safe wrapper for os.path.ismount() that handles exceptions gracefully.
    
    On Windows, os.path.ismount() can raise OSError [WinError 123] for paths that
    don't exist yet or contain certain characters (e.g., paths with spaces).
    This function also handles TypeError for None or invalid path types.
    
    Args:
        path: The path to check for mount status. Can be str, bytes, or path-like.
              None or invalid types are handled gracefully.
        
    Returns:
        True if the path is a mount point, False otherwise (including error cases).
    """
    # Handle None or empty paths before calling os.path.ismount
    if path is None:
        return False
    
    try:
        return os.path.ismount(path)
    except (OSError, TypeError, ValueError) as e:
        # OSError: WinError 123 "The filename, directory name, or volume label syntax is incorrect"
        #          Can happen on Windows for paths that don't exist or have unusual formats
        # TypeError: Raised if path is not a valid path type (e.g., int, object)
        # ValueError: Raised for embedded null characters or other invalid path values
        log.debug(f"safe_ismount: os.path.ismount({path!r}) raised {type(e).__name__}: {e}")
        return False


_IGNORE_FILES = {".DS_Store", ".metadata_never_index"}
_AO_PLACEHOLDER_ITEMS = {"Earth nav data", "terrain", "textures", ".AO_PLACEHOLDER"}

def cleanup_mountpoint(mountpoint):
    placeholder_path = os.path.join(mountpoint, ".AO_PLACEHOLDER")
    if os.path.lexists(mountpoint):
        log.info(f"Cleaning up mountpoint: {mountpoint}")
        os.rmdir(mountpoint)
    if safe_ismount(mountpoint):
        log.debug(f"Skipping cleanup: still mounted: {mountpoint}")
    else:
        for d in ('Earth nav data', 'terrain', 'textures'):
            os.makedirs(os.path.join(mountpoint, d), exist_ok=True)
        Path(placeholder_path).touch()


def _is_frozen() -> bool:
    """
    Check if running as a frozen/compiled application (PyInstaller).
    
    PyInstaller sets sys.frozen = True when running as a bundled executable.
    This is used to determine how to launch subprocess workers.
    """
    return getattr(sys, 'frozen', False)


def is_only_ao_placeholder(mountpoint: str) -> bool:
    """True if the directory contains only our known placeholder structure."""
    try:
        entries = [e for e in os.listdir(mountpoint) if e not in _IGNORE_FILES]
    except FileNotFoundError:
        return True  # treat missing dir as 'empty'
    except Exception as e:
        log.debug(f"is_only_ao_placeholder listdir failed: {e}")
        return False
    return set(entries).issubset(_AO_PLACEHOLDER_ITEMS)


def clear_ao_placeholder(mountpoint: str) -> None:
    """Remove our placeholder structure (and only that)."""
    try:
        for name in _AO_PLACEHOLDER_ITEMS:
            p = os.path.join(mountpoint, name)
            if os.path.isdir(p) and not os.path.islink(p):
                shutil.rmtree(p, ignore_errors=True)
            elif os.path.lexists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
        log.info(f"Cleared AO placeholder from: {mountpoint}")
    except Exception as e:
        log.warning(f"clear_ao_placeholder failed for {mountpoint}: {e}")