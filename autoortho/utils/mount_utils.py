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
        # Ensure mountpoint directory exists
        os.makedirs(mountpoint, exist_ok=True)
        # Present placeholder content for X-Plane when not mounted
        for d in ('Earth nav data', 'terrain', 'textures'):
            os.makedirs(os.path.join(mountpoint, d), exist_ok=True)
        Path(placeholder_path).touch()
    except Exception as e:
        log.debug(f"cleanup_mountpoint error for {mountpoint}: {e}")