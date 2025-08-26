import os
import logging
log = logging.getLogger(__name__)


def setup_macfuse_mount(path):
    """Setup macFUSE mount point."""
    if os.path.lexists(path):
        if os.path.ismount(path):
            log.warning(f"Mount point {path} is already mounted")
            return False
        elif os.path.isdir(path):
            # Check if directory is empty
            if os.listdir(path):
                log.warning(f"Mount point {path} exists and is not empty")
                return False
        else:
            log.error(f"Mount point {path} exists but is not a directory")
            return False
    else:
        # Create mount point directory
        try:
            os.makedirs(path, exist_ok=True)
            log.info(f"Created mount point directory: {path}")
        except OSError as e:
            log.error(f"Could not create mount point {path}: {e}")
            return False

    return True
