import os
import logging
import subprocess
import time
from pathlib import Path
log = logging.getLogger(__name__)

_IGNORE_FILES = {".DS_Store", ".metadata_never_index"}


def _is_effectively_empty(dirpath):
    try:
        entries = [e for e in os.listdir(dirpath) if e not in _IGNORE_FILES]
        return len(entries) == 0
    except Exception as e:
        log.debug(f"Listdir failed for {dirpath}: {e}")
        return False


def setup_macfuse_mount(path):
    """Setup macFUSE/FUSE-T mount point safely."""
    path = os.path.expanduser(path)
    real = os.path.realpath(path)

    # Reject symlink mountpoints; less surprises
    if os.path.islink(path):
        log.error(f"Mount point {path} is a symlink; refusing.")
        return False

    if os.path.lexists(real):
        if os.path.ismount(real):
            log.warning(f"Mount point {path} is already mounted")
            return False
        if not os.path.isdir(real):
            log.error(f"Mount point {path} exists but is not a directory")
            return False
        if not _is_effectively_empty(real):
            log.warning(f"Mount point {path} exists and is not empty")
            return False
        # OK: empty directory; proceed
    else:
        try:
            os.makedirs(real, exist_ok=True)
            log.info(f"Created mount point directory: {real}")
        except OSError as e:
            log.error(f"Could not create mount point {real}: {e}")
            return False

    # Permission sanity: need write/execute on the directory to mount
    if not os.access(real, os.W_OK | os.X_OK):
        log.error(f"Mount point {real} is not writable/searchable by current user")
        return False

    # Finder/Volumes hint (debug level – not an error)
    if not real.startswith("/Volumes/"):
        log.debug("Not mounted in volumes, will not be visible in Finder (use volname option).")

    return True


def is_macfuse_mount(path):
    """Return True if mountpoint is backed by macFUSE/osxfuse/fuse.

    Uses the system 'mount' output to positively identify filesystem type,
    which is more reliable than os.path.ismount for readiness checks.
    """
    try:
        abs_path = os.path.abspath(path)
        out = subprocess.check_output(["mount"], text=True, errors="ignore")
    except Exception as e:
        log.debug(f"macFUSE mount check failed for {path}: {e}")
        return False

    needle = f" on {abs_path} "
    for line in out.splitlines():
        if needle in line:
            if "(" in line and ")" in line:
                inside = line[line.find("(") + 1: line.rfind(")")]
                opts = [p.strip().lower() for p in inside.split(",")]
                if any(s.startswith("fuse") or "osxfuse" in s or "macfuse" in s for s in opts):
                    return True
    return False


def wait_for_mount_ready(path, timeout=30.0, poll_interval=0.3):
    """Poll until a macFUSE mount is ready at path or timeout."""
    deadline = time.time() + float(timeout)
    while time.time() < deadline:
        if is_macfuse_mount(path):
            return True
        time.sleep(float(poll_interval))
    return False


def remove_stale_poison(root, mountpoint):
    """Remove leftover .poison markers that could immediately kill a new mount."""
    for p in (
        os.path.join(os.path.expanduser(root), ".poison"),
        os.path.join(os.path.expanduser(mountpoint), ".poison"),
    ):
        try:
            os.remove(p)
            log.debug(f"Removed stale poison file: {p}")
        except FileNotFoundError:
            pass
        except Exception as e:
            log.debug(f"Failed removing stale poison {p}: {e}")


def spawn_mac_fuse_worker(root, mountpoint, *, nothreads=False, log_dir=None, python_exe=None):
    """Spawn the isolated FUSE worker process for macOS.

    Returns (Popen, log_fh) so the caller can wait() and close the log handle.
    """
    try:
        import sys as _sys
        exe = python_exe or _sys.executable

        # Prepare logging
        if log_dir is None:
            log_dir = os.path.join(Path.home(), ".autoortho-data", "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"autoortho-{os.path.basename(os.path.abspath(mountpoint))}.log")
        log_fh = open(log_path, "a", buffering=1)

        args = [
            exe,
            "-m", "autoortho.mount_worker",
            os.path.expanduser(root),
            os.path.expanduser(mountpoint),
        ]
        if nothreads:
            args.append("--nothreads")

        proc = subprocess.Popen(
            args,
            stdout=log_fh,
            stderr=log_fh,
            close_fds=True,
        )
        log.info(f"Spawned macFUSE worker pid={proc.pid} for {mountpoint}")
        return proc, log_fh
    except Exception:
        # Ensure file handle is not leaked on failure
        try:
            log_fh.close()  # type: ignore[name-defined]
        except Exception:
            pass
        raise
