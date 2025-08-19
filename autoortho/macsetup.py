import os
import sys
import platform
import subprocess
from aoconfig import CFG

import logging
log = logging.getLogger(__name__)

from ctypes.util import find_library

def find_mac_libs():
    """Find macFUSE libraries on macOS."""
    fuse_libs = []

    # Check for macFUSE installation
    log.info("Looking for macFUSE or fuse-t ...")
    
    # Common macFUSE library paths
    possible_paths = [
        '/usr/local/lib/libfuse.dylib',
        '/usr/local/lib/libfuse-t.dylib',
    ]
    
    # Try to find libfuse using ctypes utility
    _lib_fuse = find_library('libfuse')
    if _lib_fuse:
        log.info(f"Found macfuse via find_library: {_lib_fuse}")
        fuse_libs.append(("mac-FUSE", _lib_fuse))

    _lib_fuse_t = find_library('libfuse-t')
    if _lib_fuse_t:
        log.info(f"Found fuse-t via find_library: {_lib_fuse_t}")
        fuse_libs.append(("mac-FUSE", _lib_fuse_t))
    else:
        # Check common installation paths
        for path in possible_paths:
            if os.path.exists(path):
                log.info(f"Found a fuse library at {path}")
                fuse_libs.append(("mac-FUSE", path))
                break


    if not fuse_libs:
        log.error("No macFUSE or fuse-t installation detected!")
        return None, None

    fusemode, fuselib = fuse_libs[0]
    log.info(f"Will use detected {fusemode} with libs {fuselib}")
    
    # Set the FUSE library path environment variable
    os.environ['FUSE_LIBRARY_PATH'] = fuselib
    
    return fusemode, fuselib


def check_mount_permissions():
    """Check if user has permission to mount filesystems."""
    # On macOS, mounting typically requires admin privileges
    # or the user to be in the operator group
    try:
        # Check if user is in operator group (which can mount)
        import grp
        try:
            operator_group = grp.getgrnam('operator')
            if os.getuid() in operator_group.gr_mem:
                log.info("User is in operator group - mount permissions OK")
                return True
        except KeyError:
            pass
        
        # Check if user is admin/root
        if os.getuid() == 0:
            log.info("Running as root - mount permissions OK")
            return True
        
        # Check if user is in admin group
        try:
            admin_group = grp.getgrnam('admin')
            current_user = os.getlogin()
            if current_user in admin_group.gr_mem:
                log.info("User is in admin group - mount permissions likely OK")
                return True
        except:
            pass
            
        log.warning("User may not have mount permissions. You might need to run with sudo or be in the operator group.")
        return False
        
    except Exception as e:
        log.warning(f"Could not check mount permissions: {e}")
        return True  # Assume it's OK and let FUSE handle the error


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


def cleanup_macfuse_mount(path):
    """Cleanup macFUSE mount point."""
    if os.path.ismount(path):
        try:
            # Try to unmount
            subprocess.run(['umount', path], check=True, capture_output=True)
            log.info(f"Successfully unmounted {path}")
        except subprocess.CalledProcessError as e:
            log.error(f"Failed to unmount {path}: {e}")
            # Try force unmount
            try:
                subprocess.run(['umount', '-f', path], check=True, capture_output=True)
                log.info(f"Force unmounted {path}")
            except subprocess.CalledProcessError as e2:
                log.error(f"Force unmount also failed: {e2}")
                return False
    
    return True
