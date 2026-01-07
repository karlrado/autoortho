"""
aopipeline - Native Performance Pipeline for AutoOrtho

This module provides native (C) implementations of performance-critical
operations that bypass Python's GIL limitation:

- AoCache: Parallel batch cache file I/O
- AoDecode: Parallel JPEG decoding  
- AoDDS: Native DDS texture building with ISPC compression
- AoBundle: Cache bundle format for consolidated file I/O

Optimal Usage (Hybrid Pipeline):
    from autoortho.aopipeline import AoDDS, AoBundle
    
    # Option 1: Python reads files, native decodes
    jpeg_datas = [Path(p).read_bytes() for p in chunk_paths]
    dds_bytes = AoDDS.build_from_jpegs(jpeg_datas)
    
    # Option 2: Bundle format (single file = fastest)
    if AoBundle.bundle_exists(cache_dir, col, row, zoom, maptype):
        dds_bytes = AoBundle.build_dds_from_bundle(bundle_path)
"""

import logging
import os
import sys

log = logging.getLogger(__name__)

# On Windows, add the DLL directory to PATH BEFORE importing modules
# This ensures dependencies (libturbojpeg, libgomp, etc.) can be found
if sys.platform == 'win32':
    _lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib', 'windows')
    if os.path.isdir(_lib_dir):
        # Add to PATH so Windows can find DLL dependencies
        os.environ['PATH'] = _lib_dir + os.pathsep + os.environ.get('PATH', '')
        # Also use add_dll_directory if available (Python 3.8+)
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(_lib_dir)

# Direct imports - each module handles its own availability checking
from . import AoCache
from . import AoDecode
from . import AoDDS
from . import AoBundle


def is_available() -> bool:
    """Check if any native pipeline component is available."""
    return any([
        AoCache.is_available(),
        AoDecode.is_available(),
        AoDDS.is_available(),
    ])


def get_available_components() -> list:
    """Return list of available native components."""
    components = []
    if AoCache.is_available():
        components.append('cache')
    if AoDecode.is_available():
        components.append('decode')
    if AoDDS.is_available():
        components.append('dds')
    if AoBundle.is_available():
        components.append('bundle')
    return components


__all__ = [
    'AoCache', 'AoDecode', 'AoDDS', 'AoBundle',
    'is_available', 'get_available_components'
]
