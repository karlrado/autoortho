"""
aopipeline - Native Performance Pipeline for AutoOrtho

This module provides native (C) implementations of performance-critical
operations that bypass Python's GIL limitation:

- AoCache: Parallel batch cache file I/O
- AoDecode: Parallel JPEG decoding  
- AoDDS: Native DDS texture building with ISPC compression

Usage:
    from autoortho.aopipeline import AoCache, AoDDS
    
    if AoCache.is_available():
        results = AoCache.batch_read_cache(paths)
    
    if AoDDS.is_available():
        dds_bytes = AoDDS.build_tile_native(cache_dir, row, col, ...)
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
    return components


__all__ = ['AoCache', 'AoDecode', 'AoDDS', 'is_available', 'get_available_components']
