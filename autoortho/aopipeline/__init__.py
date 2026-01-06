"""
aopipeline - Native Performance Pipeline for AutoOrtho

This module provides native (C) implementations of performance-critical
operations that bypass Python's GIL limitation:

- AoCache: Parallel batch cache file I/O
- AoDecode: Parallel JPEG decoding
- AoDDS: Native DDS texture building with ISPC compression
- AoHttp: Native HTTP client pool with libcurl

Usage:
    from autoortho.aopipeline import AoCache, AoDDS
    
    if AoCache.is_available():
        results = AoCache.batch_read_cache(paths)
    
    if AoDDS.is_available():
        dds_bytes = AoDDS.build_tile_native(cache_dir, row, col, ...)
"""

import logging

log = logging.getLogger(__name__)

# Direct imports - each module handles its own availability checking
from . import AoCache
from . import AoDecode
from . import AoDDS
from . import AoHttp

# Convenience functions
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
    if AoHttp.is_available():
        components.append('http')
    return components


__all__ = ['AoCache', 'AoDecode', 'AoDDS', 'AoHttp', 'is_available', 'get_available_components']
