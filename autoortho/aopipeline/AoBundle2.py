"""
AoBundle2 - Python wrapper for native AOB2 multi-zoom mutable cache bundle operations.

AOB2 bundles extend AOB1 with:
- Multiple zoom levels in a single file
- Mutable data sections (append-only with garbage tracking)
- In-place chunk patching and zoom expansion
- Efficient compaction when fragmentation exceeds threshold

Example usage:
    from autoortho.aopipeline import AoBundle2
    
    # Create a bundle from existing cache files
    AoBundle2.create_bundle(
        cache_dir="/path/to/cache",
        tile_row=456, tile_col=123,
        maptype="BI", zoom=16,
        output_path="/path/to/bundles/456_123_BI.aob2"
    )
    
    # Build DDS directly from bundle (fastest path)
    dds_bytes = AoBundle2.build_dds(
        "/path/to/bundles/456_123_BI.aob2",
        target_zoom=16
    )
    
    # Pure Python fallback (always works)
    from AoBundle2 import Bundle2Python
    bundle = Bundle2Python("/path/to/bundle.aob2")
    jpeg_datas = bundle.get_all_chunks(16)
"""

from ctypes import (
    CDLL, POINTER, Structure, byref, cast, c_float, c_int64,
    c_int32, c_uint8, c_uint16, c_uint32, c_uint64, c_char_p, c_void_p, c_size_t
)
from collections import OrderedDict
from enum import IntFlag
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple
import logging
import mmap
import os
import struct
import sys
import threading
import time

log = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

BUNDLE2_EXTENSION = ".aob2"
BUNDLE2_MAGIC = 0x32424F41  # "AOB2" in little-endian
BUNDLE2_VERSION = 2
BUNDLE2_HEADER_SIZE = 64
BUNDLE2_MAX_ZOOM_LEVELS = 8
BUNDLE2_MAX_MAPTYPE = 64
BUNDLE2_COMPACTION_THRESHOLD = 0.30

FORMAT_BC1 = 0  # DXT1, no alpha
FORMAT_BC3 = 1  # DXT5, with alpha

# I/O optimization: buffer size for efficient large writes
BUNDLE2_WRITE_BUFFER_SIZE = 256 * 1024  # 256KB


def _preallocate_file(f, size: int) -> bool:
    """
    Pre-allocate file size for more efficient writes.
    
    This hints to the filesystem to allocate contiguous blocks,
    reducing fragmentation and improving write performance.
    
    Args:
        f: Open file object in write mode
        size: Target file size in bytes
        
    Returns:
        True if successful, False otherwise (non-fatal).
    """
    if size <= 0:
        return False
    
    try:
        fd = f.fileno()
    except (OSError, AttributeError):
        return False
    
    try:
        # Linux: posix_fallocate is most efficient - actually allocates blocks
        if sys.platform.startswith('linux'):
            try:
                os.posix_fallocate(fd, 0, size)
                return True
            except (OSError, AttributeError):
                pass
        
        # Windows: seek to end and write a byte, then seek back
        if sys.platform == 'win32':
            try:
                f.seek(size - 1)
                f.write(b'\x00')
                f.seek(0)
                return True
            except OSError:
                pass
        else:
            # POSIX fallback (macOS, others): ftruncate
            try:
                os.ftruncate(fd, size)
                f.seek(0)
                return True
            except (OSError, AttributeError):
                pass
    except Exception:
        pass
    
    return False  # Non-fatal


def _atomic_replace_with_retry(temp_path: str, target_path: str, max_retries: int = 5):
    """
    Atomically replace target_path with temp_path, with retry on Windows file locking.
    
    On Windows, os.replace() fails with WinError 5 (Access Denied) if another
    process/thread has the target file open. This function retries with exponential
    backoff to handle transient file locks from concurrent readers.
    
    Args:
        temp_path: Path to the temporary file to rename
        target_path: Path to the destination (will be overwritten)
        max_retries: Maximum number of retry attempts (default 5 = ~1.5s total)
    """
    import platform
    
    last_error = None
    for attempt in range(max_retries):
        try:
            os.replace(temp_path, target_path)
            return  # Success
        except OSError as e:
            last_error = e
            # On Windows, WinError 5 = Access Denied (file locked)
            # On other platforms, this shouldn't happen but handle gracefully
            if platform.system() == 'Windows' and hasattr(e, 'winerror') and e.winerror == 5:
                if attempt < max_retries - 1:
                    # Exponential backoff: 50ms, 100ms, 200ms, 400ms
                    delay = 0.05 * (2 ** attempt)
                    time.sleep(delay)
                    continue
            # For other errors or final attempt, break and raise
            break
    
    # Clean up temp file and raise the last error
    try:
        os.unlink(temp_path)
    except OSError:
        pass
    
    if last_error:
        raise last_error


class BundleFlags(IntFlag):
    """Bundle flags (stored in header)."""
    NONE = 0x0000
    MUTABLE = 0x0001
    MULTI_ZOOM = 0x0002
    COMPACTION_NEEDED = 0x0004
    LOCKED = 0x0008


class ChunkFlags(IntFlag):
    """Chunk flags (stored in chunk index entry)."""
    MISSING = 0x0000
    VALID = 0x0001
    PLACEHOLDER = 0x0002
    UPSCALED = 0x0004
    GARBAGE = 0x0080


# ============================================================================
# Bundle Metadata Cache (LRU)
# ============================================================================
# Caches parsed bundle metadata (header + zoom table + chunk indices) across
# Bundle2Python instances. This avoids redundant parsing when the same bundle
# is accessed multiple times.

class BundleMetadata(NamedTuple):
    """Cached bundle metadata (header + zoom table + chunk indices)."""
    header: dict
    maptype: str
    zoom_table: list
    chunk_indices: dict
    file_size: int


class BundleMetadataCache:
    """
    Thread-safe LRU cache for bundle metadata.
    
    Caches parsed headers, zoom tables, and chunk indices keyed by (path, mtime).
    Invalidates automatically when file modification time changes.
    
    Performance: Avoids re-parsing ~500 bytes of metadata per bundle open.
    For hot paths accessing the same bundles repeatedly, this eliminates
    redundant I/O and struct unpacking.
    """
    
    def __init__(self, maxsize: int = 128):
        """
        Create metadata cache.
        
        Args:
            maxsize: Maximum number of bundle metadata entries to cache
        """
        self._maxsize = maxsize
        self._cache: OrderedDict[Tuple[str, float], BundleMetadata] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
    
    def get(self, path: str) -> Optional[BundleMetadata]:
        """
        Get cached metadata for a bundle.
        
        Args:
            path: Path to bundle file
            
        Returns:
            Cached BundleMetadata if valid cache entry exists, None otherwise
        """
        try:
            mtime = os.path.getmtime(path)
            file_size = os.path.getsize(path)
        except OSError:
            return None
        
        cache_key = (path, mtime)
        
        with self._lock:
            if cache_key in self._cache:
                # Verify file size matches (cheap corruption check)
                metadata = self._cache[cache_key]
                if metadata.file_size == file_size:
                    # Move to end (most recently used)
                    self._cache.move_to_end(cache_key)
                    self._hits += 1
                    return metadata
                else:
                    # File size changed - invalidate
                    del self._cache[cache_key]
            
            self._misses += 1
            return None
    
    def put(self, path: str, metadata: BundleMetadata) -> None:
        """
        Cache metadata for a bundle.
        
        Args:
            path: Path to bundle file
            metadata: Parsed metadata to cache
        """
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return
        
        cache_key = (path, mtime)
        
        with self._lock:
            # Remove if already exists (will re-add at end)
            if cache_key in self._cache:
                del self._cache[cache_key]
            
            self._cache[cache_key] = metadata
            
            # Evict oldest if over capacity
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)
    
    def invalidate(self, path: str) -> None:
        """
        Invalidate all cached entries for a path.
        
        Args:
            path: Path to bundle file
        """
        with self._lock:
            # Remove all entries for this path (regardless of mtime)
            keys_to_remove = [k for k in self._cache if k[0] == path]
            for k in keys_to_remove:
                del self._cache[k]
    
    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                'size': len(self._cache),
                'maxsize': self._maxsize,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
            }


# Global metadata cache (shared across all Bundle2Python instances)
_metadata_cache = BundleMetadataCache(maxsize=128)


def get_metadata_cache() -> BundleMetadataCache:
    """Get the global bundle metadata cache."""
    return _metadata_cache


# ============================================================================
# Library Loading
# ============================================================================

_lib = None
_lib_path = None


def _get_lib_dir() -> Path:
    """Get the directory containing the native library."""
    this_dir = Path(__file__).parent
    
    if sys.platform == 'win32':
        return this_dir / 'lib' / 'windows'
    elif sys.platform == 'darwin':
        return this_dir / 'lib' / 'macos'
    else:
        return this_dir / 'lib' / 'linux'


def _get_lib_name() -> str:
    """Get the library filename for current platform."""
    if sys.platform == 'win32':
        return 'aopipeline.dll'
    elif sys.platform == 'darwin':
        return 'libaopipeline.dylib'
    else:
        return 'libaopipeline.so'


def _load_library() -> CDLL:
    """Load the native library."""
    global _lib, _lib_path
    
    if _lib is not None:
        return _lib
    
    lib_dir = _get_lib_dir()
    lib_name = _get_lib_name()
    lib_path = lib_dir / lib_name
    
    if not lib_path.exists():
        raise FileNotFoundError(
            f"Native library not found: {lib_path}. "
            f"Please build the aopipeline library."
        )
    
    try:
        # Windows: Add DLL directory to search path (Python 3.8+)
        if sys.platform == 'win32':
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(str(lib_dir))
        
        _lib = CDLL(str(lib_path))
        _lib_path = lib_path
        
        _setup_signatures(_lib)
        
        return _lib
        
    except OSError as e:
        error_str = str(e)
        
        # Provide platform-specific help
        if sys.platform == 'linux' and 'libgomp' in error_str:
            raise ImportError(
                f"OpenMP runtime library not found.\n"
                f"Install with:\n"
                f"  Ubuntu/Debian: sudo apt install libgomp1\n"
                f"  Fedora/RHEL:   sudo dnf install libgomp\n"
                f"  Arch Linux:    sudo pacman -S gcc-libs\n"
                f"Original error: {e}"
            ) from e
        elif sys.platform == 'win32':
            raise ImportError(
                f"Failed to load native library. Ensure all DLLs are present:\n"
                f"  - aopipeline.dll\n"
                f"  - libgomp-1.dll\n"
                f"  - libturbojpeg.dll\n"
                f"  - libgcc_s_seh-1.dll\n"
                f"  - libwinpthread-1.dll\n"
                f"Original error: {e}"
            ) from e
        else:
            raise ImportError(f"Failed to load native library: {e}") from e


def _setup_signatures(lib: CDLL):
    """Set up function signatures for the library."""
    # aobundle2_create
    lib.aobundle2_create.argtypes = [
        c_char_p,   # cache_dir
        c_int32,    # tile_row
        c_int32,    # tile_col
        c_char_p,   # maptype
        c_int32,    # zoom
        c_int32,    # chunks_per_side
        c_char_p,   # output_path
        c_void_p    # result (optional)
    ]
    lib.aobundle2_create.restype = c_int32
    
    # aobundle2_create_from_data
    lib.aobundle2_create_from_data.argtypes = [
        c_int32,            # tile_row
        c_int32,            # tile_col
        c_char_p,           # maptype
        c_int32,            # zoom
        POINTER(c_void_p),  # jpeg_data
        POINTER(c_uint32),  # jpeg_sizes
        c_int32,            # chunk_count
        c_char_p,           # output_path
        c_void_p            # result (optional)
    ]
    lib.aobundle2_create_from_data.restype = c_int32
    
    # aobundle2_create_empty
    lib.aobundle2_create_empty.argtypes = [
        c_int32,    # tile_row
        c_int32,    # tile_col
        c_char_p,   # maptype
        c_int32,    # initial_zoom
        c_int32,    # chunks_per_side
        c_char_p    # output_path
    ]
    lib.aobundle2_create_empty.restype = c_int32
    
    # aobundle2_build_dds
    lib.aobundle2_build_dds.argtypes = [
        c_char_p,           # bundle_path
        c_int32,            # target_zoom
        c_int32,            # format
        POINTER(c_uint8),   # missing_color
        POINTER(c_uint8),   # dds_output
        c_uint32,           # output_size
        POINTER(c_uint32)   # bytes_written
    ]
    lib.aobundle2_build_dds.restype = c_int32
    
    # aobundle2_get_fragmentation
    lib.aobundle2_get_fragmentation.argtypes = [c_char_p]
    lib.aobundle2_get_fragmentation.restype = c_float
    
    # aobundle2_needs_compaction
    lib.aobundle2_needs_compaction.argtypes = [c_char_p, c_float]
    lib.aobundle2_needs_compaction.restype = c_int32
    
    # aobundle2_compact
    lib.aobundle2_compact.argtypes = [c_char_p]
    lib.aobundle2_compact.restype = c_int64
    
    # aobundle2_validate
    lib.aobundle2_validate.argtypes = [c_char_p]
    lib.aobundle2_validate.restype = c_int32
    
    # aobundle2_version
    lib.aobundle2_version.argtypes = []
    lib.aobundle2_version.restype = c_char_p
    
    # aobundle2_crc32
    lib.aobundle2_crc32.argtypes = [c_void_p, c_size_t]
    lib.aobundle2_crc32.restype = c_uint32
    
    # aodds functions for size calculation
    lib.aodds_calc_dds_size.argtypes = [c_int32, c_int32, c_int32, c_int32]
    lib.aodds_calc_dds_size.restype = c_uint32


# ============================================================================
# Native API Wrapper
# ============================================================================

def is_available() -> bool:
    """Check if the native bundle library is available."""
    try:
        _load_library()
        return True
    except (ImportError, FileNotFoundError, OSError):
        return False


def create_bundle(
    cache_dir: str,
    tile_row: int,
    tile_col: int,
    maptype: str,
    zoom: int,
    chunks_per_side: int = 16,
    output_path: Optional[str] = None
) -> str:
    """
    Create a bundle from individual JPEG cache files.
    
    This consolidates all chunk JPEGs for a tile into a single bundle file.
    
    Args:
        cache_dir: Directory containing cached JPEGs
        tile_row: Tile row coordinate
        tile_col: Tile column coordinate
        maptype: Map source identifier (e.g., "BI", "EOX")
        zoom: Zoom level
        chunks_per_side: Number of chunks per side (default 16)
        output_path: Output bundle path (default: auto-generate)
    
    Returns:
        Path to the created bundle file
    
    Raises:
        RuntimeError: If bundle creation fails
    """
    lib = _load_library()
    
    if output_path is None:
        from ..utils.bundle_paths import get_bundle2_path, ensure_bundle2_dir
        ensure_bundle2_dir(cache_dir, tile_row, tile_col, zoom, maptype)
        output_path = get_bundle2_path(cache_dir, tile_row, tile_col, maptype, zoom)
    
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    success = lib.aobundle2_create(
        cache_dir.encode('utf-8'),
        tile_row,
        tile_col,
        maptype.encode('utf-8'),
        zoom,
        chunks_per_side,
        output_path.encode('utf-8'),
        None  # No result struct
    )
    
    if not success:
        raise RuntimeError(f"Failed to create bundle at {output_path}")
    
    return output_path


def create_bundle_from_data(
    tile_row: int,
    tile_col: int,
    maptype: str,
    zoom: int,
    jpeg_datas: List[Optional[bytes]],
    output_path: str
) -> str:
    """
    Create a bundle from JPEG data arrays.
    
    Args:
        tile_row: Tile row coordinate
        tile_col: Tile column coordinate
        maptype: Map source identifier
        zoom: Zoom level
        jpeg_datas: List of JPEG bytes (None for missing chunks)
        output_path: Output bundle path
    
    Returns:
        Path to the created bundle file
    """
    lib = _load_library()
    
    chunk_count = len(jpeg_datas)
    
    # Create ctypes arrays
    ptr_array = (c_void_p * chunk_count)()
    size_array = (c_uint32 * chunk_count)()
    
    # Keep references to prevent garbage collection
    byte_refs = []
    
    for i, data in enumerate(jpeg_datas):
        if data is not None:
            # Create a ctypes buffer from the bytes
            buf = (c_uint8 * len(data)).from_buffer_copy(data)
            byte_refs.append(buf)
            ptr_array[i] = cast(buf, c_void_p)
            size_array[i] = len(data)
        else:
            ptr_array[i] = None
            size_array[i] = 0
    
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    success = lib.aobundle2_create_from_data(
        tile_row,
        tile_col,
        maptype.encode('utf-8'),
        zoom,
        cast(ptr_array, POINTER(c_void_p)),
        cast(size_array, POINTER(c_uint32)),
        chunk_count,
        output_path.encode('utf-8'),
        None
    )
    
    if not success:
        raise RuntimeError(f"Failed to create bundle from data at {output_path}")
    
    return output_path


def build_dds(
    bundle_path: str,
    target_zoom: int,
    format: str = "BC1",
    missing_color: Tuple[int, int, int] = (66, 77, 55),
    chunks_per_side: int = 16
) -> bytes:
    """
    Build DDS directly from a bundle file (optimal single-call path).
    
    Args:
        bundle_path: Path to bundle file
        target_zoom: Target zoom level
        format: Compression format - "BC1" (DXT1) or "BC3" (DXT5)
        missing_color: RGB tuple for missing chunks
        chunks_per_side: Chunks per side (default 16)
    
    Returns:
        Complete DDS file as bytes
    
    Raises:
        RuntimeError: If DDS build fails
    """
    lib = _load_library()
    
    tile_size = chunks_per_side * 256
    fmt = FORMAT_BC1 if format.upper() in ("BC1", "DXT1") else FORMAT_BC3
    dds_size = lib.aodds_calc_dds_size(tile_size, tile_size, 0, fmt)
    
    # Allocate output buffer
    buffer = (c_uint8 * dds_size)()
    bytes_written = c_uint32()
    
    # Missing color array
    color = (c_uint8 * 3)(missing_color[0], missing_color[1], missing_color[2])
    
    success = lib.aobundle2_build_dds(
        bundle_path.encode('utf-8'),
        target_zoom,
        fmt,
        color,
        cast(buffer, POINTER(c_uint8)),
        dds_size,
        byref(bytes_written)
    )
    
    if not success:
        raise RuntimeError(f"Failed to build DDS from bundle: {bundle_path}")
    
    return bytes(buffer[:bytes_written.value])


def get_fragmentation(bundle_path: str) -> float:
    """
    Get fragmentation ratio for a bundle file.
    
    Args:
        bundle_path: Path to bundle file
    
    Returns:
        Fragmentation ratio (0.0 to 1.0), or -1.0 on error
    """
    lib = _load_library()
    return lib.aobundle2_get_fragmentation(bundle_path.encode('utf-8'))


def needs_compaction(bundle_path: str, threshold: float = BUNDLE2_COMPACTION_THRESHOLD) -> bool:
    """
    Check if a bundle needs compaction.
    
    Args:
        bundle_path: Path to bundle file
        threshold: Fragmentation threshold (default 0.30)
    
    Returns:
        True if compaction is needed
    """
    lib = _load_library()
    result = lib.aobundle2_needs_compaction(bundle_path.encode('utf-8'), threshold)
    return result == 1


def compact(bundle_path: str) -> int:
    """
    Compact a bundle file by removing garbage data.
    
    Args:
        bundle_path: Path to bundle file
    
    Returns:
        Bytes reclaimed, 0 if no compaction needed, -1 on error
    """
    lib = _load_library()
    return lib.aobundle2_compact(bundle_path.encode('utf-8'))


def consolidate_atomic(
    bundle_path: str,
    tile_row: int,
    tile_col: int,
    maptype: str,
    zoom: int,
    jpeg_datas: List[Optional[bytes]],
    source_paths: Optional[List[str]] = None,
    delete_sources: bool = True
) -> bool:
    """
    Atomically consolidate JPEGs into bundle with file locking.
    
    This is the preferred method for consolidation because:
    - Atomic: Uses temp file + rename for safe file replacement
    - Locked: Uses OS file locking to prevent concurrent access issues
    - Safe deletion: Only deletes source files after successful commit
    - Cross-platform: Works on Windows and Unix
    
    Args:
        bundle_path: Path to bundle file (created if doesn't exist)
        tile_row: Tile row coordinate
        tile_col: Tile column coordinate
        maptype: Map type string
        zoom: Zoom level for the chunks
        jpeg_datas: List of JPEG bytes (None for missing chunks)
        source_paths: Optional list of source JPEG file paths to delete
        delete_sources: If True and source_paths provided, delete sources after commit
    
    Returns:
        True on success, False on failure
        
    Thread Safety:
        - Uses OS file locking, safe for concurrent calls
        - Source files only deleted after successful bundle commit
    """
    lib = _load_library()
    if lib is None:
        return False
    
    chunk_count = len(jpeg_datas)
    if chunk_count == 0:
        return False
    
    # Prepare JPEG data arrays
    jpeg_data_arr = (POINTER(c_uint8) * chunk_count)()
    jpeg_sizes_arr = (c_uint32 * chunk_count)()
    
    # Keep references to prevent garbage collection during C call
    data_refs = []
    
    for i, data in enumerate(jpeg_datas):
        if data is not None and len(data) > 0:
            data_buf = (c_uint8 * len(data))(*data)
            data_refs.append(data_buf)
            jpeg_data_arr[i] = cast(data_buf, POINTER(c_uint8))
            jpeg_sizes_arr[i] = len(data)
        else:
            jpeg_data_arr[i] = None
            jpeg_sizes_arr[i] = 0
    
    # Prepare source paths array (if provided)
    source_paths_arr = None
    if source_paths and delete_sources:
        source_paths_arr = (c_char_p * chunk_count)()
        for i, path in enumerate(source_paths):
            if path:
                source_paths_arr[i] = path.encode('utf-8')
            else:
                source_paths_arr[i] = None
    
    # Set up function signature
    lib.aobundle2_consolidate_atomic.argtypes = [
        c_char_p,       # bundle_path
        c_int32,        # tile_row
        c_int32,        # tile_col
        c_char_p,       # maptype
        c_int32,        # zoom
        POINTER(POINTER(c_uint8)),  # jpeg_data
        POINTER(c_uint32),          # jpeg_sizes
        c_int32,        # chunk_count
        POINTER(c_char_p),          # source_paths
        c_int32,        # delete_sources
        c_void_p        # result (NULL for now)
    ]
    lib.aobundle2_consolidate_atomic.restype = c_int32
    
    # Call native function
    success = lib.aobundle2_consolidate_atomic(
        bundle_path.encode('utf-8'),
        tile_row,
        tile_col,
        maptype.encode('utf-8'),
        zoom,
        cast(jpeg_data_arr, POINTER(POINTER(c_uint8))),
        jpeg_sizes_arr,
        chunk_count,
        source_paths_arr,
        1 if (delete_sources and source_paths) else 0,
        None  # No result struct for now
    )
    
    return success == 1


def validate(bundle_path: str) -> bool:
    """
    Validate bundle file integrity.
    
    Args:
        bundle_path: Path to bundle file
    
    Returns:
        True if valid, False if invalid/corrupt
    """
    lib = _load_library()
    return lib.aobundle2_validate(bundle_path.encode('utf-8')) == 1


def get_version() -> str:
    """Get version information for the native bundle library."""
    lib = _load_library()
    return lib.aobundle2_version().decode('utf-8')


# ============================================================================
# Header-Only Quick Access Functions
# ============================================================================
# These functions read only the bundle header and zoom table (~200 bytes)
# without loading the entire file, making them ~1000x faster for large bundles.

def has_zoom_quick(bundle_path: str, zoom: int) -> bool:
    """
    Check if a bundle has a specific zoom level WITHOUT loading the entire file.
    
    Reads only the header (~64 bytes) and zoom table (~12 bytes per zoom level),
    making this ~1000x faster than opening a full Bundle2Python for large bundles.
    
    Args:
        bundle_path: Path to bundle file
        zoom: Zoom level to check
    
    Returns:
        True if bundle exists and contains the specified zoom level
        False if bundle doesn't exist, is invalid, or doesn't have the zoom
    """
    try:
        with open(bundle_path, 'rb') as f:
            # Read header (64 bytes)
            header_data = f.read(BUNDLE2_HEADER_SIZE)
            if len(header_data) < BUNDLE2_HEADER_SIZE:
                return False
            
            # Parse minimum header fields (first 22 bytes)
            header_fmt = '<I H H i i H H H H'
            try:
                (magic, version, flags, tile_row, tile_col,
                 maptype_len, zoom_count, min_zoom, max_zoom) = \
                    struct.unpack(header_fmt, header_data[:22])
            except struct.error:
                return False
            
            # Validate magic
            if magic != BUNDLE2_MAGIC:
                return False
            
            # Quick range check (avoids reading zoom table if obviously out of range)
            if zoom < min_zoom or zoom > max_zoom:
                return False
            
            # If only one zoom level and range check passed, it must be this zoom
            if zoom_count == 1 and min_zoom == max_zoom == zoom:
                return True
            
            # Read zoom table entries
            maptype_padded = (maptype_len + 7) & ~7
            zoom_table_offset = BUNDLE2_HEADER_SIZE + maptype_padded
            
            f.seek(zoom_table_offset)
            
            zoom_entry_size = 12  # struct.calcsize('<H H I I')
            zoom_table_data = f.read(zoom_count * zoom_entry_size)
            
            if len(zoom_table_data) < zoom_count * zoom_entry_size:
                return False
            
            # Check each zoom entry
            for i in range(zoom_count):
                offset = i * zoom_entry_size
                entry_zoom = struct.unpack('<H', zoom_table_data[offset:offset + 2])[0]
                if entry_zoom == zoom:
                    return True
            
            return False
            
    except (OSError, IOError, struct.error):
        return False


def get_bundle_zoom_levels_quick(bundle_path: str) -> List[int]:
    """
    Get list of zoom levels in a bundle WITHOUT loading the entire file.
    
    Args:
        bundle_path: Path to bundle file
    
    Returns:
        List of zoom levels, or empty list if bundle invalid
    """
    try:
        with open(bundle_path, 'rb') as f:
            # Read header
            header_data = f.read(BUNDLE2_HEADER_SIZE)
            if len(header_data) < BUNDLE2_HEADER_SIZE:
                return []
            
            header_fmt = '<I H H i i H H H H'
            try:
                (magic, version, flags, tile_row, tile_col,
                 maptype_len, zoom_count, min_zoom, max_zoom) = \
                    struct.unpack(header_fmt, header_data[:22])
            except struct.error:
                return []
            
            if magic != BUNDLE2_MAGIC:
                return []
            
            # Read zoom table
            maptype_padded = (maptype_len + 7) & ~7
            zoom_table_offset = BUNDLE2_HEADER_SIZE + maptype_padded
            
            f.seek(zoom_table_offset)
            zoom_entry_size = 12
            zoom_table_data = f.read(zoom_count * zoom_entry_size)
            
            zooms = []
            for i in range(zoom_count):
                offset = i * zoom_entry_size
                if offset + 2 <= len(zoom_table_data):
                    entry_zoom = struct.unpack('<H', zoom_table_data[offset:offset + 2])[0]
                    zooms.append(entry_zoom)
            
            return zooms
            
    except (OSError, IOError, struct.error):
        return []


# ============================================================================
# Pure Python Bundle Reader (Fallback)
# ============================================================================

class Bundle2Python:
    """
    Pure Python bundle reader for AOB2 format using memory-mapped I/O.
    
    Memory-mapped access provides:
    - Lazy loading: only accessed pages are read from disk
    - OS-managed caching: efficient memory usage via page cache
    - Zero-copy access: slice operations return views, not copies
    
    This class provides read-only access to AOB2 bundles without requiring
    the native library. Useful for testing and as a fallback.
    
    Usage:
        bundle = Bundle2Python("/path/to/bundle.aob2")
        jpeg_data = bundle.get_chunk(zoom=16, index=0)
        all_jpegs = bundle.get_all_chunks(zoom=16)
        bundle.close()  # Or use as context manager
        
        # Context manager usage (recommended):
        with Bundle2Python("/path/to/bundle.aob2") as bundle:
            jpeg_data = bundle.get_chunk(zoom=16, index=0)
    """
    
    def __init__(self, path: str, use_metadata_cache: bool = True):
        """
        Open a bundle file for reading.
        
        Args:
            path: Path to bundle file
            use_metadata_cache: If True, use global metadata cache for faster opens
        """
        self.path = path
        self._file = None
        self._mmap = None
        self._header = None
        self._maptype = None
        self._zoom_table = []
        self._chunk_indices = {}
        self._use_metadata_cache = use_metadata_cache
        
        self._open()
    
    def _open(self):
        """Open file and create memory map."""
        self._file = open(self.path, 'rb')
        try:
            # Create read-only memory map
            # length=0 maps entire file
            self._mmap = mmap.mmap(
                self._file.fileno(),
                length=0,
                access=mmap.ACCESS_READ
            )
        except (ValueError, OSError) as e:
            # Fallback for empty files or mmap failure
            self._file.close()
            self._file = None
            raise ValueError(f"Cannot mmap bundle: {e}")
        
        # Try to use cached metadata first
        if self._use_metadata_cache:
            cached = _metadata_cache.get(self.path)
            if cached is not None:
                # Use cached metadata (avoids re-parsing)
                self._header = cached.header
                self._maptype = cached.maptype
                self._zoom_table = cached.zoom_table
                self._chunk_indices = cached.chunk_indices
                return
        
        # Parse metadata fresh
        self._parse_header()
        self._parse_zoom_table()
        self._parse_chunk_indices()
        
        # Cache the parsed metadata
        if self._use_metadata_cache:
            metadata = BundleMetadata(
                header=self._header,
                maptype=self._maptype,
                zoom_table=self._zoom_table,
                chunk_indices=self._chunk_indices,
                file_size=len(self._mmap)
            )
            _metadata_cache.put(self.path, metadata)
    
    def _parse_header(self):
        """Parse the 64-byte header from mmap."""
        if len(self._mmap) < BUNDLE2_HEADER_SIZE:
            raise ValueError(f"File too small for header: {len(self._mmap)}")
        
        # Header format (64 bytes):
        # uint32 magic, uint16 version, uint16 flags,
        # int32 tile_row, int32 tile_col,
        # uint16 maptype_len, uint16 zoom_count,
        # uint16 min_zoom, uint16 max_zoom,
        # uint32 total_chunks, uint32 data_section_offset,
        # uint32 garbage_bytes, uint64 last_modified,
        # uint32 checksum, 16 bytes reserved
        header_fmt = '<I H H i i H H H H I I I Q I 16s'
        header_data = struct.unpack(header_fmt, self._mmap[:BUNDLE2_HEADER_SIZE])
        
        magic = header_data[0]
        if magic != BUNDLE2_MAGIC:
            raise ValueError(f"Invalid magic: 0x{magic:08X}, expected 0x{BUNDLE2_MAGIC:08X}")
        
        self._header = {
            'magic': magic,
            'version': header_data[1],
            'flags': BundleFlags(header_data[2]),
            'tile_row': header_data[3],
            'tile_col': header_data[4],
            'maptype_len': header_data[5],
            'zoom_count': header_data[6],
            'min_zoom': header_data[7],
            'max_zoom': header_data[8],
            'total_chunks': header_data[9],
            'data_section_offset': header_data[10],
            'garbage_bytes': header_data[11],
            'last_modified': header_data[12],
            'checksum': header_data[13],
        }
        
        # Parse maptype
        maptype_offset = BUNDLE2_HEADER_SIZE
        maptype_len = self._header['maptype_len']
        self._maptype = self._mmap[maptype_offset:maptype_offset + maptype_len].decode('utf-8')
    
    def _parse_zoom_table(self):
        """Parse the zoom level table from mmap."""
        maptype_padded = (self._header['maptype_len'] + 7) & ~7
        zoom_table_offset = BUNDLE2_HEADER_SIZE + maptype_padded
        
        # Each zoom entry is 12 bytes:
        # uint16 zoom_level, uint16 chunks_per_side, uint32 index_offset, uint32 chunk_count
        zoom_entry_fmt = '<H H I I'
        zoom_entry_size = struct.calcsize(zoom_entry_fmt)
        
        for i in range(self._header['zoom_count']):
            offset = zoom_table_offset + i * zoom_entry_size
            entry_data = struct.unpack(zoom_entry_fmt, 
                                       self._mmap[offset:offset + zoom_entry_size])
            
            self._zoom_table.append({
                'zoom_level': entry_data[0],
                'chunks_per_side': entry_data[1],
                'index_offset': entry_data[2],
                'chunk_count': entry_data[3],
            })
    
    def _parse_chunk_indices(self):
        """Parse chunk indices for all zoom levels from mmap."""
        # Each chunk index entry is 16 bytes:
        # uint32 data_offset, uint32 size, uint16 flags, uint16 quality, uint32 timestamp
        chunk_entry_fmt = '<I I H H I'
        chunk_entry_size = struct.calcsize(chunk_entry_fmt)
        
        for zoom_entry in self._zoom_table:
            zoom = zoom_entry['zoom_level']
            indices = []
            
            for i in range(zoom_entry['chunk_count']):
                offset = zoom_entry['index_offset'] + i * chunk_entry_size
                entry_data = struct.unpack(chunk_entry_fmt,
                                          self._mmap[offset:offset + chunk_entry_size])
                
                indices.append({
                    'data_offset': entry_data[0],
                    'size': entry_data[1],
                    'flags': ChunkFlags(entry_data[2]),
                    'quality': entry_data[3],
                    'timestamp': entry_data[4],
                })
            
            self._chunk_indices[zoom] = indices
    
    @property
    def header(self) -> dict:
        """Get header information."""
        return self._header.copy()
    
    @property
    def maptype(self) -> str:
        """Get map type."""
        return self._maptype
    
    @property
    def zoom_levels(self) -> List[int]:
        """Get list of available zoom levels."""
        return [z['zoom_level'] for z in self._zoom_table]
    
    @property
    def file_size(self) -> int:
        """Get the bundle file size in bytes."""
        if self._mmap is not None:
            return len(self._mmap)
        return 0
    
    @property
    def is_open(self) -> bool:
        """Check if bundle is still open."""
        return self._mmap is not None and not self._mmap.closed
    
    def has_zoom(self, zoom: int) -> bool:
        """Check if bundle has data for a zoom level."""
        return zoom in self._chunk_indices
    
    def get_chunk_count(self, zoom: int) -> int:
        """Get number of chunks at a zoom level."""
        if zoom not in self._chunk_indices:
            return 0
        return len(self._chunk_indices[zoom])
    
    def get_chunk(self, zoom: int, index: int) -> Optional[bytes]:
        """
        Get JPEG data for a specific chunk.
        
        Returns bytes (copy of data) for safety when passing to external code.
        For zero-copy access within controlled scope, use get_chunk_view().
        
        Args:
            zoom: Zoom level
            index: Chunk index
        
        Returns:
            JPEG bytes, or None if chunk is missing
        """
        if zoom not in self._chunk_indices:
            return None
        
        indices = self._chunk_indices[zoom]
        if index < 0 or index >= len(indices):
            return None
        
        entry = indices[index]
        
        # Check if valid data
        if entry['size'] == 0 or (entry['flags'] & ChunkFlags.GARBAGE):
            return None
        
        data_start = self._header['data_section_offset'] + entry['data_offset']
        data_end = data_start + entry['size']
        
        # Return bytes copy (safe for external use)
        return bytes(self._mmap[data_start:data_end])
    
    def get_chunk_view(self, zoom: int, index: int) -> Optional[memoryview]:
        """
        Get zero-copy memoryview of chunk data.
        
        WARNING: The returned memoryview is only valid while the bundle is open!
        Use this only when you control the lifetime and need maximum performance.
        
        Args:
            zoom: Zoom level
            index: Chunk index
        
        Returns:
            memoryview of JPEG data, or None if chunk is missing
        """
        if zoom not in self._chunk_indices:
            return None
        
        indices = self._chunk_indices[zoom]
        if index < 0 or index >= len(indices):
            return None
        
        entry = indices[index]
        
        # Check if valid data
        if entry['size'] == 0 or (entry['flags'] & ChunkFlags.GARBAGE):
            return None
        
        data_start = self._header['data_section_offset'] + entry['data_offset']
        data_end = data_start + entry['size']
        
        # Return memoryview (zero-copy, but only valid while mmap exists)
        return memoryview(self._mmap)[data_start:data_end]
    
    def get_all_chunks(self, zoom: int) -> List[Optional[bytes]]:
        """
        Get all chunk data for a zoom level.
        
        Args:
            zoom: Zoom level
        
        Returns:
            List of JPEG bytes (None for missing chunks)
        """
        if zoom not in self._chunk_indices:
            return []
        
        return [self.get_chunk(zoom, i) for i in range(len(self._chunk_indices[zoom]))]
    
    def close(self, invalidate_cache: bool = False):
        """
        Close the bundle and release resources.
        
        Args:
            invalidate_cache: If True, invalidate cached metadata for this path
                             (use when bundle file might have been modified)
        """
        if self._mmap is not None:
            try:
                self._mmap.close()
            except Exception:
                pass
            self._mmap = None
        
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None
        
        if invalidate_cache:
            _metadata_cache.invalidate(self.path)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()
        return False
    
    def __del__(self):
        """Destructor - cleanup if not explicitly closed."""
        self.close()
    
    def get_chunk_info(self, zoom: int, index: int) -> Optional[dict]:
        """Get metadata for a specific chunk."""
        if zoom not in self._chunk_indices:
            return None
        
        indices = self._chunk_indices[zoom]
        if index < 0 or index >= len(indices):
            return None
        
        return indices[index].copy()


# ============================================================================
# Pure Python Bundle Writer (Fallback)
# ============================================================================

def create_bundle_python(
    cache_dir: str,
    tile_row: int,
    tile_col: int,
    maptype: str,
    zoom: int,
    chunks_per_side: int = 16,
    output_path: Optional[str] = None
) -> str:
    """
    Create a bundle using pure Python (fallback).
    
    Same as create_bundle but doesn't require native library.
    
    Note: tile_row and tile_col are the starting coordinates of the chunk grid,
    NOT tile indices. Chunks are at (tile_col + offset, tile_row + offset).
    """
    if output_path is None:
        # Handle imports for both frozen (PyInstaller) and direct Python execution
        try:
            from autoortho.utils.bundle_paths import get_bundle2_path, ensure_bundle2_dir
        except ImportError:
            from utils.bundle_paths import get_bundle2_path, ensure_bundle2_dir
        ensure_bundle2_dir(cache_dir, tile_row, tile_col, zoom, maptype)
        output_path = get_bundle2_path(cache_dir, tile_row, tile_col, maptype, zoom)
    
    chunk_count = chunks_per_side * chunks_per_side
    
    # Read all JPEG files
    # tile_row/tile_col ARE the starting coordinates of the chunk grid
    jpeg_datas = []
    for i in range(chunk_count):
        chunk_row_offset = i // chunks_per_side
        chunk_col_offset = i % chunks_per_side
        abs_col = tile_col + chunk_col_offset
        abs_row = tile_row + chunk_row_offset
        
        path = os.path.join(cache_dir, f"{abs_col}_{abs_row}_{zoom}_{maptype}.jpg")
        try:
            with open(path, 'rb') as f:
                jpeg_datas.append(f.read())
        except FileNotFoundError:
            jpeg_datas.append(None)
    
    return create_bundle_from_data_python(
        tile_row, tile_col, maptype, zoom, jpeg_datas, output_path
    )


def create_bundle_from_data_python(
    tile_row: int,
    tile_col: int,
    maptype: str,
    zoom: int,
    jpeg_datas: List[Optional[bytes]],
    output_path: str
) -> str:
    """
    Create a bundle from JPEG data arrays using pure Python.
    """
    import math
    
    chunk_count = len(jpeg_datas)
    chunks_per_side = int(math.sqrt(chunk_count))
    
    # Prepare maptype
    maptype_bytes = maptype.encode('utf-8')[:BUNDLE2_MAX_MAPTYPE - 1]
    maptype_padded_len = (len(maptype_bytes) + 7) & ~7
    
    # Calculate layout
    # Header: 64 bytes
    # Maptype: padded to 8 bytes
    # Zoom table: 12 bytes x 1
    # Chunk indices: 16 bytes x chunk_count
    # Data section: variable
    
    zoom_table_offset = BUNDLE2_HEADER_SIZE + maptype_padded_len
    index_offset = zoom_table_offset + 12  # 12 bytes for one zoom entry
    data_offset = index_offset + chunk_count * 16
    
    # Build data section and calculate offsets
    data_parts = []
    chunk_entries = []
    current_data_offset = 0
    timestamp = int(time.time())
    valid_count = 0
    
    for jpeg in jpeg_datas:
        if jpeg:
            chunk_entries.append({
                'data_offset': current_data_offset,
                'size': len(jpeg),
                'flags': ChunkFlags.VALID,
                'quality': 0,
                'timestamp': timestamp,
            })
            data_parts.append(jpeg)
            current_data_offset += len(jpeg)
            valid_count += 1
        else:
            chunk_entries.append({
                'data_offset': 0,
                'size': 0,
                'flags': ChunkFlags.MISSING,
                'quality': 0,
                'timestamp': timestamp,
            })
    
    # Build header (64 bytes - reserved is 16 bytes to match BUNDLE2_HEADER_SIZE)
    header = struct.pack(
        '<I H H i i H H H H I I I Q I 16s',
        BUNDLE2_MAGIC,              # magic
        BUNDLE2_VERSION,            # version
        BundleFlags.MUTABLE,        # flags
        tile_row,                   # tile_row
        tile_col,                   # tile_col
        len(maptype_bytes),         # maptype_len
        1,                          # zoom_count
        zoom,                       # min_zoom
        zoom,                       # max_zoom
        chunk_count,                # total_chunks
        data_offset,                # data_section_offset
        0,                          # garbage_bytes
        int(time.time()),           # last_modified
        0,                          # checksum (placeholder)
        b'\x00' * 16                # reserved (16 bytes to make header 64 bytes)
    )
    
    # Calculate checksum (on header with checksum=0)
    checksum = _crc32_python(header)
    
    # Rebuild header with correct checksum (64 bytes total)
    header = struct.pack(
        '<I H H i i H H H H I I I Q I 16s',
        BUNDLE2_MAGIC,
        BUNDLE2_VERSION,
        BundleFlags.MUTABLE,
        tile_row,
        tile_col,
        len(maptype_bytes),
        1,
        zoom,
        zoom,
        chunk_count,
        data_offset,
        0,
        int(time.time()),
        checksum,
        b'\x00' * 16
    )
    
    # Build zoom table entry
    zoom_entry = struct.pack(
        '<H H I I',
        zoom,               # zoom_level
        chunks_per_side,    # chunks_per_side
        index_offset,       # index_offset
        chunk_count         # chunk_count
    )
    
    # Build chunk index entries
    index_data = b''
    for entry in chunk_entries:
        index_data += struct.pack(
            '<I I H H I',
            entry['data_offset'],
            entry['size'],
            entry['flags'],
            entry['quality'],
            entry['timestamp']
        )
    
    # Build complete buffer for single write (reduces syscall overhead)
    buffer = bytearray()
    buffer.extend(header)
    buffer.extend(maptype_bytes)
    buffer.extend(b'\x00' * (maptype_padded_len - len(maptype_bytes)))
    buffer.extend(zoom_entry)
    buffer.extend(index_data)
    for data in data_parts:
        buffer.extend(data)
    
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write atomically with 256KB buffer and file preallocation
    temp_path = output_path + f'.tmp.{os.getpid()}'
    with open(temp_path, 'wb', buffering=BUNDLE2_WRITE_BUFFER_SIZE) as f:
        _preallocate_file(f, len(buffer))
        f.write(buffer)
    
    # Use atomic replace with retry for Windows file locking
    # os.replace() may fail on Windows if another thread has the file open
    _atomic_replace_with_retry(temp_path, output_path)
    return output_path


def update_bundle_with_zoom(
    bundle_path: str,
    new_zoom: int,
    new_jpeg_datas: List[Optional[bytes]],
    new_chunks_per_side: int
) -> str:
    """
    Update an existing bundle by adding a new zoom level.
    
    This preserves existing zoom levels and adds the new one.
    If the zoom level already exists, it is replaced.
    
    Args:
        bundle_path: Path to existing bundle
        new_zoom: Zoom level to add
        new_jpeg_datas: JPEG data for the new zoom level
        new_chunks_per_side: Chunks per side for the new zoom level
    
    Returns:
        Path to the updated bundle
    """
    import math
    
    # Invalidate any cached metadata for this bundle since we're modifying it
    _metadata_cache.invalidate(bundle_path)
    
    # Read existing bundle data then CLOSE before writing
    # Critical: On Windows, mmap holds the file open and prevents os.replace()
    # Use context manager to ensure file is released before write attempt
    with Bundle2Python(bundle_path, use_metadata_cache=False) as existing:
        header = existing.header
        maptype = existing.maptype  # Copy before closing
        
        # Collect all zoom level data (existing + new)
        zoom_data = {}  # {zoom: {'chunks_per_side': n, 'jpeg_datas': [...]}}
        
        # Get existing zoom levels (get_all_chunks returns copies, safe after close)
        for zoom in existing.zoom_levels:
            chunk_count = existing.get_chunk_count(zoom)
            chunks_per_side = int(math.sqrt(chunk_count))
            jpeg_datas = existing.get_all_chunks(zoom)
            zoom_data[zoom] = {
                'chunks_per_side': chunks_per_side,
                'jpeg_datas': jpeg_datas
            }
    # Bundle is now closed - safe to write to the same path
    
    # Add or replace new zoom level
    zoom_data[new_zoom] = {
        'chunks_per_side': new_chunks_per_side,
        'jpeg_datas': new_jpeg_datas
    }
    
    # Write combined bundle
    return create_multi_zoom_bundle(
        header['tile_row'],
        header['tile_col'],
        maptype,
        zoom_data,
        bundle_path
    )


def merge_bundle_zoom_data(
    bundle_path: str,
    zoom: int,
    new_jpeg_datas: List[Optional[bytes]],
    chunks_per_side: int
) -> str:
    """
    Merge new JPEG data into an existing bundle zoom level.
    
    For each chunk position:
    - If new_data is not None: use new_data (overwrite)
    - If new_data is None: keep existing data (preserve)
    
    This allows incremental filling of missing chunks without losing
    existing data. Essential for handling partial consolidation when
    min_chunk_ratio < 1.0, and for filling in missing chunks when
    users increase the ratio later.
    
    Args:
        bundle_path: Path to existing bundle
        zoom: Zoom level to merge
        new_jpeg_datas: New JPEG data (None = keep existing)
        chunks_per_side: Chunks per side for the zoom level
    
    Returns:
        Path to the updated bundle
    
    Thread Safety:
        Caller must hold the per-bundle lock to prevent concurrent modifications.
        Uses atomic write (temp file + rename) for crash safety.
    """
    import math
    
    # Invalidate any cached metadata for this bundle since we're modifying it
    _metadata_cache.invalidate(bundle_path)
    
    # Read existing bundle data then CLOSE before writing
    # Critical: On Windows, mmap holds the file open and prevents os.replace()
    with Bundle2Python(bundle_path, use_metadata_cache=False) as existing:
        header = existing.header
        maptype = existing.maptype
        
        zoom_data = {}
        
        # Copy all existing zoom levels
        for z in existing.zoom_levels:
            chunk_count = existing.get_chunk_count(z)
            cps = int(math.sqrt(chunk_count))
            zoom_data[z] = {
                'chunks_per_side': cps,
                'jpeg_datas': existing.get_all_chunks(z)
            }
    # Bundle is now closed - safe to write to the same path
    
    # MERGE: Combine existing and new data for target zoom
    if zoom in zoom_data:
        existing_datas = zoom_data[zoom]['jpeg_datas']
        merged = []
        for i, new_data in enumerate(new_jpeg_datas):
            if new_data is not None:
                # Use new data (overwrite existing or fill missing)
                merged.append(new_data)
            elif i < len(existing_datas):
                # Keep existing data (preserve what we have)
                merged.append(existing_datas[i])
            else:
                # No data available
                merged.append(None)
        zoom_data[zoom] = {
            'chunks_per_side': chunks_per_side,
            'jpeg_datas': merged
        }
    else:
        # Zoom doesn't exist in bundle yet - just add it
        zoom_data[zoom] = {
            'chunks_per_side': chunks_per_side,
            'jpeg_datas': new_jpeg_datas
        }
    
    # Write combined bundle with merged data
    return create_multi_zoom_bundle(
        header['tile_row'],
        header['tile_col'],
        maptype,
        zoom_data,
        bundle_path
    )


def create_multi_zoom_bundle(
    tile_row: int,
    tile_col: int,
    maptype: str,
    zoom_data: dict,  # {zoom: {'chunks_per_side': n, 'jpeg_datas': [...]}}
    output_path: str
) -> str:
    """
    Create a bundle with multiple zoom levels.
    
    Args:
        tile_row, tile_col: Tile coordinates
        maptype: Map source identifier
        zoom_data: Dict mapping zoom level to chunks_per_side and jpeg_datas
        output_path: Output bundle path
    
    Returns:
        Path to the created bundle
    """
    if not zoom_data:
        raise ValueError("zoom_data cannot be empty")
    
    zoom_levels = sorted(zoom_data.keys())
    min_zoom = min(zoom_levels)
    max_zoom = max(zoom_levels)
    zoom_count = len(zoom_levels)
    
    # Prepare maptype
    maptype_bytes = maptype.encode('utf-8')[:BUNDLE2_MAX_MAPTYPE - 1]
    maptype_padded_len = (len(maptype_bytes) + 7) & ~7
    
    # Calculate layout
    # Header: 64 bytes
    # Maptype: padded to 8 bytes
    # Zoom table: 12 bytes x zoom_count
    # Chunk indices: 16 bytes x total_chunks (sum across all zooms)
    # Data section: variable
    
    zoom_table_offset = BUNDLE2_HEADER_SIZE + maptype_padded_len
    
    # Process each zoom level to calculate offsets
    total_chunks = 0
    zoom_entries = []
    all_chunk_entries = []
    all_data_parts = []
    current_data_offset = 0
    timestamp = int(time.time())
    
    index_start_offset = zoom_table_offset + 12 * zoom_count
    current_index_offset = index_start_offset
    
    for zoom in zoom_levels:
        zd = zoom_data[zoom]
        chunks_per_side = zd['chunks_per_side']
        jpeg_datas = zd['jpeg_datas']
        chunk_count = len(jpeg_datas)
        
        total_chunks += chunk_count
        
        # Store zoom entry info
        zoom_entries.append({
            'zoom': zoom,
            'chunks_per_side': chunks_per_side,
            'index_offset': current_index_offset,
            'chunk_count': chunk_count
        })
        
        # Process chunks for this zoom
        for jpeg in jpeg_datas:
            if jpeg:
                all_chunk_entries.append({
                    'data_offset': current_data_offset,
                    'size': len(jpeg),
                    'flags': ChunkFlags.VALID,
                    'quality': 0,
                    'timestamp': timestamp,
                })
                all_data_parts.append(jpeg)
                current_data_offset += len(jpeg)
            else:
                all_chunk_entries.append({
                    'data_offset': 0,
                    'size': 0,
                    'flags': ChunkFlags.MISSING,
                    'quality': 0,
                    'timestamp': timestamp,
                })
        
        current_index_offset += chunk_count * 16
    
    data_section_offset = current_index_offset
    
    # Adjust data_offset in chunk entries (relative to data section)
    # They're already relative from 0, which is correct
    
    # Build header
    header = struct.pack(
        '<I H H i i H H H H I I I Q I 16s',
        BUNDLE2_MAGIC,
        BUNDLE2_VERSION,
        BundleFlags.MUTABLE,
        tile_row,
        tile_col,
        len(maptype_bytes),
        zoom_count,
        min_zoom,
        max_zoom,
        total_chunks,
        data_section_offset,
        0,  # garbage_bytes
        timestamp,
        0,  # checksum placeholder
        b'\x00' * 16
    )
    
    # Calculate checksum
    checksum = _crc32_python(header)
    
    # Rebuild header with checksum
    header = struct.pack(
        '<I H H i i H H H H I I I Q I 16s',
        BUNDLE2_MAGIC,
        BUNDLE2_VERSION,
        BundleFlags.MUTABLE,
        tile_row,
        tile_col,
        len(maptype_bytes),
        zoom_count,
        min_zoom,
        max_zoom,
        total_chunks,
        data_section_offset,
        0,
        timestamp,
        checksum,
        b'\x00' * 16
    )
    
    # Build zoom table
    zoom_table_data = b''
    for ze in zoom_entries:
        zoom_table_data += struct.pack(
            '<H H I I',
            ze['zoom'],
            ze['chunks_per_side'],
            ze['index_offset'],
            ze['chunk_count']
        )
    
    # Build chunk indices
    index_data = b''
    for entry in all_chunk_entries:
        index_data += struct.pack(
            '<I I H H I',
            entry['data_offset'],
            entry['size'],
            entry['flags'],
            entry['quality'],
            entry['timestamp']
        )
    
    # Build complete buffer for single write (reduces syscall overhead)
    buffer = bytearray()
    buffer.extend(header)
    buffer.extend(maptype_bytes)
    buffer.extend(b'\x00' * (maptype_padded_len - len(maptype_bytes)))
    buffer.extend(zoom_table_data)
    buffer.extend(index_data)
    for data in all_data_parts:
        buffer.extend(data)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write atomically with 256KB buffer and file preallocation
    temp_path = output_path + f'.tmp.{os.getpid()}'
    with open(temp_path, 'wb', buffering=BUNDLE2_WRITE_BUFFER_SIZE) as f:
        _preallocate_file(f, len(buffer))
        f.write(buffer)
    
    # Use atomic replace with retry for Windows file locking
    # os.replace() may fail on Windows if another thread has the file open
    _atomic_replace_with_retry(temp_path, output_path)
    return output_path


def _crc32_python(data: bytes) -> int:
    """Calculate CRC32 checksum (IEEE 802.3 polynomial)."""
    import binascii
    return binascii.crc32(data) & 0xFFFFFFFF


# ============================================================================
# Bundle2 Class (Thread-safe wrapper)
# ============================================================================

class Bundle2:
    """
    Thread-safe wrapper for AOB2 bundle operations with reader caching.
    
    Caches the Bundle2Python reader and invalidates on file modification,
    eliminating redundant file opens for sequential access patterns.
    
    Provides both native and pure Python implementations with automatic
    fallback to Python when native is not available.
    """
    
    def __init__(self, path: str, create: bool = False, use_native: bool = True):
        """
        Open or create a bundle.
        
        Args:
            path: Path to bundle file
            create: If True, create empty bundle if not exists
            use_native: If True, prefer native implementation
        """
        self.path = path
        self._use_native = use_native and is_available()
        self._lock = threading.Lock()
        self._cached_reader: Optional[Bundle2Python] = None
        self._cached_mtime: Optional[float] = None
        
        if create and not os.path.exists(path):
            # Would need tile info to create - not implemented here
            raise ValueError("Cannot create bundle without tile info; use create_bundle()")
    
    def _get_reader(self) -> Bundle2Python:
        """
        Get or create cached reader, refreshing if file was modified.
        
        Thread-safe: uses lock to prevent race conditions.
        """
        try:
            current_mtime = os.path.getmtime(self.path)
        except OSError:
            current_mtime = None
        
        with self._lock:
            # Check if we need to refresh the reader
            needs_refresh = (
                self._cached_reader is None or
                not self._cached_reader.is_open or
                self._cached_mtime != current_mtime
            )
            
            if needs_refresh:
                # Close old reader if exists
                if self._cached_reader is not None:
                    try:
                        self._cached_reader.close()
                    except Exception:
                        pass
                
                # Create new reader
                self._cached_reader = Bundle2Python(self.path)
                self._cached_mtime = current_mtime
            
            return self._cached_reader
    
    def get_chunk(self, zoom: int, index: int) -> Optional[bytes]:
        """Get JPEG data for a specific chunk."""
        return self._get_reader().get_chunk(zoom, index)
    
    def get_all_chunks(self, zoom: int) -> List[Optional[bytes]]:
        """Get all chunk data for a zoom level."""
        return self._get_reader().get_all_chunks(zoom)
    
    def has_zoom(self, zoom: int) -> bool:
        """Check if bundle has data for a zoom level."""
        return self._get_reader().has_zoom(zoom)
    
    @property
    def zoom_levels(self) -> List[int]:
        """Get list of available zoom levels."""
        return self._get_reader().zoom_levels
    
    @property
    def header(self) -> dict:
        """Get header information."""
        return self._get_reader().header
    
    @property
    def maptype(self) -> str:
        """Get map type."""
        return self._get_reader().maptype
    
    def build_dds(
        self,
        zoom: int,
        format: str = "BC1",
        missing_color: Tuple[int, int, int] = (66, 77, 55),
        chunks_per_side: int = 16
    ) -> bytes:
        """Build DDS from bundle."""
        if self._use_native:
            return build_dds(self.path, zoom, format, missing_color, chunks_per_side)
        else:
            raise NotImplementedError("Pure Python DDS building not implemented in Bundle2")
    
    def invalidate_cache(self):
        """Force reader cache invalidation (e.g., after external modification)."""
        with self._lock:
            if self._cached_reader is not None:
                try:
                    self._cached_reader.close(invalidate_cache=True)
                except Exception:
                    pass
                self._cached_reader = None
                self._cached_mtime = None
            else:
                # Even without a cached reader, invalidate global metadata cache
                _metadata_cache.invalidate(self.path)
    
    def close(self):
        """Close the bundle and release cached reader."""
        self.invalidate_cache()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()
        return False


# ============================================================================
# Convenience Functions
# ============================================================================

def open_bundle(path: str, use_native: bool = True) -> Bundle2:
    """Open an existing bundle file."""
    return Bundle2(path, create=False, use_native=use_native)


def open_python(path: str) -> Bundle2Python:
    """Open a bundle using pure Python (for hybrid mode)."""
    return Bundle2Python(path)


# ============================================================================
# Module Initialization
# ============================================================================

# Try to load library on import to fail fast
try:
    _load_library()
except Exception:
    pass  # Will use Python fallback if not available
