"""
AoDDS.py - Python wrapper for native DDS texture building

Provides high-performance DDS generation that bypasses Python's GIL
by delegating the entire build pipeline to native C code:
- Cache reading + JPEG decoding + image composition
- Mipmap generation + DXT compression
- All in a single native call

Usage:
    from autoortho.aopipeline import AoDDS
    
    # Build a complete DDS tile from cached JPEGs
    dds_bytes = AoDDS.build_tile_native(
        cache_dir="/path/to/cache",
        row=1234, col=5678,
        maptype="BI", zoom=16,
        chunks_per_side=16,
        format="BC1"
    )
"""

from ctypes import (
    CDLL, POINTER, Structure, c_void_p,
    c_char, c_char_p, c_int32, c_uint8, c_uint32, c_double,
    byref, cast
)
import logging
import os
import sys
from typing import Optional, Tuple, NamedTuple

log = logging.getLogger(__name__)

# ============================================================================
# Library Loading
# ============================================================================

_aodds = None
_load_error = None


def _get_lib_path() -> str:
    """Get the path to the native library for the current platform."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if sys.platform == 'darwin':
        lib_subdir = 'macos'
        lib_name = 'libaopipeline.dylib'
    elif sys.platform == 'win32':
        lib_subdir = 'windows'
        lib_name = 'aopipeline.dll'
    else:
        lib_subdir = 'linux'
        lib_name = 'libaopipeline.so'
    
    # Try platform-specific lib directory first
    lib_path = os.path.join(base_dir, 'lib', lib_subdir, lib_name)
    if os.path.exists(lib_path):
        return lib_path
    
    # Fall back to same directory (for development/testing)
    alt_path = os.path.join(base_dir, lib_name)
    if os.path.exists(alt_path):
        return alt_path
    
    raise FileNotFoundError(
        f"Native library not found. Expected at: {lib_path} or {alt_path}"
    )


def _load_library():
    """Load the native library and set up function signatures."""
    global _aodds, _load_error
    
    if _aodds is not None:
        return _aodds
    
    if _load_error is not None:
        raise _load_error
    
    try:
        lib_path = _get_lib_path()
        log.debug(f"Loading aodds native library from: {lib_path}")
        _aodds = CDLL(lib_path)
        
        # Configure function signatures
        _setup_signatures(_aodds)
        
        # Initialize ISPC
        _aodds.aodds_init_ispc()
        
        # Log version info
        version = _aodds.aodds_version()
        log.info(f"Loaded native DDS library: {version.decode()}")
        
        return _aodds
        
    except Exception as e:
        _load_error = ImportError(f"Failed to load aodds native library: {e}")
        log.warning(f"Native DDS library not available: {e}")
        raise _load_error


def _setup_signatures(lib):
    """Configure ctypes function signatures for type safety."""
    
    # aodds_build_tile
    lib.aodds_build_tile.argtypes = [POINTER(DDSTileRequest), c_void_p]
    lib.aodds_build_tile.restype = c_int32
    
    # aodds_calc_dds_size
    lib.aodds_calc_dds_size.argtypes = [c_int32, c_int32, c_int32, c_int32]
    lib.aodds_calc_dds_size.restype = c_uint32
    
    # aodds_calc_mipmap_count
    lib.aodds_calc_mipmap_count.argtypes = [c_int32, c_int32]
    lib.aodds_calc_mipmap_count.restype = c_int32
    
    # aodds_write_header
    lib.aodds_write_header.argtypes = [
        POINTER(c_uint8), c_int32, c_int32, c_int32, c_int32
    ]
    lib.aodds_write_header.restype = c_int32
    
    # aodds_init_ispc
    lib.aodds_init_ispc.argtypes = []
    lib.aodds_init_ispc.restype = c_int32
    
    # aodds_version
    lib.aodds_version.argtypes = []
    lib.aodds_version.restype = c_char_p


# ============================================================================
# Data Structures
# ============================================================================

class DDSStats(Structure):
    """Statistics from DDS building."""
    _fields_ = [
        ('chunks_found', c_int32),
        ('chunks_decoded', c_int32),
        ('chunks_failed', c_int32),
        ('mipmaps_generated', c_int32),
        ('elapsed_ms', c_double),
    ]


class DDSTileRequest(Structure):
    """
    Tile build request structure.
    Maps to dds_tile_request_t in C.
    """
    _fields_ = [
        ('cache_dir', c_char_p),
        ('tile_row', c_int32),
        ('tile_col', c_int32),
        ('maptype', c_char_p),
        ('zoom', c_int32),
        ('chunks_per_side', c_int32),
        ('format', c_int32),
        ('missing_r', c_uint8),
        ('missing_g', c_uint8),
        ('missing_b', c_uint8),
        ('dds_buffer', POINTER(c_uint8)),
        ('dds_buffer_size', c_uint32),
        ('dds_written', c_uint32),
        ('stats', DDSStats),
        ('success', c_int32),
        ('error', c_char * 256),
    ]


class BuildResult(NamedTuple):
    """Result from DDS tile building."""
    data: bytes
    success: bool
    chunks_found: int
    chunks_decoded: int
    chunks_failed: int
    mipmaps: int
    elapsed_ms: float
    error: str = ''


# Format constants
FORMAT_BC1 = 0  # DXT1
FORMAT_BC3 = 1  # DXT5


# ============================================================================
# Public API
# ============================================================================

def build_tile_native(
    cache_dir: str,
    row: int,
    col: int,
    maptype: str,
    zoom: int,
    chunks_per_side: int = 16,
    format: str = "BC1",
    missing_color: Tuple[int, int, int] = (66, 77, 55),
    pool: Optional[c_void_p] = None
) -> bytes:
    """
    Build a complete DDS tile from cached JPEGs using native code.
    
    This is the main entry point for native DDS building. It performs
    the entire pipeline in native code without Python GIL involvement:
    1. Batch read all chunk cache files
    2. Parallel decode all JPEGs
    3. Compose chunks into full tile image
    4. Generate all mipmap levels
    5. Compress each mipmap with ISPC texcomp
    6. Return complete DDS file as bytes
    
    Args:
        cache_dir: Directory containing cached JPEGs
        row: Tile row coordinate
        col: Tile column coordinate
        maptype: Map source identifier (e.g., "BI", "EOX")
        zoom: Zoom level for chunk fetching
        chunks_per_side: Number of chunks per side (default 16)
        format: Compression format - "BC1" (DXT1) or "BC3" (DXT5)
        missing_color: RGB tuple for missing chunks
        pool: Optional buffer pool handle from AoDecode
    
    Returns:
        Complete DDS file as bytes
    
    Raises:
        RuntimeError: If DDS build fails
        ImportError: If native library not available
    """
    lib = _load_library()
    
    # Calculate buffer size
    tile_size = chunks_per_side * 256
    fmt = FORMAT_BC1 if format.upper() in ("BC1", "DXT1") else FORMAT_BC3
    dds_size = lib.aodds_calc_dds_size(tile_size, tile_size, 0, fmt)
    
    # Allocate output buffer
    buffer = (c_uint8 * dds_size)()
    
    # Build request
    request = DDSTileRequest()
    request.cache_dir = cache_dir.encode('utf-8')
    request.tile_row = row
    request.tile_col = col
    request.maptype = maptype.encode('utf-8')
    request.zoom = zoom
    request.chunks_per_side = chunks_per_side
    request.format = fmt
    request.missing_r = missing_color[0]
    request.missing_g = missing_color[1]
    request.missing_b = missing_color[2]
    request.dds_buffer = cast(buffer, POINTER(c_uint8))
    request.dds_buffer_size = dds_size
    
    # Call native function
    pool_handle = pool if pool else None
    if not lib.aodds_build_tile(byref(request), pool_handle):
        error = request.error.decode('utf-8', errors='replace').rstrip('\x00')
        raise RuntimeError(f"DDS build failed: {error}")
    
    # Return DDS bytes
    return bytes(buffer[:request.dds_written])


def build_tile_native_detailed(
    cache_dir: str,
    row: int,
    col: int,
    maptype: str,
    zoom: int,
    chunks_per_side: int = 16,
    format: str = "BC1",
    missing_color: Tuple[int, int, int] = (66, 77, 55),
    pool: Optional[c_void_p] = None
) -> BuildResult:
    """
    Build a DDS tile with detailed result information.
    
    Same as build_tile_native but returns a BuildResult with statistics.
    """
    lib = _load_library()
    
    tile_size = chunks_per_side * 256
    fmt = FORMAT_BC1 if format.upper() in ("BC1", "DXT1") else FORMAT_BC3
    dds_size = lib.aodds_calc_dds_size(tile_size, tile_size, 0, fmt)
    
    buffer = (c_uint8 * dds_size)()
    
    request = DDSTileRequest()
    request.cache_dir = cache_dir.encode('utf-8')
    request.tile_row = row
    request.tile_col = col
    request.maptype = maptype.encode('utf-8')
    request.zoom = zoom
    request.chunks_per_side = chunks_per_side
    request.format = fmt
    request.missing_r = missing_color[0]
    request.missing_g = missing_color[1]
    request.missing_b = missing_color[2]
    request.dds_buffer = cast(buffer, POINTER(c_uint8))
    request.dds_buffer_size = dds_size
    
    pool_handle = pool if pool else None
    success = lib.aodds_build_tile(byref(request), pool_handle)
    
    return BuildResult(
        data=bytes(buffer[:request.dds_written]) if success else b'',
        success=bool(success),
        chunks_found=request.stats.chunks_found,
        chunks_decoded=request.stats.chunks_decoded,
        chunks_failed=request.stats.chunks_failed,
        mipmaps=request.stats.mipmaps_generated,
        elapsed_ms=request.stats.elapsed_ms,
        error=request.error.decode('utf-8', errors='replace').rstrip('\x00') if not success else ''
    )


def calc_dds_size(
    width: int,
    height: int,
    mipmap_count: int = 0,
    format: str = "BC1"
) -> int:
    """
    Calculate required DDS buffer size.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        mipmap_count: Number of mipmaps (0 = auto-calculate)
        format: "BC1" or "BC3"
    
    Returns:
        Required buffer size in bytes
    """
    lib = _load_library()
    fmt = FORMAT_BC1 if format.upper() in ("BC1", "DXT1") else FORMAT_BC3
    return lib.aodds_calc_dds_size(width, height, mipmap_count, fmt)


def calc_mipmap_count(width: int, height: int) -> int:
    """
    Calculate number of mipmap levels for given dimensions.
    
    Args:
        width: Image width
        height: Image height
    
    Returns:
        Number of mipmap levels
    """
    lib = _load_library()
    return lib.aodds_calc_mipmap_count(width, height)


def init_ispc() -> bool:
    """
    Initialize the ISPC compression library.
    
    Returns:
        True if ISPC available, False if using fallback compression
    """
    lib = _load_library()
    return bool(lib.aodds_init_ispc())


def get_version() -> str:
    """Get version information for the native DDS library."""
    lib = _load_library()
    return lib.aodds_version().decode('utf-8')


def is_available() -> bool:
    """Check if the native DDS library is available."""
    try:
        _load_library()
        return True
    except (ImportError, FileNotFoundError):
        return False


# ============================================================================
# Hybrid Pipeline: Python reads files, Native decodes + compresses
# ============================================================================

def build_from_jpegs(
    jpeg_datas: list,
    format: str = "BC1",
    missing_color: Tuple[int, int, int] = (66, 77, 55),
    pool: Optional[c_void_p] = None
) -> bytes:
    """
    Build DDS from pre-read JPEG data (HYBRID APPROACH).
    
    This is the optimal entry point for high-performance tile building:
    - Call this AFTER reading cache files in Python (which is fast)
    - Native code handles decode + compose + compress (where parallelism helps)
    
    This avoids:
    - File I/O overhead in native code
    - ctypes path string overhead
    - Thread overhead for small file reads
    
    Args:
        jpeg_datas: List of JPEG bytes (None or b'' for missing chunks)
                    Length must be a perfect square (16, 64, 256, etc.)
        format: Compression format - "BC1" (DXT1) or "BC3" (DXT5)
        missing_color: RGB tuple for missing chunks
        pool: Optional buffer pool handle from AoDecode
    
    Returns:
        Complete DDS file as bytes
    
    Raises:
        RuntimeError: If DDS build fails
        ValueError: If chunk_count is not a perfect square
    
    Example:
        # Python reads files (fast for cached files)
        jpeg_datas = []
        for chunk in chunks:
            try:
                jpeg_datas.append(Path(chunk.cache_path).read_bytes())
            except FileNotFoundError:
                jpeg_datas.append(None)
        
        # Native builds DDS (parallel decode + compress)
        dds_bytes = build_from_jpegs(jpeg_datas)
    """
    import math
    
    lib = _load_library()
    chunk_count = len(jpeg_datas)
    
    if chunk_count == 0:
        raise ValueError("No JPEG data provided")
    
    # Verify perfect square
    chunks_per_side = int(math.sqrt(chunk_count))
    if chunks_per_side * chunks_per_side != chunk_count:
        raise ValueError(
            f"chunk_count must be a perfect square, got {chunk_count}"
        )
    
    # Calculate buffer size
    tile_size = chunks_per_side * 256
    fmt = FORMAT_BC1 if format.upper() in ("BC1", "DXT1") else FORMAT_BC3
    dds_size = lib.aodds_calc_dds_size(tile_size, tile_size, 0, fmt)
    
    # Allocate output buffer
    buffer = (c_uint8 * dds_size)()
    
    # Build arrays for C
    # Create array of pointers to JPEG data
    jpeg_ptrs = (POINTER(c_uint8) * chunk_count)()
    jpeg_sizes = (c_uint32 * chunk_count)()
    
    # Keep references to prevent GC
    jpeg_buffers = []
    
    for i, data in enumerate(jpeg_datas):
        if data and len(data) > 0:
            # Create a buffer and keep reference
            buf = (c_uint8 * len(data)).from_buffer_copy(data)
            jpeg_buffers.append(buf)
            jpeg_ptrs[i] = cast(buf, POINTER(c_uint8))
            jpeg_sizes[i] = len(data)
        else:
            jpeg_ptrs[i] = None
            jpeg_sizes[i] = 0
    
    bytes_written = c_uint32()
    pool_handle = pool if pool else None
    
    # Setup function signature if not already done
    if not hasattr(lib, '_hybrid_setup_done'):
        lib.aodds_build_from_jpegs.argtypes = [
            POINTER(POINTER(c_uint8)),  # jpeg_data
            POINTER(c_uint32),          # jpeg_sizes
            c_int32,                    # chunk_count
            c_int32,                    # format
            c_uint8, c_uint8, c_uint8,  # missing color
            POINTER(c_uint8),           # dds_output
            c_uint32,                   # output_size
            POINTER(c_uint32),          # bytes_written
            c_void_p                    # pool
        ]
        lib.aodds_build_from_jpegs.restype = c_int32
        lib._hybrid_setup_done = True
    
    # Call native function
    success = lib.aodds_build_from_jpegs(
        jpeg_ptrs,
        jpeg_sizes,
        chunk_count,
        fmt,
        missing_color[0],
        missing_color[1],
        missing_color[2],
        cast(buffer, POINTER(c_uint8)),
        dds_size,
        byref(bytes_written),
        pool_handle
    )
    
    if not success:
        raise RuntimeError("Failed to build DDS from JPEGs")
    
    return bytes(buffer[:bytes_written.value])


# ============================================================================
# Module initialization
# ============================================================================

# Try to load library on import to fail fast
try:
    _load_library()
except Exception:
    pass  # Will raise on first use if not available

