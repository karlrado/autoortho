"""
AoBundle - Python wrapper for native cache bundle operations.

Cache bundles consolidate 256 individual JPEG cache files into a single
bundle file for massively reduced I/O overhead.

Benefits:
- Single file open instead of 256
- Sequential read for optimal disk I/O
- Memory-mappable for zero-copy access
- Atomic writes (temp file + rename)
- Fast index lookup (O(1) for any chunk)

Example usage:
    from autoortho.aopipeline import AoBundle
    
    # Create a bundle from existing cache files
    AoBundle.create_bundle(
        cache_dir="/path/to/cache",
        tile_col=123, tile_row=456,
        maptype="BI", zoom=16,
        output_path="/path/to/bundles/tile_123_456.aob"
    )
    
    # Build DDS directly from bundle (fastest path)
    dds_bytes = AoBundle.build_dds_from_bundle(
        "/path/to/bundles/tile_123_456.aob"
    )
"""

from ctypes import (
    CDLL, POINTER, Structure, byref, cast,
    c_int32, c_uint8, c_uint16, c_uint32, c_char_p, c_void_p, c_size_t
)
from pathlib import Path
from typing import List, Optional, Tuple
import os
import sys

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
    # aobundle_create
    lib.aobundle_create.argtypes = [
        c_char_p,   # cache_dir
        c_int32,    # tile_col
        c_int32,    # tile_row
        c_char_p,   # maptype
        c_int32,    # zoom
        c_int32,    # chunks_per_side
        c_char_p    # output_path
    ]
    lib.aobundle_create.restype = c_int32
    
    # aobundle_build_dds
    lib.aobundle_build_dds.argtypes = [
        c_char_p,           # bundle_path
        c_int32,            # format
        POINTER(c_uint8),   # missing_color
        POINTER(c_uint8),   # dds_output
        c_uint32,           # output_size
        POINTER(c_uint32)   # bytes_written
    ]
    lib.aobundle_build_dds.restype = c_int32
    
    # aobundle_version
    lib.aobundle_version.argtypes = []
    lib.aobundle_version.restype = c_char_p
    
    # Also need aodds functions for size calculation
    lib.aodds_calc_dds_size.argtypes = [c_int32, c_int32, c_int32, c_int32]
    lib.aodds_calc_dds_size.restype = c_uint32


# ============================================================================
# Constants
# ============================================================================

FORMAT_BC1 = 0  # DXT1, no alpha
FORMAT_BC3 = 1  # DXT5, with alpha

BUNDLE_EXTENSION = ".aob"


# ============================================================================
# Public API
# ============================================================================

def create_bundle(
    cache_dir: str,
    tile_col: int,
    tile_row: int,
    maptype: str,
    zoom: int,
    chunks_per_side: int = 16,
    output_path: Optional[str] = None
) -> str:
    """
    Create a bundle from individual JPEG cache files.
    
    This consolidates all chunk JPEGs for a tile into a single bundle file
    for faster loading.
    
    Args:
        cache_dir: Directory containing cached JPEGs
        tile_col: Tile column coordinate
        tile_row: Tile row coordinate
        maptype: Map source identifier (e.g., "BI", "EOX")
        zoom: Zoom level
        chunks_per_side: Number of chunks per side (default 16)
        output_path: Output bundle path (default: auto-generate in cache_dir)
    
    Returns:
        Path to the created bundle file
    
    Raises:
        RuntimeError: If bundle creation fails
    """
    lib = _load_library()
    
    if output_path is None:
        output_path = os.path.join(
            cache_dir, 
            f"tile_{tile_col}_{tile_row}_{zoom}_{maptype}{BUNDLE_EXTENSION}"
        )
    
    success = lib.aobundle_create(
        cache_dir.encode('utf-8'),
        tile_col,
        tile_row,
        maptype.encode('utf-8'),
        zoom,
        chunks_per_side,
        output_path.encode('utf-8')
    )
    
    if not success:
        raise RuntimeError(f"Failed to create bundle at {output_path}")
    
    return output_path


def build_dds_from_bundle(
    bundle_path: str,
    format: str = "BC1",
    missing_color: Tuple[int, int, int] = (66, 77, 55)
) -> bytes:
    """
    Build DDS directly from a bundle file (optimal single-call path).
    
    This is the fastest way to load a tile:
    - Single file open (instead of 256)
    - Memory-mapped access (zero-copy)
    - Parallel JPEG decode + compress
    
    Args:
        bundle_path: Path to bundle file
        format: Compression format - "BC1" (DXT1) or "BC3" (DXT5)
        missing_color: RGB tuple for missing chunks
    
    Returns:
        Complete DDS file as bytes
    
    Raises:
        RuntimeError: If DDS build fails
    """
    import math
    
    lib = _load_library()
    
    # Get bundle info to calculate size
    # Assume 16x16 chunks for now (could read from bundle header)
    chunks_per_side = 16
    tile_size = chunks_per_side * 256
    
    fmt = FORMAT_BC1 if format.upper() in ("BC1", "DXT1") else FORMAT_BC3
    dds_size = lib.aodds_calc_dds_size(tile_size, tile_size, 0, fmt)
    
    # Allocate output buffer
    buffer = (c_uint8 * dds_size)()
    bytes_written = c_uint32()
    
    # Missing color array
    color = (c_uint8 * 3)(missing_color[0], missing_color[1], missing_color[2])
    
    success = lib.aobundle_build_dds(
        bundle_path.encode('utf-8'),
        fmt,
        color,
        cast(buffer, POINTER(c_uint8)),
        dds_size,
        byref(bytes_written)
    )
    
    if not success:
        raise RuntimeError(f"Failed to build DDS from bundle: {bundle_path}")
    
    return bytes(buffer[:bytes_written.value])


def get_bundle_path(
    cache_dir: str,
    tile_col: int,
    tile_row: int,
    zoom: int,
    maptype: str
) -> str:
    """
    Get the expected bundle path for a tile.
    
    Args:
        cache_dir: Cache directory
        tile_col: Tile column
        tile_row: Tile row
        zoom: Zoom level
        maptype: Map type
    
    Returns:
        Expected bundle file path
    """
    return os.path.join(
        cache_dir,
        f"tile_{tile_col}_{tile_row}_{zoom}_{maptype}{BUNDLE_EXTENSION}"
    )


def bundle_exists(
    cache_dir: str,
    tile_col: int,
    tile_row: int,
    zoom: int,
    maptype: str
) -> bool:
    """Check if a bundle exists for the given tile."""
    path = get_bundle_path(cache_dir, tile_col, tile_row, zoom, maptype)
    return os.path.exists(path)


def get_version() -> str:
    """Get version information for the native bundle library."""
    lib = _load_library()
    return lib.aobundle_version().decode('utf-8')


def is_available() -> bool:
    """Check if the native bundle library is available."""
    try:
        _load_library()
        return True
    except (ImportError, FileNotFoundError):
        return False


# ============================================================================
# Pure Python Fallback (for when native is not available)
# ============================================================================

def create_bundle_python(
    cache_dir: str,
    tile_col: int,
    tile_row: int,
    maptype: str,
    zoom: int,
    chunks_per_side: int = 16,
    output_path: Optional[str] = None
) -> str:
    """
    Create a bundle using pure Python (fallback).
    
    Same as create_bundle but doesn't require native library.
    """
    import struct
    
    if output_path is None:
        output_path = os.path.join(
            cache_dir,
            f"tile_{tile_col}_{tile_row}_{zoom}_{maptype}{BUNDLE_EXTENSION}"
        )
    
    chunk_count = chunks_per_side * chunks_per_side
    
    # Read all JPEG files
    jpeg_datas = []
    for i in range(chunk_count):
        chunk_row = i // chunks_per_side
        chunk_col = i % chunks_per_side
        abs_col = tile_col * chunks_per_side + chunk_col
        abs_row = tile_row * chunks_per_side + chunk_row
        
        path = os.path.join(cache_dir, f"{abs_col}_{abs_row}_{zoom}_{maptype}.jpg")
        try:
            with open(path, 'rb') as f:
                jpeg_datas.append(f.read())
        except FileNotFoundError:
            jpeg_datas.append(None)
    
    # Build header
    maptype_bytes = maptype.encode('utf-8')[:64]
    maptype_padded_len = (len(maptype_bytes) + 7) & ~7
    
    header = struct.pack(
        '<I H H i i H H 12s',
        0x31424F41,  # Magic "AOB1"
        1,           # Version
        chunk_count,
        tile_col,
        tile_row,
        zoom,
        len(maptype_bytes),
        b'\x00' * 12  # Reserved
    )
    
    # Build index and data
    index_entries = []
    data_parts = []
    data_offset = 0
    
    for jpeg in jpeg_datas:
        if jpeg:
            index_entries.append(struct.pack('<I I', data_offset, len(jpeg)))
            data_parts.append(jpeg)
            data_offset += len(jpeg)
        else:
            index_entries.append(struct.pack('<I I', 0, 0))
    
    # Write atomically
    temp_path = output_path + '.tmp'
    with open(temp_path, 'wb') as f:
        f.write(header)
        f.write(maptype_bytes)
        f.write(b'\x00' * (maptype_padded_len - len(maptype_bytes)))
        for entry in index_entries:
            f.write(entry)
        for data in data_parts:
            f.write(data)
    
    # Use os.replace() for atomic rename that overwrites existing files on all platforms
    # os.rename() fails on Windows if destination exists (WinError 183)
    try:
        os.replace(temp_path, output_path)
    except OSError:
        # If replace fails (e.g., file locked), clean up temp and re-raise
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise
    return output_path


# ============================================================================
# Module Initialization
# ============================================================================

# Try to load library on import to fail fast
try:
    _load_library()
except Exception:
    pass  # Will use Python fallback if not available

