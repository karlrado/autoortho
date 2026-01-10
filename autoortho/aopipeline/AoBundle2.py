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
from enum import IntFlag
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
import struct
import sys
import threading
import time

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
    
    _lib = CDLL(str(lib_path))
    _lib_path = lib_path
    
    _setup_signatures(_lib)
    
    return _lib


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
        ensure_bundle2_dir(cache_dir, tile_row, tile_col, zoom)
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
# Pure Python Bundle Reader (Fallback)
# ============================================================================

class Bundle2Python:
    """
    Pure Python bundle reader for AOB2 format.
    
    This class provides read-only access to AOB2 bundles without requiring
    the native library. Useful for testing and as a fallback.
    
    Usage:
        bundle = Bundle2Python("/path/to/bundle.aob2")
        jpeg_data = bundle.get_chunk(zoom=16, index=0)
        all_jpegs = bundle.get_all_chunks(zoom=16)
    """
    
    def __init__(self, path: str):
        """Open a bundle file for reading."""
        self.path = path
        self._data = None
        self._header = None
        self._maptype = None
        self._zoom_table = []
        self._chunk_indices = {}
        
        self._load()
    
    def _load(self):
        """Load and parse the bundle file."""
        with open(self.path, 'rb') as f:
            self._data = f.read()
        
        self._parse_header()
        self._parse_zoom_table()
        self._parse_chunk_indices()
    
    def _parse_header(self):
        """Parse the 64-byte header."""
        if len(self._data) < BUNDLE2_HEADER_SIZE:
            raise ValueError(f"File too small for header: {len(self._data)}")
        
        # Header format (64 bytes):
        # uint32 magic, uint16 version, uint16 flags,
        # int32 tile_row, int32 tile_col,
        # uint16 maptype_len, uint16 zoom_count,
        # uint16 min_zoom, uint16 max_zoom,
        # uint32 total_chunks, uint32 data_section_offset,
        # uint32 garbage_bytes, uint64 last_modified,
        # uint32 checksum, 12 bytes reserved
        header_fmt = '<I H H i i H H H H I I I Q I 12s'
        header_data = struct.unpack(header_fmt, self._data[:BUNDLE2_HEADER_SIZE])
        
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
        self._maptype = self._data[maptype_offset:maptype_offset + maptype_len].decode('utf-8')
    
    def _parse_zoom_table(self):
        """Parse the zoom level table."""
        maptype_padded = (self._header['maptype_len'] + 7) & ~7
        zoom_table_offset = BUNDLE2_HEADER_SIZE + maptype_padded
        
        # Each zoom entry is 12 bytes:
        # uint16 zoom_level, uint16 chunks_per_side, uint32 index_offset, uint32 chunk_count
        zoom_entry_fmt = '<H H I I'
        zoom_entry_size = struct.calcsize(zoom_entry_fmt)
        
        for i in range(self._header['zoom_count']):
            offset = zoom_table_offset + i * zoom_entry_size
            entry_data = struct.unpack(zoom_entry_fmt, self._data[offset:offset + zoom_entry_size])
            
            self._zoom_table.append({
                'zoom_level': entry_data[0],
                'chunks_per_side': entry_data[1],
                'index_offset': entry_data[2],
                'chunk_count': entry_data[3],
            })
    
    def _parse_chunk_indices(self):
        """Parse chunk indices for all zoom levels."""
        # Each chunk index entry is 16 bytes:
        # uint32 data_offset, uint32 size, uint16 flags, uint16 quality, uint32 timestamp
        chunk_entry_fmt = '<I I H H I'
        chunk_entry_size = struct.calcsize(chunk_entry_fmt)
        
        for zoom_entry in self._zoom_table:
            zoom = zoom_entry['zoom_level']
            indices = []
            
            for i in range(zoom_entry['chunk_count']):
                offset = zoom_entry['index_offset'] + i * chunk_entry_size
                entry_data = struct.unpack(chunk_entry_fmt, self._data[offset:offset + chunk_entry_size])
                
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
        
        return self._data[data_start:data_end]
    
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
    
    def get_chunk_info(self, zoom: int, index: int) -> Optional[dict]:
        """Get metadata for a specific chunk."""
        if zoom not in self._chunk_indices:
            return None
        
        indices = self._chunk_indices[zoom]
        if index < 0 or index >= len(indices):
            return None
        
        return indices[index].copy()
    
    def close(self):
        """Release memory (optional - called for compatibility)."""
        self._data = None


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
    """
    if output_path is None:
        from ..utils.bundle_paths import get_bundle2_path, ensure_bundle2_dir
        ensure_bundle2_dir(cache_dir, tile_row, tile_col, zoom)
        output_path = get_bundle2_path(cache_dir, tile_row, tile_col, maptype, zoom)
    
    chunk_count = chunks_per_side * chunks_per_side
    
    # Read all JPEG files
    jpeg_datas = []
    for i in range(chunk_count):
        chunk_row_offset = i // chunks_per_side
        chunk_col_offset = i % chunks_per_side
        abs_col = tile_col * chunks_per_side + chunk_col_offset
        abs_row = tile_row * chunks_per_side + chunk_row_offset
        
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
    
    # Build header
    header = struct.pack(
        '<I H H i i H H H H I I I Q I 12s',
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
        b'\x00' * 12                # reserved
    )
    
    # Calculate checksum (on header with checksum=0)
    checksum = _crc32_python(header)
    
    # Rebuild header with correct checksum
    header = struct.pack(
        '<I H H i i H H H H I I I Q I 12s',
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
        b'\x00' * 12
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
    
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write atomically
    temp_path = output_path + f'.tmp.{os.getpid()}'
    with open(temp_path, 'wb') as f:
        f.write(header)
        f.write(maptype_bytes)
        f.write(b'\x00' * (maptype_padded_len - len(maptype_bytes)))
        f.write(zoom_entry)
        f.write(index_data)
        for data in data_parts:
            f.write(data)
    
    os.rename(temp_path, output_path)
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
    Thread-safe wrapper for AOB2 bundle operations.
    
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
        self._python_bundle = None
        
        if create and not os.path.exists(path):
            # Would need tile info to create - not implemented here
            raise ValueError("Cannot create bundle without tile info; use create_bundle()")
        
        if not self._use_native:
            self._python_bundle = Bundle2Python(path)
    
    def get_chunk(self, zoom: int, index: int) -> Optional[bytes]:
        """Get JPEG data for a specific chunk."""
        if self._use_native:
            # Native implementation reads on demand
            bundle = Bundle2Python(self.path)
            return bundle.get_chunk(zoom, index)
        else:
            return self._python_bundle.get_chunk(zoom, index)
    
    def get_all_chunks(self, zoom: int) -> List[Optional[bytes]]:
        """Get all chunk data for a zoom level."""
        if self._use_native:
            bundle = Bundle2Python(self.path)
            return bundle.get_all_chunks(zoom)
        else:
            return self._python_bundle.get_all_chunks(zoom)
    
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
    
    def close(self):
        """Close the bundle."""
        if self._python_bundle:
            self._python_bundle.close()
            self._python_bundle = None


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
