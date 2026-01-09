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
import threading
from typing import Optional, Tuple, NamedTuple, Union, List

import numpy as np

log = logging.getLogger(__name__)

# Format constants (match dds_format_t in aodds.h)
FORMAT_BC1 = 0  # DXT1
FORMAT_BC3 = 1  # DXT5


# ============================================================================
# Buffer Pool for Zero-Copy DDS Building
# ============================================================================

class DDSBufferPool:
    """
    Thread-safe pool of reusable numpy buffers for DDS building.
    
    Eliminates per-call allocation overhead (~15ms) and avoids copying
    data back to Python (~65ms) by reusing pre-allocated numpy arrays.
    
    Usage:
        pool = DDSBufferPool(max_dds_size=11_200_000)  # ~10.7 MB for 4096x4096
        
        # Acquire buffer, use it, release it
        buffer, buffer_id = pool.acquire()
        try:
            result = build_tile_to_buffer(buffer, ...)
            # buffer now contains DDS data - use it directly or copy if needed
            dds_bytes = bytes(buffer[:result.bytes_written])
        finally:
            pool.release(buffer_id)
    
    For even better performance, use the buffer directly without copying:
        memoryview(buffer[:result.bytes_written])
    """
    
    # Standard DDS sizes for common tile configurations
    # These include DDS header (128 bytes) + all mipmaps + safety margin
    # Values calculated from aodds_calc_dds_size() + 1KB padding for safety
    SIZE_8192x8192_BC1 = 44_740_000  # 32x32 chunks, BC1 format with mipmaps (~42.67 MB)
    SIZE_4096x4096_BC1 = 11_186_000  # 16x16 chunks, BC1 format with mipmaps (~10.67 MB)
    SIZE_2048x2048_BC1 = 2_797_500   # 8x8 chunks, BC1 format with mipmaps (~2.67 MB)
    SIZE_1024x1024_BC1 = 700_000     # 4x4 chunks, BC1 format with mipmaps (~0.67 MB)
    
    def __init__(self, buffer_size: int = SIZE_4096x4096_BC1, pool_size: int = 4):
        """
        Create a buffer pool.
        
        Args:
            buffer_size: Size of each buffer in bytes (default: 4096x4096 BC1)
            pool_size: Number of buffers to pre-allocate
        """
        self._buffer_size = buffer_size
        self._pool_size = pool_size
        self._lock = threading.Lock()
        
        # Pre-allocate numpy arrays
        self._buffers = [
            np.zeros(buffer_size, dtype=np.uint8)
            for _ in range(pool_size)
        ]
        self._available = list(range(pool_size))
        self._in_use = set()
        
        log.debug(f"DDSBufferPool: created {pool_size} buffers of {buffer_size:,} bytes each")
    
    def acquire(self, timeout: float = 5.0) -> Tuple[np.ndarray, int]:
        """
        Acquire a buffer from the pool.
        
        Args:
            timeout: Maximum time to wait for a buffer (seconds)
        
        Returns:
            Tuple of (buffer, buffer_id)
            
        Raises:
            TimeoutError: If no buffer available within timeout
        """
        import time
        start = time.monotonic()
        
        while True:
            with self._lock:
                if self._available:
                    buffer_id = self._available.pop()
                    self._in_use.add(buffer_id)
                    return self._buffers[buffer_id], buffer_id
            
            if time.monotonic() - start > timeout:
                raise TimeoutError(
                    f"DDSBufferPool: no buffer available after {timeout}s "
                    f"({len(self._in_use)}/{self._pool_size} in use)"
                )
            
            # Brief sleep before retry
            time.sleep(0.001)
    
    def release(self, buffer_id: int) -> None:
        """
        Release a buffer back to the pool.
        
        Args:
            buffer_id: The buffer ID returned from acquire()
        """
        with self._lock:
            if buffer_id in self._in_use:
                self._in_use.remove(buffer_id)
                self._available.append(buffer_id)
            else:
                log.warning(f"DDSBufferPool: releasing buffer {buffer_id} that wasn't in use")
    
    def try_acquire(self) -> Optional[Tuple[np.ndarray, int]]:
        """
        Try to acquire a buffer without blocking.
        
        Returns:
            Tuple of (buffer, buffer_id) if available, None otherwise
        """
        with self._lock:
            if self._available:
                buffer_id = self._available.pop()
                self._in_use.add(buffer_id)
                return self._buffers[buffer_id], buffer_id
            return None
    
    @property
    def buffer_size(self) -> int:
        """Size of each buffer in bytes."""
        return self._buffer_size
    
    @property
    def available_count(self) -> int:
        """Number of buffers currently available."""
        with self._lock:
            return len(self._available)
    
    @property
    def in_use_count(self) -> int:
        """Number of buffers currently in use."""
        with self._lock:
            return len(self._in_use)
    
    def get_ctypes_ptr(self, buffer: np.ndarray) -> POINTER(c_uint8):
        """
        Get a ctypes pointer to a numpy buffer for passing to C.
        
        Args:
            buffer: A numpy array from this pool
            
        Returns:
            ctypes pointer suitable for passing to native functions
        """
        return buffer.ctypes.data_as(POINTER(c_uint8))


# Global default buffer pool (lazily initialized)
_default_pool: Optional[DDSBufferPool] = None
_default_pool_lock = threading.Lock()


def get_default_pool() -> DDSBufferPool:
    """
    Get or create the default global buffer pool.
    
    Returns:
        The default DDSBufferPool instance
    """
    global _default_pool
    if _default_pool is None:
        with _default_pool_lock:
            if _default_pool is None:
                _default_pool = DDSBufferPool()
    return _default_pool

# ============================================================================
# Streaming Builder Structures
# ============================================================================

class BuilderConfig(Structure):
    """
    Builder configuration structure.
    Maps to aodds_builder_config_t in C.
    """
    _fields_ = [
        ('chunks_per_side', c_int32),
        ('format', c_int32),
        ('missing_r', c_uint8),
        ('missing_g', c_uint8),
        ('missing_b', c_uint8),
        ('nocopy_mode', c_uint8),  # 1 = zero-copy mode (Python owns JPEG memory)
    ]


class BuilderStatus(Structure):
    """
    Builder status structure for monitoring progress.
    Maps to aodds_builder_status_t in C.
    """
    _fields_ = [
        ('chunks_total', c_int32),
        ('chunks_received', c_int32),
        ('chunks_decoded', c_int32),
        ('chunks_failed', c_int32),
        ('chunks_fallback', c_int32),
        ('chunks_missing', c_int32),
    ]


class StreamingBuilderResult(NamedTuple):
    """Result from StreamingBuilder.finalize()."""
    success: bool
    bytes_written: int
    error: str = ""


class StreamingBuilder:
    """
    Wrapper around native aodds_builder_t.
    
    Manages incremental chunk feeding and finalization for streaming DDS builds.
    Chunks can be added as they become available (JPEG data, fallback images,
    or marked as missing), and the final DDS is generated when finalize() is called.
    
    Usage:
        # Create builder via pool
        builder = builder_pool.acquire(config)
        
        # Add chunks as they arrive
        for idx, jpeg_bytes in chunks:
            builder.add_chunk(idx, jpeg_bytes)
        
        # Add fallback images for missing chunks
        for idx, rgba_data in fallbacks:
            builder.add_fallback_image(idx, rgba_data)
        
        # Mark remaining chunks as missing
        for idx in missing_indices:
            builder.mark_missing(idx)
        
        # Finalize and get DDS
        buffer = pool.acquire_buffer()
        result = builder.finalize(buffer)
        
        # Return builder to pool
        builder.release()
    """
    
    def __init__(self, handle: c_void_p, pool_ref: 'StreamingBuilderPool', 
                 lib, config: BuilderConfig):
        self._handle = handle
        self._pool_ref = pool_ref
        self._lib = lib
        self._config = config
        self._finalized = False
        self._released = False
        self._jpeg_refs = []  # Keep JPEG references alive for zero-copy mode
    
    def add_chunk(self, index: int, jpeg_data: bytes) -> bool:
        """
        Add JPEG data for a chunk.
        
        The JPEG data is stored for deferred parallel decode at finalize time.
        This enables faster throughput by decoding all chunks in parallel with OpenMP.
        
        Args:
            index: Chunk index in row-major order (0 to chunks_per_side^2 - 1)
            jpeg_data: JPEG bytes
            
        Returns:
            True if chunk was stored, False if failed or chunk already set
            
        Thread Safety: Thread-safe - multiple threads can add chunks.
        """
        if self._released:
            raise RuntimeError("StreamingBuilder has been released")
        
        if not jpeg_data:
            return False
        
        jpeg_ptr = cast(jpeg_data, POINTER(c_uint8))
        result = self._lib.aodds_builder_add_chunk(
            self._handle,
            c_int32(index),
            jpeg_ptr,
            c_uint32(len(jpeg_data))
        )
        return bool(result)
    
    def add_chunks_batch(self, chunks: List[Tuple[int, bytes]]) -> int:
        """
        Add multiple JPEG chunks in a single native call.
        
        More efficient than calling add_chunk() repeatedly due to reduced
        Python/C crossing overhead. Chunks are stored for deferred parallel
        decode at finalize time.
        
        Args:
            chunks: List of (index, jpeg_bytes) tuples
            
        Returns:
            Number of chunks successfully added
            
        Thread Safety: Thread-safe.
        """
        if self._released:
            raise RuntimeError("StreamingBuilder has been released")
        
        if not chunks:
            return 0
        
        count = len(chunks)
        
        # Build arrays for C call
        indices = (c_int32 * count)()
        jpeg_ptrs = (c_void_p * count)()
        jpeg_sizes = (c_uint32 * count)()
        
        # Keep references to bytes objects to prevent GC during call
        # (No copy needed - cast directly to pointer, bytes are immutable
        # and CPython doesn't move objects in memory)
        jpeg_refs = []
        
        for i, (idx, jpeg_data) in enumerate(chunks):
            indices[i] = idx
            jpeg_sizes[i] = len(jpeg_data) if jpeg_data else 0
            
            if jpeg_data:
                # Keep reference alive and cast directly - NO COPY
                jpeg_refs.append(jpeg_data)
                jpeg_ptrs[i] = cast(jpeg_data, c_void_p).value
            else:
                jpeg_ptrs[i] = None
        
        result = self._lib.aodds_builder_add_chunks_batch(
            self._handle,
            c_int32(count),
            indices,
            cast(jpeg_ptrs, POINTER(c_void_p)),
            jpeg_sizes
        )
        
        return result
    
    def add_chunks_batch_nocopy(self, chunks: list, jpeg_refs_out: list = None) -> int:
        """
        Add multiple JPEG chunks in a single call using ZERO-COPY mode.
        
        ZERO-COPY: C stores pointers directly without copying. The caller MUST
        ensure that the JPEG bytes objects remain alive until finalize() completes.
        
        This is faster than add_chunks_batch when Python already holds references
        that outlive the builder (e.g., chunk.data held by Tile objects).
        
        Args:
            chunks: List of (index, jpeg_bytes) tuples
            jpeg_refs_out: Optional list to append JPEG byte references to
                           (for caller to maintain references until finalize)
            
        Returns:
            Number of chunks successfully added
            
        Thread Safety: Thread-safe.
        
        WARNING: Memory safety is caller's responsibility! If any bytes object
        is garbage collected before finalize(), C will read freed memory.
        """
        if self._released:
            raise RuntimeError("StreamingBuilder has been released")
        
        if not chunks:
            return 0
        
        count = len(chunks)
        
        # Build arrays for C call
        indices = (c_int32 * count)()
        jpeg_ptrs = (c_void_p * count)()
        jpeg_sizes = (c_uint32 * count)()
        
        # CRITICAL: Keep references alive! Either caller provides list,
        # or we store internally
        refs = jpeg_refs_out if jpeg_refs_out is not None else []
        
        for i, (idx, jpeg_data) in enumerate(chunks):
            indices[i] = idx
            jpeg_sizes[i] = len(jpeg_data) if jpeg_data else 0
            
            if jpeg_data:
                # Keep reference and cast directly to pointer
                refs.append(jpeg_data)
                jpeg_ptrs[i] = cast(jpeg_data, c_void_p).value
            else:
                jpeg_ptrs[i] = None
        
        # Call the nocopy version - C stores pointers without copying
        result = self._lib.aodds_builder_add_chunks_batch_nocopy(
            self._handle,
            c_int32(count),
            indices,
            cast(jpeg_ptrs, POINTER(c_void_p)),
            jpeg_sizes
        )
        
        # If no external list provided, store internally to keep alive
        if jpeg_refs_out is None and refs:
            if not hasattr(self, '_jpeg_refs'):
                self._jpeg_refs = []
            self._jpeg_refs.extend(refs)
        
        return result
    
    def add_fallback_image(self, index: int, rgba_data: bytes,
                           width: int = 256, height: int = 256) -> bool:
        """
        Add a pre-decoded fallback image for a chunk.
        
        Used when Python has resolved a fallback (disk cache, mipmap scale, network).
        
        Args:
            index: Chunk index in row-major order
            rgba_data: RGBA pixel data (width * height * 4 bytes)
            width: Image width (must be 256)
            height: Image height (must be 256)
            
        Returns:
            True on success, False on failure
            
        Thread Safety: Thread-safe.
        """
        if self._released:
            raise RuntimeError("StreamingBuilder has been released")
        
        if not rgba_data or len(rgba_data) != width * height * 4:
            return False
        
        rgba_ptr = cast(rgba_data, POINTER(c_uint8))
        result = self._lib.aodds_builder_add_fallback_image(
            self._handle,
            c_int32(index),
            rgba_ptr,
            c_int32(width),
            c_int32(height)
        )
        return bool(result)
    
    def mark_missing(self, index: int) -> None:
        """
        Mark a chunk as permanently missing.
        
        Call this when all fallbacks have been exhausted. The chunk position
        will be filled with missing_color during finalization.
        
        Args:
            index: Chunk index in row-major order
            
        Thread Safety: Thread-safe.
        """
        if self._released:
            raise RuntimeError("StreamingBuilder has been released")
        
        self._lib.aodds_builder_mark_missing(self._handle, c_int32(index))
    
    def get_status(self) -> dict:
        """
        Get current builder status.
        
        Returns:
            Dictionary with status fields:
            - chunks_total: Total chunks expected
            - chunks_received: Chunks added or marked
            - chunks_decoded: Successfully decoded JPEGs
            - chunks_failed: JPEG decode failures
            - chunks_fallback: Chunks using fallback images
            - chunks_missing: Chunks marked as missing
            
        Thread Safety: Thread-safe.
        """
        if self._released:
            raise RuntimeError("StreamingBuilder has been released")
        
        status = BuilderStatus()
        self._lib.aodds_builder_get_status(self._handle, byref(status))
        return {
            'chunks_total': status.chunks_total,
            'chunks_received': status.chunks_received,
            'chunks_decoded': status.chunks_decoded,
            'chunks_failed': status.chunks_failed,
            'chunks_fallback': status.chunks_fallback,
            'chunks_missing': status.chunks_missing,
        }
    
    def is_complete(self) -> bool:
        """
        Check if all chunks have been processed.
        
        Returns:
            True if all chunks are added/marked, False otherwise
            
        Thread Safety: Thread-safe.
        """
        if self._released:
            raise RuntimeError("StreamingBuilder has been released")
        
        return bool(self._lib.aodds_builder_is_complete(self._handle))
    
    def finalize(self, buffer: np.ndarray) -> StreamingBuilderResult:
        """
        Finalize and write DDS to buffer.
        
        Performs:
        1. Fill missing chunks with missing_color
        2. Compose all chunks into tile
        3. Generate all mipmap levels
        4. Compress with BC1/BC3
        5. Write complete DDS to buffer
        
        Args:
            buffer: Pre-allocated numpy buffer (use DDSBufferPool)
            
        Returns:
            StreamingBuilderResult with success status and bytes_written
            
        Thread Safety: NOT thread-safe with add_chunk. Ensure all adding complete.
        """
        if self._released:
            raise RuntimeError("StreamingBuilder has been released")
        if self._finalized:
            raise RuntimeError("StreamingBuilder already finalized")
        
        bytes_written = c_uint32()
        buffer_ptr = buffer.ctypes.data_as(POINTER(c_uint8))
        
        result = self._lib.aodds_builder_finalize(
            self._handle,
            buffer_ptr,
            c_uint32(len(buffer)),
            byref(bytes_written)
        )
        
        self._finalized = True
        
        if result:
            return StreamingBuilderResult(
                success=True,
                bytes_written=bytes_written.value
            )
        else:
            return StreamingBuilderResult(
                success=False,
                bytes_written=0,
                error="Failed to finalize DDS"
            )
    
    def finalize_to_file(self, output_path: str) -> Tuple[bool, int]:
        """
        Finalize and write DDS directly to file.
        
        Zero-copy optimization for prefetch cache.
        
        Args:
            output_path: Path to output DDS file
            
        Returns:
            Tuple of (success, bytes_written)
            
        Thread Safety: NOT thread-safe with add_chunk.
        """
        if self._released:
            raise RuntimeError("StreamingBuilder has been released")
        if self._finalized:
            raise RuntimeError("StreamingBuilder already finalized")
        
        bytes_written = c_uint32()
        output_path_bytes = output_path.encode('utf-8')
        
        result = self._lib.aodds_builder_finalize_to_file(
            self._handle,
            output_path_bytes,
            byref(bytes_written)
        )
        
        self._finalized = True
        
        return bool(result), bytes_written.value
    
    def release(self) -> None:
        """
        Return builder to pool for reuse.
        
        Must be called when done with the builder (use try/finally).
        """
        if not self._released and self._pool_ref is not None:
            self._pool_ref._return_builder(self._handle)
            self._released = True
            self._handle = None
            # Clear JPEG refs now that finalize has completed
            self._jpeg_refs.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


class StreamingBuilderPool:
    """
    Pool of reusable StreamingBuilder instances.
    
    Reduces allocation overhead for high-frequency tile builds by reusing
    native builder instances.
    
    Usage:
        pool = StreamingBuilderPool(pool_size=4)
        
        # Acquire builder
        builder = pool.acquire(config={'chunks_per_side': 16, 'format': 'BC1'})
        
        try:
            # Use builder...
            pass
        finally:
            builder.release()
        
        # Or use context manager
        with pool.acquire(config) as builder:
            # Use builder...
            pass
    """
    
    def __init__(self, pool_size: int = 4, decode_pool: c_void_p = None):
        """
        Create a streaming builder pool.
        
        Args:
            pool_size: Number of builders to pre-allocate
            decode_pool: Optional native decode buffer pool (aodecode_pool_t*)
        """
        self._pool_size = pool_size
        self._decode_pool = decode_pool
        self._lock = threading.Lock()
        self._available: list = []
        self._lib = None
        self._initialized = False
        
        log.debug(f"StreamingBuilderPool: created with pool_size={pool_size}")
    
    def _ensure_initialized(self):
        """Lazy initialization of native library and function signatures."""
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            self._lib = _load_library()
            
            # Setup builder function signatures
            self._lib.aodds_builder_create.argtypes = [
                POINTER(BuilderConfig), c_void_p
            ]
            self._lib.aodds_builder_create.restype = c_void_p
            
            self._lib.aodds_builder_reset.argtypes = [
                c_void_p, POINTER(BuilderConfig)
            ]
            self._lib.aodds_builder_reset.restype = None
            
            self._lib.aodds_builder_destroy.argtypes = [c_void_p]
            self._lib.aodds_builder_destroy.restype = None
            
            self._lib.aodds_builder_add_chunk.argtypes = [
                c_void_p, c_int32, POINTER(c_uint8), c_uint32
            ]
            self._lib.aodds_builder_add_chunk.restype = c_int32
            
            # Batch API for adding multiple chunks in one call
            self._lib.aodds_builder_add_chunks_batch.argtypes = [
                c_void_p,           # builder
                c_int32,            # count
                POINTER(c_int32),   # indices array
                POINTER(c_void_p),  # jpeg_data array (array of pointers)
                POINTER(c_uint32),  # jpeg_sizes array
            ]
            self._lib.aodds_builder_add_chunks_batch.restype = c_int32
            
            # Zero-copy batch API - C stores pointers only, caller owns memory
            self._lib.aodds_builder_add_chunks_batch_nocopy.argtypes = [
                c_void_p,           # builder
                c_int32,            # count
                POINTER(c_int32),   # indices array
                POINTER(c_void_p),  # jpeg_data array (array of pointers)
                POINTER(c_uint32),  # jpeg_sizes array
            ]
            self._lib.aodds_builder_add_chunks_batch_nocopy.restype = c_int32
            
            self._lib.aodds_builder_add_fallback_image.argtypes = [
                c_void_p, c_int32, POINTER(c_uint8), c_int32, c_int32
            ]
            self._lib.aodds_builder_add_fallback_image.restype = c_int32
            
            self._lib.aodds_builder_mark_missing.argtypes = [c_void_p, c_int32]
            self._lib.aodds_builder_mark_missing.restype = None
            
            self._lib.aodds_builder_get_status.argtypes = [
                c_void_p, POINTER(BuilderStatus)
            ]
            self._lib.aodds_builder_get_status.restype = None
            
            self._lib.aodds_builder_is_complete.argtypes = [c_void_p]
            self._lib.aodds_builder_is_complete.restype = c_int32
            
            self._lib.aodds_builder_finalize.argtypes = [
                c_void_p, POINTER(c_uint8), c_uint32, POINTER(c_uint32)
            ]
            self._lib.aodds_builder_finalize.restype = c_int32
            
            self._lib.aodds_builder_finalize_to_file.argtypes = [
                c_void_p, c_char_p, POINTER(c_uint32)
            ]
            self._lib.aodds_builder_finalize_to_file.restype = c_int32
            
            self._initialized = True
    
    def acquire(self, config: dict, timeout: float = 5.0) -> Optional[StreamingBuilder]:
        """
        Acquire a builder from the pool.
        
        Args:
            config: Builder configuration dict:
                - chunks_per_side: int (typically 16)
                - format: str ("BC1" or "BC3")
                - missing_color: tuple of (r, g, b) bytes
            timeout: Maximum wait time in seconds
            
        Returns:
            StreamingBuilder instance, or None if pool exhausted
        """
        self._ensure_initialized()
        
        # Build config structure
        fmt = FORMAT_BC1 if config.get('format', 'BC1').upper() in ('BC1', 'DXT1') else FORMAT_BC3
        missing_color = config.get('missing_color', (128, 128, 128))
        nocopy_mode = 1 if config.get('nocopy_mode', False) else 0
        
        c_config = BuilderConfig(
            chunks_per_side=config.get('chunks_per_side', 16),
            format=fmt,
            missing_r=missing_color[0],
            missing_g=missing_color[1],
            missing_b=missing_color[2],
            nocopy_mode=nocopy_mode,
        )
        
        import time
        start = time.monotonic()
        
        while True:
            with self._lock:
                if self._available:
                    handle = self._available.pop()
                    # Reset builder with new config
                    self._lib.aodds_builder_reset(handle, byref(c_config))
                    return StreamingBuilder(handle, self, self._lib, c_config)
            
            # Pool exhausted - try to create new if under limit
            with self._lock:
                current_count = len(self._available) + self._pool_size - len(self._available)
                if current_count < self._pool_size * 2:
                    # Create new builder
                    handle = self._lib.aodds_builder_create(
                        byref(c_config), self._decode_pool
                    )
                    if handle:
                        return StreamingBuilder(handle, self, self._lib, c_config)
            
            if time.monotonic() - start > timeout:
                log.warning(f"StreamingBuilderPool: timeout waiting for builder")
                return None
            
            time.sleep(0.001)
    
    def _return_builder(self, handle: c_void_p) -> None:
        """Return a builder handle to the pool."""
        with self._lock:
            self._available.append(handle)
    
    def close(self) -> None:
        """Destroy all builders in the pool."""
        with self._lock:
            for handle in self._available:
                self._lib.aodds_builder_destroy(handle)
            self._available.clear()
    
    @property
    def available_count(self) -> int:
        """Number of builders currently available."""
        with self._lock:
            return len(self._available)


# Global streaming builder pool (lazily initialized)
_default_builder_pool: Optional[StreamingBuilderPool] = None
_default_builder_pool_lock = threading.Lock()


def get_default_builder_pool() -> StreamingBuilderPool:
    """
    Get or create the default global streaming builder pool.
    
    Returns:
        The default StreamingBuilderPool instance
    """
    global _default_builder_pool
    if _default_builder_pool is None:
        with _default_builder_pool_lock:
            if _default_builder_pool is None:
                _default_builder_pool = StreamingBuilderPool()
    return _default_builder_pool


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
    
    # aodds_set_use_ispc
    lib.aodds_set_use_ispc.argtypes = [c_int32]
    lib.aodds_set_use_ispc.restype = None
    
    # aodds_get_use_ispc
    lib.aodds_get_use_ispc.argtypes = []
    lib.aodds_get_use_ispc.restype = c_int32
    
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


class BufferBuildResult(NamedTuple):
    """
    Result from zero-copy DDS tile building.
    
    Unlike BuildResult, this doesn't copy data - it returns a view into
    the provided buffer. The caller must ensure the buffer remains valid
    while using the data.
    
    Use `get_view()` for zero-copy access, or `to_bytes()` if you need
    to copy the data (e.g., to store it after releasing the buffer).
    """
    buffer: np.ndarray      # The buffer containing DDS data
    bytes_written: int      # Number of valid bytes in buffer
    success: bool
    chunks_found: int
    chunks_decoded: int
    chunks_failed: int
    mipmaps: int
    elapsed_ms: float
    error: str = ''
    
    def get_view(self) -> memoryview:
        """
        Get a zero-copy view of the DDS data.
        
        This is the fastest way to access the data but the view becomes
        invalid when the buffer is released back to the pool.
        """
        return memoryview(self.buffer[:self.bytes_written])
    
    def to_bytes(self) -> bytes:
        """
        Copy the DDS data to a new bytes object.
        
        Use this if you need to keep the data after releasing the buffer.
        This incurs a copy (~65ms for 10MB) but the data is then independent.
        """
        return bytes(self.buffer[:self.bytes_written])


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


def build_tile_to_buffer(
    buffer: np.ndarray,
    cache_dir: str,
    row: int,
    col: int,
    maptype: str,
    zoom: int,
    chunks_per_side: int = 16,
    format: str = "BC1",
    missing_color: Tuple[int, int, int] = (66, 77, 55),
    decode_pool: Optional[c_void_p] = None
) -> BufferBuildResult:
    """
    Build a DDS tile directly into a pre-allocated numpy buffer (ZERO-COPY).
    
    This is the highest-performance path for DDS building:
    - No allocation overhead (buffer already exists)
    - No copy overhead (data stays in provided buffer)
    - Use with DDSBufferPool for optimal performance
    
    Args:
        buffer: Pre-allocated numpy array (must be large enough for DDS output).
                Use DDSBufferPool.SIZE_4096x4096_BC1 for sizing.
        cache_dir: Directory containing cached JPEGs
        row: Tile row coordinate
        col: Tile column coordinate
        maptype: Map source identifier (e.g., "BI", "EOX")
        zoom: Zoom level for chunk fetching
        chunks_per_side: Number of chunks per side (default 16)
        format: Compression format - "BC1" (DXT1) or "BC3" (DXT5)
        missing_color: RGB tuple for missing chunks
        decode_pool: Optional decode buffer pool handle from AoDecode
    
    Returns:
        BufferBuildResult with success status and metadata.
        Use result.get_view() for zero-copy access to DDS data.
    
    Example:
        pool = DDSBufferPool()
        buffer, buffer_id = pool.acquire()
        try:
            result = build_tile_to_buffer(
                buffer, cache_dir, row, col, maptype, zoom
            )
            if result.success:
                # Zero-copy access to DDS data
                dds_view = result.get_view()
                # ... use dds_view ...
        finally:
            pool.release(buffer_id)
    """
    lib = _load_library()
    
    tile_size = chunks_per_side * 256
    fmt = FORMAT_BC1 if format.upper() in ("BC1", "DXT1") else FORMAT_BC3
    required_size = lib.aodds_calc_dds_size(tile_size, tile_size, 0, fmt)
    
    if len(buffer) < required_size:
        return BufferBuildResult(
            buffer=buffer,
            bytes_written=0,
            success=False,
            chunks_found=0,
            chunks_decoded=0,
            chunks_failed=0,
            mipmaps=0,
            elapsed_ms=0.0,
            error=f"Buffer too small: {len(buffer)} < {required_size} bytes needed"
        )
    
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
    request.dds_buffer = buffer.ctypes.data_as(POINTER(c_uint8))
    request.dds_buffer_size = len(buffer)
    
    pool_handle = decode_pool if decode_pool else None
    success = lib.aodds_build_tile(byref(request), pool_handle)
    
    return BufferBuildResult(
        buffer=buffer,
        bytes_written=request.dds_written if success else 0,
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


def set_use_ispc(use_ispc: bool) -> None:
    """
    Set whether to use ISPC compression or fallback (STB).
    
    This allows respecting user configuration for compressor preference.
    When use_ispc=False, the fallback compressor is used even if ISPC is available.
    
    Args:
        use_ispc: True to use ISPC (if available), False to force STB fallback
    """
    lib = _load_library()
    lib.aodds_set_use_ispc(1 if use_ispc else 0)


def get_use_ispc() -> bool:
    """
    Get whether ISPC compression is currently active.
    
    Returns:
        True if ISPC will be used, False if fallback will be used
    """
    lib = _load_library()
    return bool(lib.aodds_get_use_ispc())


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
    
    # Keep references to bytes objects to prevent GC during call
    # (No copy needed - cast directly to pointer, bytes are immutable
    # and CPython doesn't move objects in memory)
    jpeg_refs = []
    
    for i, data in enumerate(jpeg_datas):
        if data and len(data) > 0:
            # Keep reference alive and cast directly - NO COPY
            jpeg_refs.append(data)
            jpeg_ptrs[i] = cast(data, POINTER(c_uint8))
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


def build_from_jpegs_to_buffer(
    buffer: np.ndarray,
    jpeg_datas: list,
    format: str = "BC1",
    missing_color: Tuple[int, int, int] = (66, 77, 55),
    decode_pool: Optional[c_void_p] = None
) -> BufferBuildResult:
    """
    Build DDS from pre-read JPEG data into a pre-allocated buffer (ZERO-COPY).
    
    This is the OPTIMAL path combining:
    - HYBRID approach (Python reads files)
    - ZERO-COPY output (pre-allocated buffer)
    
    Performance improvement over build_from_jpegs():
    - No allocation overhead (~15ms saved)
    - No copy overhead (~65ms saved)
    - Total: ~80ms faster for 4096x4096 tiles
    
    Args:
        buffer: Pre-allocated numpy array (must be large enough for DDS output)
        jpeg_datas: List of JPEG bytes (None or b'' for missing chunks)
        format: Compression format - "BC1" (DXT1) or "BC3" (DXT5)
        missing_color: RGB tuple for missing chunks
        decode_pool: Optional decode buffer pool handle from AoDecode
    
    Returns:
        BufferBuildResult with success status and metadata.
        Use result.get_view() for zero-copy access to DDS data.
    
    Example:
        pool = DDSBufferPool()
        buffer, buffer_id = pool.acquire()
        try:
            # Python reads files (fast)
            jpeg_datas = [Path(p).read_bytes() for p in chunk_paths]
            
            # Build DDS with zero-copy output
            result = build_from_jpegs_to_buffer(buffer, jpeg_datas)
            if result.success:
                dds_view = result.get_view()
                # ... use dds_view ...
        finally:
            pool.release(buffer_id)
    """
    import math
    
    lib = _load_library()
    chunk_count = len(jpeg_datas)
    
    if chunk_count == 0:
        return BufferBuildResult(
            buffer=buffer, bytes_written=0, success=False,
            chunks_found=0, chunks_decoded=0, chunks_failed=0,
            mipmaps=0, elapsed_ms=0.0, error="No JPEG data provided"
        )
    
    # Verify perfect square
    chunks_per_side = int(math.sqrt(chunk_count))
    if chunks_per_side * chunks_per_side != chunk_count:
        return BufferBuildResult(
            buffer=buffer, bytes_written=0, success=False,
            chunks_found=0, chunks_decoded=0, chunks_failed=0,
            mipmaps=0, elapsed_ms=0.0,
            error=f"chunk_count must be a perfect square, got {chunk_count}"
        )
    
    # Calculate required buffer size
    tile_size = chunks_per_side * 256
    fmt = FORMAT_BC1 if format.upper() in ("BC1", "DXT1") else FORMAT_BC3
    required_size = lib.aodds_calc_dds_size(tile_size, tile_size, 0, fmt)
    
    if len(buffer) < required_size:
        return BufferBuildResult(
            buffer=buffer, bytes_written=0, success=False,
            chunks_found=0, chunks_decoded=0, chunks_failed=0,
            mipmaps=0, elapsed_ms=0.0,
            error=f"Buffer too small: {len(buffer)} < {required_size} bytes needed"
        )
    
    # Build arrays for C
    jpeg_ptrs = (POINTER(c_uint8) * chunk_count)()
    jpeg_sizes = (c_uint32 * chunk_count)()
    # Keep references to bytes objects to prevent GC during call
    # (No copy needed - cast directly to pointer, bytes are immutable
    # and CPython doesn't move objects in memory)
    jpeg_refs = []
    
    for i, data in enumerate(jpeg_datas):
        if data and len(data) > 0:
            # Keep reference alive and cast directly - NO COPY
            jpeg_refs.append(data)
            jpeg_ptrs[i] = cast(data, POINTER(c_uint8))
            jpeg_sizes[i] = len(data)
        else:
            jpeg_ptrs[i] = None
            jpeg_sizes[i] = 0
    
    bytes_written = c_uint32()
    pool_handle = decode_pool if decode_pool else None
    
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
    
    # Call native function with numpy buffer
    success = lib.aodds_build_from_jpegs(
        jpeg_ptrs,
        jpeg_sizes,
        chunk_count,
        fmt,
        missing_color[0],
        missing_color[1],
        missing_color[2],
        buffer.ctypes.data_as(POINTER(c_uint8)),
        len(buffer),
        byref(bytes_written),
        pool_handle
    )
    
    return BufferBuildResult(
        buffer=buffer,
        bytes_written=bytes_written.value if success else 0,
        success=bool(success),
        chunks_found=chunk_count,  # All chunks were provided
        chunks_decoded=chunk_count if success else 0,
        chunks_failed=0,
        mipmaps=int(math.log2(tile_size)) + 1 if success else 0,
        elapsed_ms=0.0,  # Not tracked in this API
        error="" if success else "Failed to build DDS from JPEGs"
    )


# ============================================================================
# Direct-to-File Pipeline: Zero-copy DDS building to disk
# ============================================================================

class FileBuildResult(NamedTuple):
    """
    Result from direct-to-file DDS building.
    
    This is the OPTIMAL path for predictive DDS - no Python memory involvement
    after the initial JPEG data is passed to C.
    """
    success: bool
    bytes_written: int
    output_path: str
    error: str = ''


def build_from_jpegs_to_file(
    jpeg_datas: list,
    output_path: str,
    format: str = "BC1",
    missing_color: Tuple[int, int, int] = (66, 77, 55),
    decode_pool: Optional[c_void_p] = None
) -> FileBuildResult:
    """
    Build DDS from pre-read JPEG data and write directly to disk (ZERO-COPY).
    
    This is the OPTIMAL path for predictive DDS caching:
    - No allocation overhead (writes incrementally to disk)
    - No copy overhead (data goes straight to file)
    - No Python memory involvement after JPEG data is passed
    - Perfect integration with EphemeralDDSCache
    
    Performance improvement over build_from_jpegs():
    - ~65ms copy overhead eliminated (no buffer → bytes conversion)
    - No GIL contention during file write (done in C)
    - Streaming write (lower peak memory)
    
    Atomicity:
    - Uses temp file + rename pattern
    - No corrupt files on crash
    
    Args:
        jpeg_datas: List of JPEG bytes (None or b'' for missing chunks)
                    Length must be a perfect square (16, 64, 256, etc.)
        output_path: Path to output DDS file (will be created/overwritten)
        format: Compression format - "BC1" (DXT1) or "BC3" (DXT5)
        missing_color: RGB tuple for missing chunks
        decode_pool: Optional decode buffer pool handle from AoDecode
    
    Returns:
        FileBuildResult with success status and metadata.
    
    Example:
        # Python reads files (fast)
        jpeg_datas = [Path(p).read_bytes() for p in chunk_paths]
        
        # Build DDS directly to disk (no Python memory for output)
        result = build_from_jpegs_to_file(
            jpeg_datas,
            "/path/to/cache/tile_12345.dds"
        )
        if result.success:
            print(f"Wrote {result.bytes_written} bytes to {result.output_path}")
    """
    import math
    
    lib = _load_library()
    chunk_count = len(jpeg_datas)
    
    if chunk_count == 0:
        return FileBuildResult(
            success=False, bytes_written=0, output_path=output_path,
            error="No JPEG data provided"
        )
    
    # Verify perfect square
    chunks_per_side = int(math.sqrt(chunk_count))
    if chunks_per_side * chunks_per_side != chunk_count:
        return FileBuildResult(
            success=False, bytes_written=0, output_path=output_path,
            error=f"chunk_count must be a perfect square, got {chunk_count}"
        )
    
    # Build arrays for C
    jpeg_ptrs = (POINTER(c_uint8) * chunk_count)()
    jpeg_sizes = (c_uint32 * chunk_count)()
    # Keep references to bytes objects to prevent GC during call
    # (No copy needed - cast directly to pointer, bytes are immutable
    # and CPython doesn't move objects in memory)
    jpeg_refs = []
    
    for i, data in enumerate(jpeg_datas):
        if data and len(data) > 0:
            # Keep reference alive and cast directly - NO COPY
            jpeg_refs.append(data)
            jpeg_ptrs[i] = cast(data, POINTER(c_uint8))
            jpeg_sizes[i] = len(data)
        else:
            jpeg_ptrs[i] = None
            jpeg_sizes[i] = 0
    
    bytes_written = c_uint32()
    pool_handle = decode_pool if decode_pool else None
    fmt = FORMAT_BC1 if format.upper() in ("BC1", "DXT1") else FORMAT_BC3
    
    # Setup function signature if not already done
    if not hasattr(lib, '_file_output_setup_done'):
        lib.aodds_build_from_jpegs_to_file.argtypes = [
            POINTER(POINTER(c_uint8)),  # jpeg_data
            POINTER(c_uint32),          # jpeg_sizes
            c_int32,                    # chunk_count
            c_int32,                    # format
            c_uint8, c_uint8, c_uint8,  # missing color
            c_char_p,                   # output_path
            POINTER(c_uint32),          # bytes_written
            c_void_p                    # pool
        ]
        lib.aodds_build_from_jpegs_to_file.restype = c_int32
        lib._file_output_setup_done = True
    
    # Encode path for cross-platform compatibility
    # On Windows, use UTF-8 which works with most file systems
    path_bytes = output_path.encode('utf-8')
    
    # Call native function
    success = lib.aodds_build_from_jpegs_to_file(
        jpeg_ptrs,
        jpeg_sizes,
        chunk_count,
        fmt,
        missing_color[0],
        missing_color[1],
        missing_color[2],
        path_bytes,
        byref(bytes_written),
        pool_handle
    )
    
    return FileBuildResult(
        success=bool(success),
        bytes_written=bytes_written.value if success else 0,
        output_path=output_path,
        error="" if success else "Failed to build DDS to file"
    )


def build_tile_to_file(
    cache_dir: str,
    row: int,
    col: int,
    maptype: str,
    zoom: int,
    output_path: str,
    chunks_per_side: int = 16,
    format: str = "BC1",
    missing_color: Tuple[int, int, int] = (66, 77, 55),
    decode_pool: Optional[c_void_p] = None
) -> FileBuildResult:
    """
    Build DDS from cache files and write directly to disk (NATIVE ZERO-COPY).
    
    This is the OPTIMAL path for NATIVE MODE predictive DDS caching:
    - C reads cache files directly
    - C decodes + compresses in parallel
    - C writes directly to disk (no Python involvement)
    - Perfect integration with EphemeralDDSCache
    
    This provides the same optimization as build_from_jpegs_to_file() but for
    native mode where C handles file I/O.
    
    Performance improvement over build_tile_native():
    - ~65ms copy overhead eliminated (no buffer → bytes conversion)
    - Streaming write (lower peak memory)
    
    Atomicity:
    - Uses temp file + rename pattern
    - No corrupt files on crash
    
    Args:
        cache_dir: Directory containing cached JPEGs
        row: Tile row coordinate
        col: Tile column coordinate
        maptype: Map source identifier (e.g., "BI", "EOX")
        zoom: Zoom level for chunk fetching
        output_path: Path to output DDS file
        chunks_per_side: Number of chunks per side (default 16)
        format: Compression format - "BC1" (DXT1) or "BC3" (DXT5)
        missing_color: RGB tuple for missing chunks
        decode_pool: Optional decode buffer pool handle from AoDecode
    
    Returns:
        FileBuildResult with success status and metadata.
    
    Example:
        result = build_tile_to_file(
            cache_dir="/path/to/cache",
            row=1234, col=5678,
            maptype="BI", zoom=16,
            output_path="/path/to/output.dds"
        )
        if result.success:
            print(f"Wrote {result.bytes_written} bytes")
    """
    lib = _load_library()
    
    bytes_written = c_uint32()
    pool_handle = decode_pool if decode_pool else None
    fmt = FORMAT_BC1 if format.upper() in ("BC1", "DXT1") else FORMAT_BC3
    
    # Setup function signature if not already done
    if not hasattr(lib, '_tile_to_file_setup_done'):
        lib.aodds_build_tile_to_file.argtypes = [
            c_char_p,                   # cache_dir
            c_int32,                    # tile_row
            c_int32,                    # tile_col
            c_char_p,                   # maptype
            c_int32,                    # zoom
            c_int32,                    # chunks_per_side
            c_int32,                    # format
            c_uint8, c_uint8, c_uint8,  # missing color
            c_char_p,                   # output_path
            POINTER(c_uint32),          # bytes_written
            c_void_p                    # pool
        ]
        lib.aodds_build_tile_to_file.restype = c_int32
        lib._tile_to_file_setup_done = True
    
    # Encode paths for cross-platform compatibility
    cache_dir_bytes = cache_dir.encode('utf-8')
    maptype_bytes = maptype.encode('utf-8')
    output_path_bytes = output_path.encode('utf-8')
    
    # Call native function
    success = lib.aodds_build_tile_to_file(
        cache_dir_bytes,
        row,
        col,
        maptype_bytes,
        zoom,
        chunks_per_side,
        fmt,
        missing_color[0],
        missing_color[1],
        missing_color[2],
        output_path_bytes,
        byref(bytes_written),
        pool_handle
    )
    
    return FileBuildResult(
        success=bool(success),
        bytes_written=bytes_written.value if success else 0,
        output_path=output_path,
        error="" if success else "Failed to build DDS to file"
    )


# ============================================================================
# Module initialization
# ============================================================================

# Try to load library on import to fail fast
try:
    _load_library()
except Exception:
    pass  # Will raise on first use if not available

