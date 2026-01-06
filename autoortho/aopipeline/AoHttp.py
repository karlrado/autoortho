"""
AoHttp.py - Python wrapper for native HTTP client pool

Provides high-performance parallel HTTP downloads that bypass Python's GIL
by delegating to libcurl multi-interface.

Usage:
    from autoortho.aopipeline import AoHttp
    
    # Initialize with connection pool
    AoHttp.init(max_connections=32)
    
    # Batch fetch URLs synchronously
    urls = ["http://example.com/a.jpg", "http://example.com/b.jpg"]
    responses = AoHttp.get_batch_sync(urls)
    
    for resp in responses:
        if resp.success:
            process(resp.data)
"""

from ctypes import (
    CDLL, POINTER, Structure, CFUNCTYPE, c_void_p,
    c_char, c_char_p, c_int32, c_uint8, c_uint32,
    byref, cast
)
import logging
import os
import sys
from typing import List, Optional, NamedTuple, Callable

log = logging.getLogger(__name__)

# ============================================================================
# Library Loading
# ============================================================================

_aohttp = None
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
    global _aohttp, _load_error
    
    if _aohttp is not None:
        return _aohttp
    
    if _load_error is not None:
        raise _load_error
    
    try:
        lib_path = _get_lib_path()
        log.debug(f"Loading aohttp native library from: {lib_path}")
        _aohttp = CDLL(lib_path)
        
        # Configure function signatures
        _setup_signatures(_aohttp)
        
        return _aohttp
        
    except Exception as e:
        _load_error = ImportError(f"Failed to load aohttp native library: {e}")
        log.warning(f"Native HTTP library not available: {e}")
        raise _load_error


def _setup_signatures(lib):
    """Configure ctypes function signatures."""
    
    # aohttp_init
    lib.aohttp_init.argtypes = [c_int32]
    lib.aohttp_init.restype = c_int32
    
    # aohttp_shutdown
    lib.aohttp_shutdown.argtypes = []
    lib.aohttp_shutdown.restype = None
    
    # aohttp_is_available
    lib.aohttp_is_available.argtypes = []
    lib.aohttp_is_available.restype = c_int32
    
    # aohttp_get_sync
    lib.aohttp_get_sync.argtypes = [c_char_p, POINTER(HttpResponse), c_int32]
    lib.aohttp_get_sync.restype = c_int32
    
    # aohttp_get_batch_sync
    lib.aohttp_get_batch_sync.argtypes = [
        POINTER(c_char_p), c_int32, POINTER(HttpResponse), c_int32
    ]
    lib.aohttp_get_batch_sync.restype = c_int32
    
    # aohttp_response_free
    lib.aohttp_response_free.argtypes = [POINTER(HttpResponse)]
    lib.aohttp_response_free.restype = None
    
    # aohttp_response_batch_free
    lib.aohttp_response_batch_free.argtypes = [POINTER(HttpResponse), c_int32]
    lib.aohttp_response_batch_free.restype = None
    
    # aohttp_set_user_agent
    lib.aohttp_set_user_agent.argtypes = [c_char_p]
    lib.aohttp_set_user_agent.restype = None
    
    # aohttp_set_ssl_verify
    lib.aohttp_set_ssl_verify.argtypes = [c_int32]
    lib.aohttp_set_ssl_verify.restype = None
    
    # aohttp_pending_count
    lib.aohttp_pending_count.argtypes = []
    lib.aohttp_pending_count.restype = c_int32
    
    # aohttp_version
    lib.aohttp_version.argtypes = []
    lib.aohttp_version.restype = c_char_p


# ============================================================================
# Data Structures
# ============================================================================

class HttpResponse(Structure):
    """
    HTTP response structure.
    Maps to aohttp_response_t in C.
    """
    _fields_ = [
        ('status_code', c_int32),
        ('data', POINTER(c_uint8)),
        ('length', c_uint32),
        ('error', c_char * 128),
    ]
    
    @property
    def success(self) -> bool:
        """Check if request was successful (2xx status)."""
        return 200 <= self.status_code < 300
    
    def get_bytes(self) -> bytes:
        """Get response body as bytes (copies data)."""
        if not self.data or self.length == 0:
            return b''
        return bytes(self.data[:self.length])
    
    def get_error(self) -> str:
        """Get error message if request failed."""
        if self.success:
            return ''
        return self.error.decode('utf-8', errors='replace').rstrip('\x00')


class FetchResult(NamedTuple):
    """Python-friendly HTTP fetch result."""
    data: bytes
    success: bool
    status_code: int
    error: str = ''


# ============================================================================
# Public API
# ============================================================================

_initialized = False


def init(max_connections: int = 32) -> bool:
    """
    Initialize the HTTP client pool.
    
    Args:
        max_connections: Maximum concurrent connections
    
    Returns:
        True if initialized successfully
    """
    global _initialized
    
    if _initialized:
        return True
    
    try:
        lib = _load_library()
        result = lib.aohttp_init(max_connections)
        _initialized = bool(result)
        
        if _initialized:
            version = lib.aohttp_version()
            log.info(f"Native HTTP client initialized: {version.decode()}")
        else:
            log.warning("Native HTTP client init failed (libcurl not available?)")
        
        return _initialized
        
    except ImportError:
        return False


def shutdown():
    """Shutdown the HTTP client pool."""
    global _initialized
    
    if not _initialized:
        return
    
    try:
        lib = _load_library()
        lib.aohttp_shutdown()
        _initialized = False
        log.info("Native HTTP client shut down")
    except ImportError:
        pass


def is_available() -> bool:
    """
    Check if native HTTP client is available.
    
    Returns True if the library can be loaded, even if not yet initialized.
    Use init() to actually initialize the HTTP pool before making requests.
    """
    try:
        _load_library()
        return True
    except (ImportError, FileNotFoundError, OSError):
        return False


def get_sync(url: str, timeout_ms: int = 30000) -> FetchResult:
    """
    Fetch a single URL synchronously.
    
    Args:
        url: URL to fetch
        timeout_ms: Request timeout in milliseconds
    
    Returns:
        FetchResult with data, success, status_code, error
    """
    if not _initialized:
        init()
    
    lib = _load_library()
    
    response = HttpResponse()
    
    if isinstance(url, str):
        url = url.encode('utf-8')
    
    lib.aohttp_get_sync(url, byref(response), timeout_ms)
    
    result = FetchResult(
        data=response.get_bytes(),
        success=response.success,
        status_code=response.status_code,
        error=response.get_error()
    )
    
    lib.aohttp_response_free(byref(response))
    
    return result


def get_batch_sync(
    urls: List[str],
    timeout_ms: int = 30000
) -> List[FetchResult]:
    """
    Fetch multiple URLs in parallel synchronously.
    
    This downloads all URLs concurrently using libcurl multi-interface,
    then returns when all complete.
    
    Args:
        urls: List of URLs to fetch
        timeout_ms: Per-request timeout in milliseconds
    
    Returns:
        List of FetchResult in same order as input URLs
    """
    if not urls:
        return []
    
    if not _initialized:
        init()
    
    lib = _load_library()
    count = len(urls)
    
    # Create URL array
    url_array = (c_char_p * count)()
    for i, url in enumerate(urls):
        if isinstance(url, str):
            url_array[i] = url.encode('utf-8')
        else:
            url_array[i] = url
    
    # Create response array
    responses = (HttpResponse * count)()
    
    # Fetch all
    lib.aohttp_get_batch_sync(url_array, count, responses, timeout_ms)
    
    # Convert to Python results
    results = []
    for i in range(count):
        results.append(FetchResult(
            data=responses[i].get_bytes(),
            success=responses[i].success,
            status_code=responses[i].status_code,
            error=responses[i].get_error()
        ))
    
    # Free responses
    lib.aohttp_response_batch_free(responses, count)
    
    return results


def set_user_agent(user_agent: str):
    """Set custom User-Agent header for all requests."""
    try:
        lib = _load_library()
        if isinstance(user_agent, str):
            user_agent = user_agent.encode('utf-8')
        lib.aohttp_set_user_agent(user_agent)
    except ImportError:
        pass


def set_ssl_verify(verify: bool = True):
    """Enable or disable SSL certificate verification."""
    try:
        lib = _load_library()
        lib.aohttp_set_ssl_verify(1 if verify else 0)
    except ImportError:
        pass


def pending_count() -> int:
    """Get number of pending/active requests."""
    try:
        lib = _load_library()
        return lib.aohttp_pending_count()
    except ImportError:
        return 0


def get_version() -> str:
    """Get version information for native HTTP library."""
    try:
        lib = _load_library()
        return lib.aohttp_version().decode('utf-8')
    except ImportError:
        return "not available"


# ============================================================================
# Module initialization
# ============================================================================

# Don't auto-init HTTP - it requires explicit init() call

