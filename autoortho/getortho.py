#!/usr/bin/env python3
import logging
import atexit
import os
import time
import threading
import concurrent.futures
import uuid
import math
import tracemalloc
from typing import Optional

from io import BytesIO
from urllib.request import urlopen, Request
from queue import Queue, PriorityQueue, Empty
from functools import wraps, lru_cache
from pathlib import Path
from collections import OrderedDict

import pydds

import requests
import psutil
from aoimage import AoImage

from aoconfig import CFG
from aostats import STATS, StatTracker, StatsBatcher, get_stat, inc_many, inc_stat, set_stat, update_process_memory_stat, clear_process_memory_stat
from utils.constants import (
    system_type, 
    CURRENT_CPU_COUNT,
    EARTH_RADIUS_M,
    PRIORITY_DISTANCE_WEIGHT,
    PRIORITY_DIRECTION_WEIGHT,
    PRIORITY_MIPMAP_WEIGHT,
    LOOKAHEAD_TIME_SEC,
)
from utils.apple_token_service import apple_token_service
from utils.dynamic_zoom import DynamicZoomManager
from utils.altitude_predictor import predict_altitude_at_closest_approach
from utils.simbrief_flight import simbrief_flight_manager

from datareftrack import dt as datareftracker

MEMTRACE = False

log = logging.getLogger(__name__)


# ============================================================================
# HTTP/2 SUPPORT (OPTIONAL)
# ============================================================================
# Try to use httpx for HTTP/2 multiplexing. Falls back to requests if not available.
# HTTP/2 allows multiple requests over a single connection, reducing latency.
# ============================================================================
_HTTPX_AVAILABLE = False
_httpx = None
try:
    import httpx as _httpx
    _HTTPX_AVAILABLE = True
    # Silence httpx's verbose request logging (it logs every request at INFO level)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    log.info("httpx available - HTTP/2 support enabled")
except ImportError:
    log.debug("httpx not installed - using requests (HTTP/1.1). Install httpx for HTTP/2 support.")


class HttpxSessionWrapper:
    """
    Wrapper around httpx.Client that provides requests-compatible API.
    This allows seamless switching between requests and httpx.
    """
    
    def __init__(self, http2=True, pool_size=10, timeout=25):
        """
        Initialize httpx client with HTTP/2 support.
        
        Args:
            http2: Enable HTTP/2 (default True)
            pool_size: Connection pool size
            timeout: Request timeout in seconds
        """
        # httpx uses limits for connection pooling
        limits = _httpx.Limits(
            max_keepalive_connections=pool_size,
            max_connections=pool_size * 2,
            keepalive_expiry=30.0
        )
        
        # Create httpx client with HTTP/2 enabled
        self._client = _httpx.Client(
            http2=http2,
            limits=limits,
            timeout=_httpx.Timeout(timeout, connect=5.0),
            follow_redirects=True
        )
        self._http2 = http2
        
    def get(self, url, headers=None, timeout=None):
        """
        HTTP GET request - returns a requests-compatible response object.
        
        Args:
            url: URL to fetch
            headers: Optional request headers
            timeout: Tuple of (connect_timeout, read_timeout) or single value
        """
        # Convert requests-style timeout tuple to httpx timeout
        if isinstance(timeout, tuple):
            connect_timeout, read_timeout = timeout
            httpx_timeout = _httpx.Timeout(read_timeout, connect=connect_timeout)
        elif timeout is not None:
            httpx_timeout = _httpx.Timeout(timeout)
        else:
            httpx_timeout = None
        
        try:
            response = self._client.get(url, headers=headers, timeout=httpx_timeout)
            # Wrap in a compatible response object
            return HttpxResponseWrapper(response)
        except _httpx.TimeoutException as e:
            raise requests.exceptions.Timeout(str(e))
        except _httpx.RequestError as e:
            raise requests.exceptions.RequestException(str(e))
    
    def close(self):
        """Close the underlying httpx client."""
        try:
            self._client.close()
        except Exception:
            pass


class HttpxResponseWrapper:
    """
    Wrapper that makes httpx.Response compatible with requests.Response API.
    """
    
    def __init__(self, httpx_response):
        self._response = httpx_response
        
    @property
    def status_code(self):
        return self._response.status_code
    
    @property
    def content(self):
        return self._response.content
    
    @property
    def text(self):
        return self._response.text
    
    @property
    def headers(self):
        return dict(self._response.headers)
    
    def close(self):
        try:
            self._response.close()
        except Exception:
            pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


def create_http_session(pool_size=10):
    """
    Factory function to create the best available HTTP session.
    
    Returns httpx client with HTTP/2 if available, otherwise requests session.
    Both have compatible APIs via the wrapper classes.
    """
    use_http2 = getattr(CFG.autoortho, 'use_http2', True)
    
    if _HTTPX_AVAILABLE and use_http2:
        try:
            session = HttpxSessionWrapper(http2=True, pool_size=pool_size)
            log.debug(f"Created httpx session with HTTP/2 (pool_size={pool_size})")
            return session
        except Exception as e:
            log.warning(f"Failed to create httpx session: {e}, falling back to requests")
    
    # Fall back to requests
    session = requests.Session()
    # IMPORTANT: pool_block=False prevents indefinite blocking when connection pool
    # is exhausted. With True, requests would block forever waiting for a connection
    # if all connections are busy (e.g., slow servers). With False, a ConnectionError
    # is raised immediately, allowing the retry logic to handle it gracefully.
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=pool_size,
        pool_maxsize=pool_size,
        max_retries=0,
        pool_block=False,  # Don't block - fail fast and retry
    )
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    log.debug(f"Created requests session with HTTP/1.1 (pool_size={pool_size}, pool_block=False)")
    return session


# JPEG decode concurrency: auto-tuned for optimal performance
# Decode is memory-bound, not CPU-bound, so we can safely exceed CPU count
# Each decode uses ~256KB RAM, so even 64 concurrent = only ~16MB
_MAX_DECODE = min(CURRENT_CPU_COUNT * 4, 64)
_decode_sem = threading.Semaphore(_MAX_DECODE)


# Track average fetch times
tile_stats = StatTracker(20, 12)
mm_stats = StatTracker(0, 5)
partial_stats = StatTracker()

# Track FULL tile creation duration (download + compose + compress)
# This is the key metric for tuning tile_time_budget
tile_creation_stats = StatTracker(0, 5, default=0, maxlen=50)

stats_batcher = None


def _ensure_stats_batcher():
    global stats_batcher
    if stats_batcher is None:
        try:
            # Create when a remote store is bound (either via env or parent bind)
            if getattr(STATS, "_remote", None) is not None or os.getenv("AO_STATS_ADDR"):
                stats_batcher = StatsBatcher(flush_interval=0.05, max_items=200)
                atexit.register(stats_batcher.stop)
        except Exception:
            stats_batcher = None


def bump(key, n=1):
    _ensure_stats_batcher()
    if stats_batcher:
        stats_batcher.add(key, n)
    else:
        inc_stat(key, n)


def bump_many(d: dict):
    _ensure_stats_batcher()
    if stats_batcher:
        stats_batcher.add_many(d)
    else:
        inc_many(d)


def get_tile_creation_stats():
    """
    Get tile creation statistics for monitoring and tuning.
    
    Returns a dictionary with:
    - count: Number of tiles created
    - avg_time_s: Average creation time in seconds
    - avg_time_by_mipmap: Dict of mipmap level -> average time in seconds
    - averages: Rolling averages from tile_creation_stats (last 50 samples per mipmap)
    
    This is useful for tuning tile_time_budget - the average creation time
    gives a good baseline for how long tiles actually take to create.
    """
    result = {
        'count': 0,
        'avg_time_s': 0.0,
        'avg_time_by_mipmap': {},
        'averages': {},
    }
    
    try:
        # Get counter-based stats (aggregate across all time)
        from aostats import get_stat
        count = get_stat('tile_create_count')
        total_time_ms = get_stat('tile_create_time_total_ms')
        
        result['count'] = count
        if count > 0:
            result['avg_time_s'] = round(total_time_ms / count / 1000.0, 3)
        
        # Get per-mipmap averages from counters
        for mm_level in range(5):
            mm_count = get_stat(f'mm_count:{mm_level}')
            mm_time = get_stat(f'tile_create_time_ms:{mm_level}')
            if mm_count > 0:
                result['avg_time_by_mipmap'][mm_level] = round(mm_time / mm_count / 1000.0, 3)
        
        # Get rolling averages from StatTracker (recent samples)
        result['averages'] = dict(tile_creation_stats.averages)
        
    except Exception as e:
        log.debug(f"get_tile_creation_stats error: {e}")
    
    return result


def log_tile_creation_summary():
    """Log a summary of tile creation statistics."""
    stats = get_tile_creation_stats()
    if stats['count'] > 0:
        log.info(f"TILE CREATION STATS: {stats['count']} tiles created, "
                f"avg time: {stats['avg_time_s']:.2f}s")
        if stats['avg_time_by_mipmap']:
            mm_str = ", ".join(f"MM{k}: {v:.2f}s" for k, v in sorted(stats['avg_time_by_mipmap'].items()))
            log.info(f"  Per-mipmap averages: {mm_str}")
        if stats['averages']:
            recent_str = ", ".join(f"MM{k}: {v:.2f}s" for k, v in sorted(stats['averages'].items()) if v > 0)
            if recent_str:
                log.info(f"  Recent averages (last 50): {recent_str}")


seasons_enabled = CFG.seasons.enabled

if seasons_enabled:
    from aoseasons import AoSeasonCache
    ao_seasons = AoSeasonCache(CFG.paths.cache_dir)

    # Per-DSF tile locks to serialize .si generation across threads
    _season_lock_map = {}
    _season_lock_map_lock = threading.Lock()

    def _season_tile_key_from_rc(row: int, col: int, zoom: int) -> str:
        # Compute DSF base name used by AoDsfSeason (e.g., +37-122)
        lon = col / pow(2, zoom) * 360 - 180
        n = math.pi - 2 * math.pi * row / pow(2, zoom)
        lat = 180 / math.pi * math.atan(0.5 * (math.exp(n) - math.exp(-n)))
        lat_i = math.floor(lat)
        lon_i = math.floor(lon)
        return f"{lat_i:+03d}{lon_i:+04d}"

    def _get_season_lock(key: str) -> threading.Lock:
        with _season_lock_map_lock:
            lock = _season_lock_map.get(key)
            if lock is None:
                lock = threading.Lock()
                _season_lock_map[key] = lock
            return lock

    def season_saturation_locked(row: int, col: int, zoom: int, day: Optional[int] = None) -> float:
        key = _season_tile_key_from_rc(row, col, zoom)
        lock = _get_season_lock(key)
        with lock:
            return ao_seasons.saturation(row, col, zoom, day)


def _is_jpeg(dataheader):
    # FFD8FF identifies image as a JPEG
    if dataheader[:3] == b'\xFF\xD8\xFF':
        return True
    else:
        return False


def _gtile_to_quadkey(til_x, til_y, zoomlevel):
    """
    Translates Google coding of tiles to Bing Quadkey coding. 
    """
    quadkey=""
    temp_x=til_x
    temp_y=til_y    
    for step in range(1,zoomlevel+1):
        size=2**(zoomlevel-step)
        a=temp_x//size
        b=temp_y//size
        temp_x=temp_x-a*size
        temp_y=temp_y-b*size
        quadkey=quadkey+str(a+2*b)
    return quadkey


def _chunk_to_latlon(row: int, col: int, zoom: int) -> tuple:
    """
    Convert tile row/col/zoom to center lat/lon coordinates.
    Returns (lat, lon) in degrees.
    
    Uses official OSM/Web Mercator formula:
    n = 2 ^ zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = arctan(sinh(π * (1 - 2 * ytile / n)))
    lat_deg = lat_rad * 180.0 / π
    
    Note: Adds 0.5 to row/col to get tile CENTER instead of top-left corner.
    """
    n = 2.0 ** zoom
    # Use tile center by adding 0.5 to both coordinates
    lon = (col + 0.5) / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * (row + 0.5) / n)))
    lat = math.degrees(lat_rad)
    return (lat, lon)


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two points in meters using Haversine formula.
    """
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    # Haversine formula
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_M * c


def _calculate_spatial_priority(chunk_row: int, chunk_col: int, chunk_zoom: int, base_mipmap_priority: int) -> float:
    """
    Calculate priority based on:
    1. Distance from player
    2. Direction relative to movement vector (predictive)
    3. Base mipmap priority (detail level)
    
    Lower priority number = more urgent (fetched first).
    Returns priority as float.
    """
    # If no valid flight data, fall back to mipmap-only priority
    if not datareftracker.data_valid or not datareftracker.connected:
        return float(base_mipmap_priority)
    
    try:
        # Get player position and velocity
        player_lat = datareftracker.lat
        player_lon = datareftracker.lon
        player_hdg = datareftracker.hdg  # degrees, 0=N, 90=E, 180=S, 270=W
        player_spd = datareftracker.spd  # m/s

        chunk_lat, chunk_lon = _chunk_to_latlon(chunk_row, chunk_col, chunk_zoom)
        
        distance_m = _haversine_distance(player_lat, player_lon, chunk_lat, chunk_lon)
        
        distance_priority = min(100, distance_m / 100)  # 100m per priority point, max 100
        
        if player_spd > 5:  # Only use predictive if moving
            lat1_rad = math.radians(player_lat)
            lat2_rad = math.radians(chunk_lat)
            dlon_rad = math.radians(chunk_lon - player_lon)
            
            x = math.sin(dlon_rad) * math.cos(lat2_rad)
            y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad)
            bearing = math.degrees(math.atan2(x, y))
            bearing = (bearing + 360) % 360
            
            angle_diff = abs(player_hdg - bearing)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            

            direction_priority = (angle_diff / 180) * 100 - 50
            
            predicted_distance = player_spd * LOOKAHEAD_TIME_SEC
            if distance_m < predicted_distance and angle_diff < 45:
                direction_priority -= 30
        else:
            direction_priority = 0
        

        total_priority = (
            base_mipmap_priority * PRIORITY_MIPMAP_WEIGHT +
            distance_priority * PRIORITY_DISTANCE_WEIGHT +
            direction_priority * PRIORITY_DIRECTION_WEIGHT
        )
        
        return max(0, total_priority)
        
    except Exception as e:
        # On any error, fall back to mipmap-only priority
        log.debug(f"Spatial priority calculation failed: {e}")
        return float(base_mipmap_priority)


def _safe_paste(dest_img, chunk_img, start_x, start_y, defer_cleanup=False):
    """
    Paste chunk_img into dest_img at (start_x, start_y) with coordinate validation.
    
    Args:
        dest_img: Destination AoImage
        chunk_img: Source chunk AoImage to paste
        start_x, start_y: Position to paste at
        defer_cleanup: If True, don't free chunk_img (caller will batch-free later)
    
    Returns True if paste succeeded, False if skipped due to invalid coordinates.
    """
    if start_x < 0 or start_y < 0:
        log.warning(f"GET_IMG: Skipping chunk with invalid coordinates ({start_x},{start_y})")
        if not defer_cleanup:
            try:
                chunk_img.close()
            except Exception:
                pass
        return False
    
    if not dest_img.paste(chunk_img, (start_x, start_y)):
        log.warning(f"GET_IMG: paste() failed for chunk at ({start_x},{start_y})")
        if not defer_cleanup:
            try:
                chunk_img.close()
            except Exception:
                pass
        return False
    
    # Free native buffer immediately unless deferred
    if not defer_cleanup:
        try:
            chunk_img.close()
        except Exception:
            pass
    
    return True


def _batch_close_images(images):
    """Batch-close multiple AoImage objects to free native memory."""
    for img in images:
        if img is not None:
            try:
                img.close()
            except Exception:
                pass


def locked(fn):
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        #result = fn(self, *args, **kwargs)
        with self._lock:
            result = fn(self, *args, **kwargs)
        return result
    return wrapped


class TimeBudget:
    """
    Track elapsed wall-clock time for a single X-Plane tile request.
    
    This class solves the per-chunk vs per-request timeout problem:
    - Previously, each chunk had its own maxwait timeout, leading to
      serialized chunks taking N * maxwait total time
    - Now, all chunks share a single time budget, ensuring the total
      wall-clock time stays close to the configured limit
    
    Usage:
        budget = TimeBudget(max_seconds=2.0)
        while not budget.exhausted:
            chunk_ready = budget.wait_with_budget(chunk.ready)
            if not chunk_ready:
                break  # Budget exhausted or event not set
    
    Thread-safety: Uses time.monotonic() which is thread-safe and immune
    to system clock adjustments.
    """
    
    # Minimum wait granularity - how often we check if budget is exhausted
    # during a wait. Smaller = more responsive but slightly more CPU.
    # 50ms is a good balance: responsive enough for UI, not too chatty.
    WAIT_GRANULARITY_SEC = 0.05
    
    def __init__(self, max_seconds: float):
        """
        Initialize a time budget.
        
        Args:
            max_seconds: Maximum wall-clock time allowed for this request.
                        This should be the value the user expects X-Plane
                        to actually wait, not a per-chunk timeout.
        """
        self.max_seconds = max_seconds
        self.start_time = time.monotonic()
        self._exhausted = False
        self._chunks_processed = 0
        self._chunks_skipped = 0
    
    @property
    def remaining(self) -> float:
        """Return remaining time in seconds (never negative)."""
        return max(0.0, self.max_seconds - self.elapsed)
    
    @property 
    def elapsed(self) -> float:
        """Return elapsed time since budget creation in seconds."""
        return time.monotonic() - self.start_time
    
    @property
    def exhausted(self) -> bool:
        """
        Check if the time budget is exhausted.
        
        Once exhausted, always returns True (sticky flag for efficiency).
        """
        if self._exhausted:
            return True
        if self.elapsed >= self.max_seconds:
            self._exhausted = True
            return True
        return False
    
    def wait_with_budget(self, event: threading.Event, max_single_wait: float = None) -> bool:
        """
        Wait on an event while respecting both the time budget AND an optional per-chunk maxwait.
        
        This combines two timeout mechanisms:
        1. Time budget: Total wall-clock time for the entire tile
        2. Max single wait: Per-chunk timeout (like the old maxwait parameter)
        
        Args:
            event: A threading.Event to wait on (e.g., chunk.ready)
            max_single_wait: Optional per-chunk timeout in seconds. If provided,
                           the wait will not exceed this time even if budget remains.
                           This corresponds to the old "maxwait" config setting.
        
        Returns:
            True if the event was set (success)
            False if budget exhausted or max_single_wait exceeded before event was set
        
        Behavior:
            - If event is already set, returns immediately with True
            - If budget is already exhausted, returns event.is_set() immediately
            - Otherwise, waits up to min(remaining_budget, max_single_wait)
        """
        # Fast path: already set
        if event.is_set():
            return True
        
        # Fast path: budget already gone
        if self.exhausted:
            return event.is_set()
        
        # Track start time for max_single_wait
        single_wait_start = time.monotonic() if max_single_wait else None
        
        # Poll with granularity until event set or budget/maxwait exhausted
        while not self.exhausted:
            # Check max_single_wait limit
            if max_single_wait is not None:
                single_elapsed = time.monotonic() - single_wait_start
                if single_elapsed >= max_single_wait:
                    log.debug(f"max_single_wait ({max_single_wait:.2f}s) exceeded")
                    break
                single_remaining = max_single_wait - single_elapsed
            else:
                single_remaining = float('inf')
            
            # Wait the minimum of: budget remaining, single wait remaining, granularity
            wait_time = min(self.remaining, single_remaining, self.WAIT_GRANULARITY_SEC)
            if wait_time <= 0:
                break
            if event.wait(timeout=wait_time):
                return True
        
        # Final check after loop exits
        return event.is_set()
    
    def record_chunk_processed(self):
        """Record that a chunk was successfully processed."""
        self._chunks_processed += 1
    
    def record_chunk_skipped(self):
        """Record that a chunk was skipped due to budget exhaustion."""
        self._chunks_skipped += 1
    
    @property
    def chunks_processed(self) -> int:
        """Number of chunks successfully processed within budget."""
        return self._chunks_processed
    
    @property
    def chunks_skipped(self) -> int:
        """Number of chunks skipped due to budget exhaustion."""
        return self._chunks_skipped
    
    def __repr__(self):
        return (f"TimeBudget(max={self.max_seconds:.2f}s, "
                f"elapsed={self.elapsed:.2f}s, "
                f"remaining={self.remaining:.2f}s, "
                f"exhausted={self.exhausted})")


class Getter(object):
    queue = None
    workers = None
    WORKING = None
    session = None

    def __init__(self, num_workers):
        
        self.count = 0
        self.queue = PriorityQueue()
        self.workers = []
        self.WORKING = threading.Event()
        self.WORKING.set()
        self.localdata = threading.local()
        # Thread-local sessions created in worker() to avoid shared-state contention

        for i in range(num_workers):
            t = threading.Thread(target=self.worker, args=(i,), daemon=True)
            t.start()
            self.workers.append(t)

        #self.stat_t = t = threading.Thread(target=self.show_stats, daemon=True)
        #self.stat_t.start()


    def stop(self):
        self.WORKING.clear()
        for t in self.workers:
            t.join()
        # If a stats thread was started, join it as well
        stat_thread = getattr(self, 'stat_t', None)
        if stat_thread is not None:
            stat_thread.join()

    def worker(self, idx):
        global STATS
        self.localdata.idx = idx
        
        # Create thread-local session with HTTP/2 support (if available)
        # Uses httpx for HTTP/2 multiplexing, falls back to requests
        try:
            pool_size = max(4, int(int(CFG.autoortho.fetch_threads) * 1.5))
            self.localdata.session = create_http_session(pool_size=pool_size)
        except Exception as _e:
            log.warning(f"Failed to initialize thread-local session: {_e}")
            self.localdata.session = requests.Session()
        
        while self.WORKING.is_set():
            try:
                obj, args, kwargs = self.queue.get(timeout=5)
                #log.debug(f"Got: {obj} {args} {kwargs}")
            except Empty:
                #log.debug(f"timeout, continue")
                #log.info(f"Got {self.counter}")
                continue

            #STATS.setdefault('count', 0) + 1
            bump('count', 1)

            try:
                # Mark chunk as in-flight (Chunk always has these attributes)
                obj.in_queue = False
                obj.in_flight = True
                
                if not self.get(obj, *args, **kwargs):
                    # Check if chunk is permanently failed before re-submitting
                    if obj.permanent_failure:
                        log.debug(f"Chunk {obj} permanently failed ({obj.failure_reason}), not re-submitting")
                        continue
                    log.warning(f"Failed getting: {obj} {args} {kwargs}, re-submit.")
                    self.submit(obj, *args, **kwargs)
            except Exception as err:
                log.error(f"ERROR {err} getting: {obj} {args} {kwargs}, re-submit.")
                # Don't re-submit if permanently failed
                if obj.permanent_failure:
                    log.debug(f"Chunk {obj} permanently failed during exception, not re-submitting")
                    continue
                self.submit(obj, *args, **kwargs)
            finally:
                obj.in_flight = False

    def get(obj, *args, **kwargs):
        raise NotImplementedError

    def submit(self, obj, *args, **kwargs):
        # Don't queue permanently failed chunks (Chunk always has this attribute)
        if obj.permanent_failure:
            return
        # Coalesce duplicate chunk submissions
        if obj.ready.is_set():
            return  # Already done
        if obj.in_queue:
            return  # Already queued
        if obj.in_flight:
            return  # Currently downloading
        obj.in_queue = True
        self.queue.put((obj, args, kwargs))

    def show_stats(self):
        while self.WORKING.is_set():
            log.info(f"{self.__class__.__name__} got: {self.count}")
            time.sleep(10)
        log.info(f"Exiting {self.__class__.__name__} stat thread.  Got: {self.count} total")


class ChunkGetter(Getter):
    def get(self, obj, *args, **kwargs):
        if obj.ready.is_set():
            log.info(f"{obj} already retrieved.  Exit")
            return True

        kwargs['idx'] = self.localdata.idx
        # Use thread-local session
        kwargs['session'] = getattr(self.localdata, 'session', None) or requests
        #log.debug(f"{obj}, {args}, {kwargs}")
        return obj.get(*args, **kwargs)

chunk_getter = ChunkGetter(int(CFG.autoortho.fetch_threads))

#class TileGetter(Getter):
#    def get(self, obj, *args, **kwargs):
#        log.debug(f"{obj}, {args}, {kwargs}")
#        return obj.get(*args)
#
#tile_getter = TileGetter(8)

log.info(f"chunk_getter: {chunk_getter}")
#log.info(f"tile_getter: {tile_getter}")

# ============================================================================
# ASYNC CACHE WRITER
# ============================================================================
# A lightweight executor for background cache writes. This allows downloaded
# chunks to be marked as "ready" immediately after download, rather than
# waiting for disk I/O to complete. Cache writes are fire-and-forget since
# a failed write only affects future cache hits, not current processing.
#
# Using 2 workers: cache writes are I/O-bound, not CPU-bound, so a small pool
# is sufficient. More workers would just contend on disk I/O.
# ============================================================================
_cache_write_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, 
    thread_name_prefix="cache_writer"
)


def _async_cache_write(chunk):
    """Fire-and-forget cache write. Errors are logged but don't affect processing."""
    try:
        # Check if cache directory still exists (may have been cleaned up by temp directory)
        # This prevents errors when diagnose() temp directories are deleted before async write
        if not os.path.exists(chunk.cache_dir):
            log.debug(f"Cache dir gone for {chunk}, skipping async write")
            return
        chunk.save_cache()
    except (FileNotFoundError, OSError) as e:
        # Directory may have been deleted between check and write (race condition)
        # This is expected for temporary directories, so just log at debug level
        log.debug(f"Async cache write skipped for {chunk}: {e}")
    except Exception as e:
        log.debug(f"Async cache write failed for {chunk}: {e}")


def shutdown_cache_writer():
    """Shutdown the cache writer executor gracefully. Called during module cleanup."""
    try:
        _cache_write_executor.shutdown(wait=False)
    except Exception:
        pass


def flush_cache_writer(timeout=30.0):
    """Wait for all pending cache writes to complete.
    
    This is useful for tests that need to verify cache contents after
    get_img() returns. Since cache writes are asynchronous, you need to
    call this before checking for cached files.
    
    This function shuts down the executor and recreates it to ensure
    all pending tasks complete. This is the only reliable way to guarantee
    all writes are flushed on all platforms (especially Windows).
    
    Args:
        timeout: Maximum time to wait in seconds (default: 30.0)
    """
    global _cache_write_executor
    try:
        # Shutdown with wait=True ensures ALL pending tasks complete
        _cache_write_executor.shutdown(wait=True)
    except Exception:
        pass
    finally:
        # Recreate the executor for subsequent operations
        _cache_write_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="cache_writer"
        )


# ============================================================================
# SPATIAL PREFETCHER
# ============================================================================
# Proactively downloads tile chunks ahead of the aircraft based on position
# and heading. This reduces in-flight stutters by having tiles ready before
# X-Plane requests them.
#
# Key design principles:
# 1. Low priority: Prefetched chunks don't compete with immediate requests
# 2. Configurable: Users can enable/disable and tune parameters
# 3. Conservative: Doesn't overwhelm network or CPU resources
# 4. Non-blocking: Runs in background, never blocks main processing
# ============================================================================

class SpatialPrefetcher:
    """
    Background prefetcher that anticipates tile needs based on aircraft movement.
    
    The prefetcher periodically checks aircraft position and heading, calculates
    which tiles will be needed, and submits low-priority download requests.
    """
    
    # Priority offset for prefetched chunks (higher = lower priority)
    # This ensures prefetch doesn't compete with immediate X-Plane requests
    PREFETCH_PRIORITY_OFFSET = 100
    
    # Minimum speed (m/s) to trigger prefetching - don't prefetch when taxiing
    MIN_SPEED_MPS = 25  # ~50 knots
    
    def __init__(self):
        """Initialize the prefetcher with configuration from aoconfig."""
        self._thread = None
        self._stop_event = threading.Event()
        self._running = False
        self._prefetch_count = 0
        self._tile_cacher = None
        
        # Track recently prefetched to avoid duplicates
        # Use an LRU-like structure with max size
        self._recently_prefetched = set()
        self._max_recent = 500
        
        # Load configuration
        self._load_config()
        
    def _load_config(self):
        """Load prefetch configuration from aoconfig."""
        self.enabled = getattr(CFG.autoortho, 'prefetch_enabled', True)
        # Lookahead is now configured in MINUTES, convert to seconds
        lookahead_min = float(getattr(CFG.autoortho, 'prefetch_lookahead', 10))
        self.lookahead_sec = lookahead_min * 60  # Convert minutes to seconds
        self.interval_sec = float(getattr(CFG.autoortho, 'prefetch_interval', 2.0))
        self.max_chunks = int(getattr(CFG.autoortho, 'prefetch_max_chunks', 24))
        
        # Clamp to reasonable ranges (1-60 minutes = 60-3600 seconds)
        self.lookahead_sec = max(60, min(3600, self.lookahead_sec))
        self.interval_sec = max(1.0, min(10.0, self.interval_sec))
        self.max_chunks = max(8, min(64, self.max_chunks))
        
    def set_tile_cacher(self, tile_cacher):
        """Set the tile cacher reference for accessing tiles."""
        self._tile_cacher = tile_cacher
        
    def start(self):
        """Start the background prefetcher thread."""
        if self._running:
            return
            
        if not self.enabled:
            log.info("Spatial prefetcher disabled by configuration")
            return
            
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._prefetch_loop,
            name="SpatialPrefetcher",
            daemon=True
        )
        self._thread.start()
        log.info(f"Spatial prefetcher started (lookahead={self.lookahead_sec/60:.0f}min, "
                f"interval={self.interval_sec}s, max_chunks={self.max_chunks})")
        
    def stop(self):
        """Stop the background prefetcher thread."""
        if not self._running:
            return
            
        self._stop_event.set()
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        log.info(f"Spatial prefetcher stopped (prefetched {self._prefetch_count} chunks total)")
        bump('prefetch_total', self._prefetch_count)
        
    def _prefetch_loop(self):
        """Main prefetch loop - runs in background thread."""
        while not self._stop_event.is_set():
            try:
                self._do_prefetch_cycle()
            except Exception as e:
                log.debug(f"Prefetch cycle error: {e}")
            
            # Wait before next cycle, but allow early exit on stop
            self._stop_event.wait(timeout=self.interval_sec)
    
    def _do_prefetch_cycle(self):
        """
        Execute one prefetch cycle.

        If SimBrief flight data is loaded and enabled, prefetches along the
        flight plan waypoints. Otherwise, uses velocity-based prediction.
        
        Falls back to velocity-based prefetching if aircraft deviates from route.
        """
        # Check if tile_cacher is available
        if self._tile_cacher is None:
            return

        # Always use instantaneous position (that's where we actually are)
        if not datareftracker.data_valid or not datareftracker.connected:
            return
            
        lat = datareftracker.lat
        lon = datareftracker.lon

        # Validate position data
        if lat < -90 or lat > 90 or lon < -180 or lon > 180:
            return

        # Check if SimBrief flight path prefetching should be used
        if self._should_use_simbrief_prefetch(lat, lon):
            chunks_submitted = self._prefetch_along_flight_plan(lat, lon)
            if chunks_submitted > 0:
                self._prefetch_count += chunks_submitted
                log.debug(f"Prefetched {chunks_submitted} chunks along flight plan (total: {self._prefetch_count})")
                bump('prefetch_chunk_count', chunks_submitted)
            return
        
        # Fall back to velocity-based prefetching
        self._do_velocity_prefetch_cycle(lat, lon)
    
    def _should_use_simbrief_prefetch(self, lat: float, lon: float) -> bool:
        """
        Check if SimBrief flight path prefetching should be used.
        
        Returns True if:
        - SimBrief flight data is loaded
        - use_flight_data toggle is enabled
        - Aircraft is on-route (within deviation threshold)
        """
        # Check if SimBrief integration is enabled
        if not hasattr(CFG, 'simbrief'):
            return False
        
        use_flight_data = getattr(CFG.simbrief, 'use_flight_data', False)
        if isinstance(use_flight_data, str):
            use_flight_data = use_flight_data.lower() in ('true', '1', 'yes', 'on')
        
        if not use_flight_data:
            return False
        
        # Check if flight data is loaded
        if not simbrief_flight_manager.is_loaded:
            return False
        
        # Get deviation threshold from config
        deviation_threshold = float(getattr(CFG.simbrief, 'route_deviation_threshold_nm', 40))
        
        # Check if aircraft is on-route
        return simbrief_flight_manager.is_on_route(lat, lon, deviation_threshold)
    
    def _prefetch_along_flight_plan(self, lat: float, lon: float) -> int:
        """
        Prefetch tiles along the SimBrief flight plan waypoints.
        
        Gets upcoming waypoints from current position and prefetches tiles
        in a radius around each waypoint, working forward from current position.
        
        Uses the flight plan altitude at each waypoint to determine the
        appropriate zoom level, matching what will actually be displayed.
        
        Returns number of chunks submitted.
        """
        chunks_submitted = 0
        
        # Get prefetch radius from config
        prefetch_radius_nm = float(getattr(CFG.simbrief, 'route_prefetch_radius_nm', 40))
        
        # Get upcoming fixes (limited number to avoid overwhelming)
        upcoming_fixes = simbrief_flight_manager.get_upcoming_fixes(lat, lon, count=15)
        
        if not upcoming_fixes:
            return 0
        
        # Prefetch around each upcoming fix, stopping when we hit max chunks
        for fix in upcoming_fixes:
            if chunks_submitted >= self.max_chunks:
                break
            
            # Determine zoom level based on the AGL altitude at this waypoint
            # AGL (Above Ground Level) is used because it represents actual height
            # above the terrain being viewed, which is more relevant for imagery quality
            zoom_level = self._get_zoom_for_altitude(fix.altitude_agl_ft)
            
            log.debug(f"Prefetch fix {fix.ident}: MSL={fix.altitude_ft}ft, "
                     f"GND={fix.ground_height_ft}ft, AGL={fix.altitude_agl_ft}ft -> ZL{zoom_level}")
            
            # Prefetch tiles around this waypoint at the appropriate zoom level
            submitted = self._prefetch_waypoint_area(
                fix.lat, fix.lon, prefetch_radius_nm, zoom_level
            )
            chunks_submitted += submitted
        
        return chunks_submitted
    
    def _prefetch_waypoint_area(self, waypoint_lat: float, waypoint_lon: float,
                                  radius_nm: float, zoom: int) -> int:
        """
        Prefetch tiles within a radius around a waypoint.
        
        Returns number of chunks submitted.
        """
        chunks_submitted = 0
        
        # Convert radius to degrees (approximate)
        # 1 nm ≈ 1/60 degree latitude
        radius_deg_lat = radius_nm / 60.0
        radius_deg_lon = radius_nm / (60.0 * math.cos(math.radians(waypoint_lat)))
        
        # Calculate tile coordinates
        n = 2 ** zoom
        
        def latlon_to_tile(lat, lon):
            """Convert lat/lon to tile coordinates."""
            x = int((lon + 180) / 360 * n)
            lat_clamped = max(-85.0511, min(85.0511, lat))
            lat_rad = math.radians(lat_clamped)
            y = int((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n)
            return (x, y)
        
        # Get tile range for the area
        col_center, row_center = latlon_to_tile(waypoint_lat, waypoint_lon)
        col_min, row_min = latlon_to_tile(
            waypoint_lat + radius_deg_lat,
            waypoint_lon - radius_deg_lon
        )
        col_max, row_max = latlon_to_tile(
            waypoint_lat - radius_deg_lat,
            waypoint_lon + radius_deg_lon
        )
        
        # Limit the area to prevent too many tiles
        max_tiles_per_dim = 3
        if row_max - row_min > max_tiles_per_dim:
            row_min = row_center - max_tiles_per_dim // 2
            row_max = row_center + max_tiles_per_dim // 2
        if col_max - col_min > max_tiles_per_dim:
            col_min = col_center - max_tiles_per_dim // 2
            col_max = col_center + max_tiles_per_dim // 2
        
        # Prefetch tiles in the area
        for row in range(row_min, row_max + 1):
            for col in range(col_min, col_max + 1):
                if chunks_submitted >= self.max_chunks:
                    return chunks_submitted
                
                # Create unique key for this tile
                tile_key = (row, col, zoom)
                
                # Skip if recently prefetched
                if tile_key in self._recently_prefetched:
                    continue
                
                # Add to recently prefetched
                self._recently_prefetched.add(tile_key)
                if len(self._recently_prefetched) > self._max_recent:
                    try:
                        self._recently_prefetched.pop()
                    except KeyError:
                        pass
                
                # Prefetch this tile
                submitted = self._prefetch_tile(row, col, zoom)
                chunks_submitted += submitted
        
        return chunks_submitted
    
    def _get_zoom_for_altitude(self, altitude_ft: int) -> int:
        """
        Get the appropriate zoom level for a given altitude.
        
        Uses the TileCacher's dynamic zoom manager if available and in dynamic mode.
        Falls back to fixed max_zoom otherwise.
        
        Args:
            altitude_ft: Altitude in feet
            
        Returns:
            Appropriate zoom level for the altitude
        """
        # Check if we have access to the tile cacher and it's in dynamic mode
        if self._tile_cacher is not None:
            max_zoom_mode = getattr(CFG.autoortho, 'max_zoom_mode', 'fixed')
            if max_zoom_mode == 'dynamic' and hasattr(self._tile_cacher, 'dynamic_zoom_manager'):
                zoom = self._tile_cacher.dynamic_zoom_manager.get_zoom_for_altitude(altitude_ft)
                log.debug(f"Dynamic zoom for altitude {altitude_ft}ft: ZL{zoom}")
                return zoom
        
        # Fall back to fixed max_zoom
        return int(getattr(CFG.autoortho, 'max_zoom', 16))
    
    def _get_predicted_altitude(self, lat: float, lon: float, 
                                  hdg: float, spd: float, 
                                  target_lat: float, target_lon: float) -> int:
        """
        Predict altitude at a target position based on current flight parameters.
        
        Uses vertical speed to estimate altitude change over the distance.
        
        Returns:
            Predicted altitude in feet
        """
        # Get current altitude AGL and vertical speed
        with datareftracker._lock:
            if not datareftracker.data_valid:
                return int(getattr(CFG.autoortho, 'max_zoom', 16))
            current_alt = datareftracker.alt_agl_ft
        
        averages = datareftracker.get_flight_averages()
        if averages is None:
            return int(current_alt)
        
        vs_fpm = averages.get('vertical_speed_fpm', 0)
        
        # Calculate distance to target
        delta_lat = target_lat - lat
        delta_lon = target_lon - lon
        cos_lat = math.cos(math.radians(lat))
        distance_m = math.sqrt((delta_lat * 111320) ** 2 + (delta_lon * 111320 * cos_lat) ** 2)
        
        # Calculate time to reach target
        if spd > 0:
            time_sec = distance_m / spd
        else:
            time_sec = 0
        
        # Predict altitude change
        alt_change = vs_fpm * (time_sec / 60)  # Convert seconds to minutes
        predicted_alt = current_alt + alt_change
        
        # Clamp to reasonable values
        return max(0, int(predicted_alt))
    
    def _do_velocity_prefetch_cycle(self, lat: float, lon: float):
        """
        Execute velocity-based prefetch cycle.
        
        Uses heading and speed to predict future position and prefetch tiles.
        Uses predicted altitude to determine appropriate zoom level.
        """
        # Try to get 60-second averaged flight data for stable prediction
        averages = datareftracker.get_flight_averages()

        if averages is not None:
            # Use averaged heading and speed for more stable predictions
            hdg = averages['heading']
            spd = averages['ground_speed_mps']
        else:
            # Fall back to instantaneous values if no averages available
            hdg = datareftracker.hdg
            spd = datareftracker.spd

        # Don't prefetch if moving slowly (taxiing, parked)
        if spd < self.MIN_SPEED_MPS:
            return

        # Calculate predicted position using (potentially averaged) heading/speed
        distance_m = spd * self.lookahead_sec

        # Convert heading to radians
        hdg_rad = math.radians(hdg)
        
        # Calculate offset in degrees
        # 1 degree latitude ≈ 111,320m
        # 1 degree longitude ≈ 111,320m * cos(latitude)
        delta_lat = (distance_m * math.cos(hdg_rad)) / 111320
        cos_lat = math.cos(math.radians(lat))
        if cos_lat > 0.01:  # Avoid division issues near poles
            delta_lon = (distance_m * math.sin(hdg_rad)) / (111320 * cos_lat)
        else:
            delta_lon = 0
        
        predicted_lat = lat + delta_lat
        predicted_lon = lon + delta_lon
        
        # Calculate predicted altitude at destination for zoom level determination
        predicted_alt = self._get_predicted_altitude(lat, lon, hdg, spd, predicted_lat, predicted_lon)
        zoom_level = self._get_zoom_for_altitude(predicted_alt)
        
        # Get tiles along the flight path at the appropriate zoom level
        chunks_submitted = self._prefetch_along_path(
            lat, lon, predicted_lat, predicted_lon, zoom_level
        )
        
        if chunks_submitted > 0:
            self._prefetch_count += chunks_submitted
            log.debug(f"Prefetched {chunks_submitted} chunks at ZL{zoom_level} (alt={predicted_alt}ft, total: {self._prefetch_count})")
            bump('prefetch_chunk_count', chunks_submitted)
    
    def _prefetch_along_path(self, lat1, lon1, lat2, lon2, zoom_level: int = None):
        """
        Prefetch chunks for tiles along the flight path.
        
        Args:
            lat1, lon1: Start position (current aircraft position)
            lat2, lon2: End position (predicted position)
            zoom_level: Zoom level to prefetch at. If None, uses config max_zoom.
        
        Returns number of chunks submitted for prefetch.
        """
        chunks_submitted = 0
        
        # Use provided zoom level or fall back to config
        if zoom_level is None:
            zoom_level = int(getattr(CFG.autoortho, 'max_zoom', 16))
        
        # Calculate tile coordinates using Web Mercator projection
        n = 2 ** zoom_level
        
        def latlon_to_tile(lat, lon):
            """Convert lat/lon to tile coordinates."""
            x = int((lon + 180) / 360 * n)
            # Clamp latitude to valid Mercator range
            lat_clamped = max(-85.0511, min(85.0511, lat))
            lat_rad = math.radians(lat_clamped)
            y = int((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n)
            return (x, y)
        
        # Get tile coordinates for start and end of path
        col1, row1 = latlon_to_tile(lat1, lon1)
        col2, row2 = latlon_to_tile(lat2, lon2)
        
        # Get tiles in the bounding box (prioritize destination end)
        row_min, row_max = min(row1, row2), max(row1, row2)
        col_min, col_max = min(col1, col2), max(col1, col2)
        
        # Limit the search area to avoid excessive prefetching
        max_tiles = 4  # Max tiles per dimension
        if row_max - row_min > max_tiles:
            row_min = row2 - max_tiles // 2
            row_max = row2 + max_tiles // 2
        if col_max - col_min > max_tiles:
            col_min = col2 - max_tiles // 2
            col_max = col2 + max_tiles // 2
        
        # Iterate from destination back to current position
        for row in range(row_max, row_min - 1, -1):
            for col in range(col_min, col_max + 1):
                if chunks_submitted >= self.max_chunks:
                    return chunks_submitted
                    
                # Create unique key for this tile
                tile_key = (row, col, zoom_level)
                
                # Skip if recently prefetched
                if tile_key in self._recently_prefetched:
                    continue
                    
                # Add to recently prefetched (with size limit)
                self._recently_prefetched.add(tile_key)
                if len(self._recently_prefetched) > self._max_recent:
                    # Remove oldest (arbitrary since set, but prevents unbounded growth)
                    try:
                        self._recently_prefetched.pop()
                    except KeyError:
                        pass
                
                # Prefetch this tile's chunks
                submitted = self._prefetch_tile(row, col, zoom_level)
                chunks_submitted += submitted
                
        return chunks_submitted
    
    def _prefetch_tile(self, row, col, zoom):
        """
        Submit prefetch requests for a tile's chunks.
        
        Returns number of chunks submitted.
        
        IMPORTANT: Uses _open_tile()/_close_tile() pair to properly manage refs.
        This ensures prefetched tiles can be evicted when no longer needed.
        """
        # Get maptype from config
        maptype = getattr(CFG.autoortho, 'maptype_override', None)
        if not maptype or maptype == "Use tile default":
            maptype = "EOX"
        
        tile = None
        try:
            # Use _open_tile() to properly increment refs (balanced with _close_tile below)
            # This ensures prefetched tiles don't accumulate refs and block eviction.
            # Previous bug: _get_tile() incremented refs for new tiles but we never
            # called _close_tile(), causing refs to stay elevated and preventing eviction.
            tile = self._tile_cacher._open_tile(row, col, maptype, zoom)
            if not tile:
                return 0
                
            # Get chunks for highest detail mipmap
            if not tile.chunks or zoom not in tile.chunks:
                # Trigger chunk creation by requesting a mipmap
                # But don't actually wait - just ensure chunks exist
                return 0
                
            chunks = tile.chunks.get(zoom, [])
            submitted = 0
            
            for chunk in chunks:
                # Skip if already ready, in flight, or failed
                if chunk.ready.is_set():
                    continue
                if chunk.in_queue or chunk.in_flight:
                    continue
                if chunk.permanent_failure:
                    continue
                    
                # Set low priority (higher number = lower priority)
                chunk.priority = self.PREFETCH_PRIORITY_OFFSET
                
                # Submit to chunk getter
                chunk_getter.submit(chunk)
                submitted += 1
                
                # Stop if we've submitted enough from this tile
                if submitted >= 4:  # Max 4 chunks per tile per cycle
                    break
                    
            return submitted
            
        except Exception as e:
            log.debug(f"Prefetch error for tile {row},{col}: {e}")
            return 0
        finally:
            # Always close the tile to decrement refs - this balances _open_tile() above.
            # The tile stays in the cache (enable_cache=True) but with refs decremented,
            # allowing eviction to work properly when the tile is no longer needed.
            if tile is not None:
                try:
                    self._tile_cacher._close_tile(row, col, maptype, zoom)
                except Exception:
                    pass  # Don't let close errors mask the original exception


# Global prefetcher instance
spatial_prefetcher = SpatialPrefetcher()


def start_prefetcher(tile_cacher):
    """Start the spatial prefetcher with the given tile cacher."""
    spatial_prefetcher.set_tile_cacher(tile_cacher)
    spatial_prefetcher.start()


def stop_prefetcher():
    """Stop the spatial prefetcher."""
    spatial_prefetcher.stop()


# HTTP status codes that indicate permanent failure (no retry)
PERMANENT_FAILURE_CODES = {400, 401, 404, 405, 406, 410, 451}

# HTTP status codes that need special handling
TRANSIENT_FAILURE_CODES = {408, 429, 500, 502, 503, 504}

# Max retries for transient failures before giving up
MAX_TRANSIENT_RETRIES = {
    408: 3,   # Request timeout - network issue
    429: 10,  # Rate limit - needs backoff
    500: 15,  # Internal error - might recover
    502: 8,   # Bad gateway - infrastructure
    503: 8,   # Service unavailable - infrastructure  
    504: 5,   # Gateway timeout - infrastructure
}


class Chunk(object):
    col = -1
    row = -1
    source = None
    chunk_id = ""
    priority = 0
    width = 256
    height = 256
    cache_dir = 'cache'
    
    attempt = 0

    starttime = 0
    fetchtime = 0

    ready = None
    data = None
    img = None
    url = None

    serverlist=['a','b','c','d']

    def __init__(self, col, row, maptype, zoom, priority=0, cache_dir='.cache'):
        self.col = col
        self.row = row
        self.zoom = zoom
        self.maptype = maptype
        self.cache_dir = cache_dir
        
        # Hack override maptype
        #self.maptype = "BI"

        # Set priority: use provided value or default to zoom level
        self.priority = priority if priority else zoom
        self.chunk_id = f"{col}_{row}_{zoom}_{maptype}"
        self.ready = threading.Event()
        self.ready.clear()
        self.download_started = threading.Event()  # Signal when download thread begins
        self.download_started.clear()
        if maptype == "Null":
            self.maptype = "EOX"

        # Failure tracking for circuit breaker
        self.permanent_failure = False
        self.failure_reason = None
        self.retry_count = 0

        # Coalescing flags to prevent duplicate submissions
        self.in_queue = False
        self.in_flight = False

        self.cache_path = os.path.join(self.cache_dir, f"{self.chunk_id}.jpg")
        
        # FIXED: Check cache during initialization and set ready if found
        if self.get_cache():
            self.download_started.set()  # Cache hit = "download" done immediately
            self.ready.set()
            log.debug(f"Chunk {self} initialized with cached data")

    def __lt__(self, other):
        return self.priority < other.priority

    def __repr__(self):
        return f"Chunk({self.col},{self.row},{self.maptype},{self.zoom},{self.priority})"

    def get_cache(self):
        if os.path.isfile(self.cache_path):
            bump('chunk_hit')
            cache_file = Path(self.cache_path)
            # Get data
            data = None
            # On Windows, the cache file can be briefly locked by AV or a concurrent writer.
            # Add a short retry/backoff loop to avoid spurious PermissionError / sharing violations.
            max_attempts = 5
            for attempt in range(1, max_attempts + 1):
                try:
                    data = cache_file.read_bytes()
                    break
                except PermissionError as e:
                    if attempt < max_attempts:
                        time.sleep(0.02 * attempt)
                        continue
                    log.warning(f"Permission denied reading cache {self}: {e}")
                    return False
                except FileNotFoundError:
                    # Raced with a concurrent replace/remove; treat as miss
                    return False
                except OSError as e:
                    winerr = getattr(e, 'winerror', None)
                    if winerr in (5, 32, 33) and attempt < max_attempts:
                        time.sleep(0.02 * attempt)
                        continue
                    log.debug(f"OSError reading cache {self}: {e}")
                    return False

            cache_file.touch()
            # Update modified data
            try:
                os.utime(self.cache_path, None)
            except (FileNotFoundError, PermissionError):
                pass 

            if _is_jpeg(data[:3]):
                #print(f"Found cache that is JPEG for {self}")
                self.data = data
                return True
            else:
                log.info(f"Loading file {self} not a JPEG! {data[:3]} path: {self.cache_path}")
                self.data = b''
                return False  # FIXED: Explicitly return False for corrupted cache
        else:
            bump('chunk_miss')
            return False

    def save_cache(self):
        # Snapshot data to avoid races with close() mutating self.data
        data = self.data
        if not data:
            return

        # Check if cache directory still exists (may have been deleted by temp cleanup)
        if not os.path.exists(self.cache_dir):
            log.debug(f"Cache directory gone for {self}, skipping save")
            return

        # Ensure cache directory exists
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except (FileNotFoundError, OSError):
            # Directory path is invalid or was deleted
            log.debug(f"Cannot create cache directory for {self}, skipping save")
            return
        except Exception:
            pass

        # Unique temp filename per writer to avoid collisions between threads/tiles
        temp_filename = os.path.join(self.cache_dir, f"{self.chunk_id}_{uuid.uuid4().hex}.tmp")

        # Write data to the unique temp file first
        try:
            with open(temp_filename, 'wb') as h:
                h.write(data)
        except FileNotFoundError as e:
            # Directory was deleted between check and write (race with temp cleanup)
            log.debug(f"Cache directory deleted during save for {self}: {e}")
            return
        except Exception as e:
            # Could not write temp file
            try:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
            except Exception:
                pass
            log.warning(f"Failed to save cache for {self}: {e}")
            return

        # Try to move into place atomically with a few retries for WinError 32
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                if os.path.exists(self.cache_path):
                    # Another writer got there first; clean up our temp
                    os.remove(temp_filename)
                    log.debug(f"Cache file already exists for {self}, skipping save (race) on attempt {attempt}")
                    return
                os.replace(temp_filename, self.cache_path)
                return
            except FileExistsError:
                try:
                    os.remove(temp_filename)
                except Exception:
                    pass
                log.debug(f"Another thread saved cache for {self}, removed temp file")
                return
            except OSError as e:
                if getattr(e, 'winerror', None) in (5, 32, 33) and attempt < max_attempts:
                    time.sleep(0.05 * attempt)
                    if os.path.exists(self.cache_path):
                        try:
                            if os.path.exists(temp_filename):
                                os.remove(temp_filename)
                        except Exception:
                            pass
                        return
                    continue
                try:
                    if os.path.exists(temp_filename):
                        os.remove(temp_filename)
                except Exception:
                    pass
                log.warning(f"Failed to save cache for {self}: {e}")
                return

    def get(self, idx=0, session=requests):
        log.debug(f"Getting {self}")
        
        # Signal that download has started (not waiting in queue anymore)
        self.download_started.set()

        if self.get_cache():
            self.ready.set()
            return True

        if not self.starttime:
            self.starttime = time.time()

        server_num = idx % (len(self.serverlist))
        server = self.serverlist[server_num]
        quadkey = _gtile_to_quadkey(self.col, self.row, self.zoom)

        # Hack override maptype
        #maptype = "ARC"

        MAPID = "s2cloudless-2023_3857"
        MATRIXSET = "g"
        MAPTYPES = {
            "EOX": f"https://{server}.tiles.maps.eox.at/wmts/?layer={MAPID}&style=default&tilematrixset={MATRIXSET}&Service=WMTS&Request=GetTile&Version=1.0.0&Format=image%2Fjpeg&TileMatrix={self.zoom}&TileCol={self.col}&TileRow={self.row}",
            "BI": f"https://t.ssl.ak.tiles.virtualearth.net/tiles/a{quadkey}.jpeg?g=15312",
            "GO2": f"http://mts{server_num}.google.com/vt/lyrs=s&x={self.col}&y={self.row}&z={self.zoom}",
            "ARC": f"http://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{self.zoom}/{self.row}/{self.col}",
            "NAIP": f"http://naip.maptiles.arcgis.com/arcgis/rest/services/NAIP/MapServer/tile/{self.zoom}/{self.row}/{self.col}",
            "USGS": f"https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{self.zoom}/{self.row}/{self.col}",
            "FIREFLY": f"https://fly.maptiles.arcgis.com/arcgis/rest/services/World_Imagery_Firefly/MapServer/tile/{self.zoom}/{self.row}/{self.col}",
            "YNDX": f"https://sat{server_num+1:02d}.maps.yandex.net/tiles?l=sat&v=3.1814.0&x={self.col}&y={self.row}&z={self.zoom}",
            "APPLE": f"https://sat-cdn.apple-mapkit.com/tile?style=7&size=1&scale=1&z={self.zoom}&x={self.col}&y={self.row}&v={apple_token_service.version}&accessKey={apple_token_service.apple_token}"
        }

        MAPTYPES_WITH_SERVER = ["YNDX", "EOX", "GO2"]

        self.url = MAPTYPES[self.maptype.upper()]
        #log.debug(f"{self} getting {url}")
        header = {
                "user-agent": "curl/7.68.0"
        }
        if self.maptype.upper() == "EOX":
            log.debug("EOX DETECTED")
            header.update({'referer': 'https://s2maps.eu/'})
       
        time.sleep((self.attempt/10))
        self.attempt += 1

        log.debug(f"Requesting {self.url} ..")
        
        resp = None
        try:

            resp = session.get(self.url, headers=header, timeout=(5, 20))
            status_code = resp.status_code

            if self.maptype.upper() == "APPLE" and status_code in (403, 410):
                log.warning("APPLE tile got %s; rotating token and retrying", status_code)
                apple_token_service.reset_apple_maps_token()
                MAPTYPES["APPLE"] = f"https://sat-cdn.apple-mapkit.com/tile?style=7&size=1&scale=1&z={self.zoom}&x={self.col}&y={self.row}&v={apple_token_service.version}&accessKey={apple_token_service.apple_token}"
                self.url = MAPTYPES[self.maptype.upper()]
                if resp is not None:
                    resp.close()
                resp = session.get(self.url, headers=header, timeout=(5, 20))
                status_code = resp.status_code

            if status_code != 200:
                log.warning(f"Failed with status {status_code} to get chunk {self}" + (" on server " + server if self.maptype.upper() in MAPTYPES_WITH_SERVER else "") + ".")
                bump_many({f"http_{status_code}": 1, "req_err": 1})
                
                # Check if this is a permanent failure
                if status_code in PERMANENT_FAILURE_CODES:
                    log.info(f"Chunk {self} permanently failed with {status_code}, marking as failed")
                    self.permanent_failure = True
                    self.failure_reason = status_code
                    self.data = b''  # Empty data
                    self.ready.set()  # Mark as ready (with no data) to unblock waiters
                    bump(f'chunk_permanent_fail_{status_code}')
                    return True  # Return True to stop worker retries
                
                # Check if transient failure has exceeded max retries
                if status_code in TRANSIENT_FAILURE_CODES:
                    self.retry_count += 1
                    max_retries = MAX_TRANSIENT_RETRIES.get(status_code, 5)
                    if self.retry_count >= max_retries:
                        log.warning(f"Chunk {self} exceeded {max_retries} retries for {status_code}, giving up")
                        self.permanent_failure = True
                        self.failure_reason = f"{status_code}_max_retries"
                        self.data = b''
                        self.ready.set()
                        bump(f'chunk_transient_fail_{status_code}_exhausted')
                        return True
                    # Increase backoff for rate limiting
                    if status_code == 429:
                        backoff_time = min(5, self.retry_count * 0.5)
                        log.debug(f"Rate limited, backing off for {backoff_time}s (attempt {self.retry_count}/{max_retries})")
                        time.sleep(backoff_time)
                    bump(f'chunk_transient_fail_{status_code}_retry')

                err = get_stat("req_err")
                if err > 50:
                    ok = get_stat("req_ok")
                    error_rate = err / ( err + ok )
                    if error_rate >= 0.10:
                        log.error(f"Very high network error rate detected : {error_rate * 100 : .2f}%")
                        log.error(f"Check your network connection, DNS, maptype choice, and firewall settings.")
                        # Enhanced circuit breaker: reduce wait times on severe error rates
                        if error_rate >= 0.25:
                            log.warning("Severe error rate (≥25%) detected, consider checking configuration")
                return False

            bump("req_ok")

            data = resp.content

            if _is_jpeg(data[:3]):
                log.debug(f"Data for {self} is JPEG")
                self.data = data
            else:
                # FFD8FF identifies image as a JPEG
                log.debug(f"Loading file {self} not a JPEG! {data[:3]} URL: {self.url}")
            #    return False
                self.data = b''

            bump('bytes_dl', len(self.data))
                
        except requests.exceptions.ConnectionError as err:
            # Connection pool exhausted or network issue - don't spam logs, just track
            bump('connection_pool_error')
            log.debug(f"Connection error for chunk {self}: {err}")
            return False
        except Exception as err:
            log.warning(f"Failed to get chunk {self} on server {server}. Err: {err} URL: {self.url}")
            return False
        finally:
            if resp:
                resp.close()

        self.fetchtime = time.monotonic() - self.starttime

        # OPTIMIZATION: Signal ready BEFORE cache write
        # The chunk data is already in memory (self.data), so we can mark it
        # as ready immediately. Cache writes are for future requests only.
        self.ready.set()
        
        # Submit cache write asynchronously - fire and forget
        # This prevents disk I/O from blocking the download worker thread
        try:
            _cache_write_executor.submit(_async_cache_write, self)
        except RuntimeError:
            # Executor shut down (program exiting), write synchronously as fallback
            self.save_cache()
        
        return True

    def close(self):
        """Release all references held by this Chunk so its memory can be reclaimed."""

        # Release image buffer if we created one
        if hasattr(self, 'img') and self.img is not None:
            try:
                # AoImage instances have a close() that frees underlying C memory
                if hasattr(self.img, "close"):
                    self.img.close()
            finally:
                self.img = None

        # Remove raw JPEG bytes
        self.data = None


class Tile(object):
    row = -1
    col = -1
    maptype = None
    zoom = -1
    min_zoom = 12
    width = 16
    height = 16
    baseline_zl = 12

    priority = -1
    #tile_condition = None
    _lock = None
    ready = None 

    chunks = None
    cache_file = None
    dds = None

    refs = None

    maxchunk_wait = float(CFG.autoortho.maxwait)
    imgs = None

    def __init__(self, col, row, maptype, zoom, min_zoom=0, priority=0,
            cache_dir=None, max_zoom=None):
        self.row = int(row)
        self.col = int(col)
        self.maptype = maptype
        self.tilename_zoom = int(zoom)
        self.chunks = {}
        self.cache_file = (-1, None)
        self.ready = threading.Event()
        self._lock = threading.RLock()
        self.refs = 0
        self.imgs = {}

        self.bytes_read = 0
        self.lowest_offset = 99999999
        
        # Track when this tile was first requested for accurate tile creation time stats
        self.first_request_time = None
        # Track if tile completion has been reported (to avoid double-counting)
        self._completion_reported = False
        # Tile-level time budget - shared across all mipmap builds for this tile
        # This ensures the budget limits the ENTIRE tile processing, not per-mipmap
        self._tile_time_budget = None
        
        # === SHARED FALLBACK CHUNK POOL ===
        # When multiple chunks fail and need the same parent chunk at a lower zoom,
        # we share the download instead of fetching it multiple times.
        # Key: (col, row, zoom) -> Chunk object
        # Example: 4 ZL16 chunks that all need (2,2,ZL15) share one download.
        self._fallback_chunk_pool = {}
        self._fallback_pool_lock = threading.Lock()
        
        # Track if lazy mipmap building has been triggered for this tile
        # This prevents repeated attempts after the first failure triggers a lazy build
        self._lazy_build_attempted = False

        #self.tile_condition = threading.Condition()
        if min_zoom:
            self.min_zoom = int(min_zoom)


        if cache_dir:
            self.cache_dir = cache_dir
        else:
            self.cache_dir = CFG.paths.cache_dir


        # Set max zoom level - if not specified, use original tile zoom (no capping)
        self.max_zoom = int(max_zoom) if max_zoom is not None else self.tilename_zoom
        # Hack override maptype
        #self.maptype = "BI"

        #self._find_cached_tiles()
        self.ready.clear()
        
        #self._find_cache_file()

        if not priority:
            self.priority = zoom

        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)

        if CFG.pydds.compressor.upper() == "ISPC":
            use_ispc=True
        else:
            use_ispc=False

        # TODO VRAM usage optimization only getting the max ZL of the zone instead of the default max zoom
        # This is very time consuming while loading flight, left commented out for now
        # self.actual_max_zoom = self._detect_available_max_zoom()
        self.tilezoom_diff = self.tilename_zoom - self.max_zoom

        if self.tilezoom_diff >= 0:

            self.chunks_per_row = self.width >> self.tilezoom_diff
            self.chunks_per_col = self.height >> self.tilezoom_diff
        else:
            self.chunks_per_row = self.width << (-self.tilezoom_diff)
            self.chunks_per_col = self.height << (-self.tilezoom_diff)


        if self.tilezoom_diff < 0:
            if self.tilezoom_diff < -1:
                raise ValueError(f"Tilezoom_diff is {self.tilezoom_diff} which is less than -1, which is not supported by X-Plane.")
            self.max_mipmap = 5 # Enforce a maximum of 5 mipmaps (8192 -> 256 max)
        else:
            self.max_mipmap = 4 # Enforce a maximum of 4 mipmaps (4096 -> 256 max)

        self.chunks_per_row = max(1, self.chunks_per_row)
        self.chunks_per_col = max(1, self.chunks_per_col)

        dds_width = self.chunks_per_row * 256
        dds_height = self.chunks_per_col * 256
        log.debug(f"Creating DDS at original size: {dds_width}x{dds_height} (ZL{self.max_zoom})")
            
        self.dds = pydds.DDS(dds_width, dds_height, ispc=use_ispc,
                dxt_format=CFG.pydds.format)

        self.id = f"{row}_{col}_{maptype}_{self.tilename_zoom}"


    def __lt__(self, other):
        return self.priority < other.priority

    def __repr__(self):
        return f"Tile({self.col}, {self.row}, {self.maptype}, {self.tilename_zoom}, min_zoom={self.min_zoom}, max_zoom={self.max_zoom}, max_mm={self.max_mipmap})"

    @locked
    def _create_chunks(self, quick_zoom=0, min_zoom=None):
        col, row, width, height, zoom, zoom_diff = self._get_quick_zoom(quick_zoom, min_zoom)

        if not self.chunks.get(zoom):
            self.chunks[zoom] = []
            log.debug(f"Creating chunks for zoom {zoom}: {width}x{height} grid starting at ({col},{row})")

            for r in range(row, row+height):
                for c in range(col, col+width):
                    # FIXED: Create chunk and let it check cache during initialization
                    chunk = Chunk(c, r, self.maptype, zoom, cache_dir=self.cache_dir)
                    self.chunks[zoom].append(chunk)
        else:
            log.debug(f"Reusing existing {len(self.chunks[zoom])} chunks for zoom {zoom}")

    def _find_cache_file(self):
        #with self.tile_condition:
        with self.tile_lock:
            for z in range(self.max_zoom, (self.min_zoom-1), -1):
                cache_file = os.path.join(self.cache_dir, f"{self.row}_{self.col}_{self.maptype}_{self.tilename_zoom}_{z}.dds")
                if os.path.exists(cache_file):
                    log.info(f"Found cache for {cache_file}...")
                    self.cache_file = (z, cache_file)
                    self.ready.set()
                    return

        #log.info(f"No cache found for {self}!")


    def _get_quick_zoom(self, quick_zoom=0, min_zoom=None):
        """Calculate tile parameters for the given zoom level.
        
        Args:
            quick_zoom: Target zoom level (0 means use max_zoom)
            min_zoom: Minimum allowed zoom level
            
        Returns:
            Tuple of (col, row, width, height, zoom, zoom_diff)
        """
        # Handle simple case: no quick zoom specified
        if not quick_zoom:
            return (self.col, self.row, self.width, self.height, self.max_zoom, 0)
        
        quick_zoom = int(quick_zoom)
        
        # Calculate the maximum zoom difference this tile can support
        max_supported_diff = min(self.max_zoom - quick_zoom, self.max_mipmap)
        
        # Determine minimum zoom level if not provided
        if min_zoom is None:
            min_zoom = max(self.max_zoom - max_supported_diff, self.min_zoom)
        
        # Clamp quick_zoom to minimum allowed value
        effective_zoom = max(quick_zoom, min_zoom)
        
        # Calculate actual zoom difference (limited by max_mipmap)
        zoom_diff = min(self.max_zoom - effective_zoom, self.max_mipmap)

        # Calculate tilename zoom difference
        tilename_zoom_diff = self.tilename_zoom - effective_zoom
        
        # Scale coordinates and dimensions based on zoom difference
        def scale_by_zoom_diff(value, diff):
            """Scale a value by 2^diff (positive diff scales down, negative scales up)"""
            if diff >= 0:
                return value >> diff  
            else:
                return value << (-diff)  
        
        scaled_col = scale_by_zoom_diff(self.col, tilename_zoom_diff)
        scaled_row = scale_by_zoom_diff(self.row, tilename_zoom_diff)
        scaled_width = max(1, scale_by_zoom_diff(self.width, tilename_zoom_diff))
        scaled_height = max(1, scale_by_zoom_diff(self.height, tilename_zoom_diff))
        
        return (scaled_col, scaled_row, scaled_width, scaled_height, effective_zoom, zoom_diff)


    def fetch(self, quick_zoom=0, background=False):
        self._create_chunks(quick_zoom)
        col, row, width, height, zoom, zoom_diff = self._get_quick_zoom(quick_zoom)

        for chunk in self.chunks[zoom]:
            chunk_getter.submit(chunk)

        for chunk in self.chunks[zoom]:
            ret = chunk.ready.wait()
            if not ret:
                log.error("Failed to get chunk.")

        return True
   
    def find_mipmap_pos(self, offset):
        for m in self.dds.mipmap_list:
            if offset < m.endpos:
                return m.idx
        return self.dds.mipmap_list[-1].idx

    def get_bytes(self, offset, length):
        
        # Guard against races where tile is being closed and DDS is cleared
        if self.dds is None:
            log.debug(f"GET_BYTES: DDS is None for {self}, likely closing; skipping")
            return True

        mipmap = self.find_mipmap_pos(offset)
        log.debug(f"Get_bytes for mipmap {mipmap} ...")
        if mipmap > self.max_mipmap:
            # Just get the entire mipmap
            self.get_mipmap(self.max_mipmap)
            return True

        # Exit if already retrieved
        if self.dds.mipmap_list[mipmap].retrieved:
            log.debug(f"We already have mipmap {mipmap} for {self}")
            return True

        mm = self.dds.mipmap_list[mipmap]
        if length >= mm.length:
            self.get_mipmap(mipmap)
            return True
        
        log.debug(f"Retrieving {length} bytes from mipmap {mipmap} offset {offset}")

        # how deep are we in a mipmap
        mm_offset = max(0, offset - self.dds.mipmap_list[mipmap].startpos)
        log.debug(f"MM_offset: {mm_offset}  Offset {offset}.  Startpos {self.dds.mipmap_list[mipmap]}")

        # Dynamically compute bytes-per-chunk-row for this mip level based on actual DDS dimensions
        base_width_px = max(4, int(self.dds.width) >> mipmap)
        base_height_px = max(4, int(self.dds.height) >> mipmap)
        blocksize = 8 if CFG.pydds.format == "BC1" else 16
        blocks_per_row = max(1, base_width_px // 4)
        bytes_per_row = blocks_per_row * blocksize
        # Each chunk-row is 256 px tall -> 64 blocks vertically
        bytes_per_chunk_row = bytes_per_row * 64

        # Compute start/end chunk-rows touched by the requested byte range
        startrow = mm_offset // bytes_per_chunk_row
        endrow = (mm_offset + max(0, length - 1)) // bytes_per_chunk_row

        # Clamp to valid range of chunk rows for this mipmap
        chunk_rows_in_mm = base_height_px // 256
        if chunk_rows_in_mm == 0:
            log.error(f"Chunk rows in mipmap {mipmap} is 0!  Base height: {base_height_px}  Mipmap: {mipmap}")

        if startrow >= chunk_rows_in_mm:
            startrow = chunk_rows_in_mm - 1
        if endrow >= chunk_rows_in_mm:
            endrow = chunk_rows_in_mm - 1
        if endrow < startrow:
            endrow = startrow

        # Prefetch one extra chunk-row ahead to reduce subsequent stalls
        if endrow < (chunk_rows_in_mm - 1):
            endrow = min(endrow + 1, chunk_rows_in_mm - 1)

        log.debug(f"Startrow: {startrow} Endrow: {endrow} bytes_per_chunk_row: {bytes_per_chunk_row} width_px: {base_width_px} height_px: {base_height_px}")
        
        # Pass the tile-level budget to get_img so it's shared across all partial reads
        new_im = self.get_img(mipmap, startrow, endrow,
                maxwait=self.get_maxwait(), time_budget=self._tile_time_budget)
        if not new_im:
            log.debug("No updates, so no image generated")
            return True

        # If tile is being closed concurrently, avoid touching DDS
        if self.dds is None:
            return True
        self.ready.clear()
        #log.info(new_im.size)
        
        start_time = time.time()

        # Only attempt partial compression from mipmap start
        if offset == 0:
            #compress_len = length
            compress_len = length - 128
        else:
            compress_len = 0

        try:
            self.dds.gen_mipmaps(new_im, mipmap, mipmap, compress_len)
        finally:
            pass
            # We may have retrieved a full image that could be saved for later
            # usage.  Don't close here.
            #new_im.close()

        # We haven't fully retrieved so unset flag; guard against DDS being cleared
        log.debug(f"UNSETTING RETRIEVED! {self}")
        try:
            if self.dds is not None and self.dds.mipmap_list:
                self.dds.mipmap_list[mipmap].retrieved = False
        except Exception:
            pass
        end_time = time.time()
        self.ready.set()

        if compress_len:
            tile_time = end_time - start_time
            partial_stats.set(mipmap, tile_time)
            # Record partial mm stats via counters for aggregation
            try:
                bump_many({
                    f"partial_mm_count:{mipmap}": 1,
                    f"partial_mm_time_total_ms:{mipmap}": int(tile_time * 1000)
                })
            except Exception:
                pass

        return True

    def read_dds_bytes(self, offset, length):
        log.debug(f"READ DDS BYTES: {offset} {length}")
        
        # Track when this tile was first requested (for accurate tile creation time stats)
        # NOTE: We do NOT create the time budget here because queue wait time shouldn't count.
        # The budget is created lazily in get_img() when actual processing begins.
        if self.first_request_time is None:
            self.first_request_time = time.monotonic()
            log.debug(f"READ_DDS_BYTES: First request for tile, budget will start when processing begins")
       
        if offset > 0 and offset < self.lowest_offset:
            self.lowest_offset = offset

        mm_idx = self.find_mipmap_pos(offset)
        mipmap = self.dds.mipmap_list[mm_idx]

        if offset == 0:
            # If offset = 0, read the header
            log.debug("READ_DDS_BYTES: Read header")
            self.get_bytes(0, length)
        else:
            # Dynamically scale the early-read heuristic based on actual mip-0 bytes per chunk-row
            blocksize = 8 if CFG.pydds.format == "BC1" else 16
            width_px_m0 = max(4, int(self.dds.width))
            blocks_per_row_m0 = max(1, width_px_m0 // 4)
            bytes_per_row_m0 = blocks_per_row_m0 * blocksize
            bytes_per_chunk_row_m0 = bytes_per_row_m0 * 64

            # If we're still within the first chunk-row of mipmap 0, just fetch from the start
            early_threshold = bytes_per_chunk_row_m0
            if mm_idx == 0 and offset < early_threshold:
                log.debug("READ_DDS_BYTES: Early region of mipmap 0 - fetching from start")
                self.get_bytes(0, length + offset)
            elif (offset + length) < mipmap.endpos:
                # Total length is within this mipmap.  Make sure we have it.
                log.debug(f"READ_DDS_BYTES: Detected middle read for mipmap {mipmap.idx}")
                if not mipmap.retrieved:
                    log.debug(f"READ_DDS_BYTES: Retrieve {mipmap.idx}")
                    self.get_mipmap(mipmap.idx)
            else:
                log.debug(f"READ_DDS_BYTES: Start before this mipmap {mipmap.idx}")
                # We already know we start before the end of this mipmap
                # We must extend beyond the length.
                
                # Get bytes prior to this mipmap
                self.get_bytes(offset, length)

                # Get the entire next mipmap
                self.get_mipmap(mm_idx + 1)
        
        self.bytes_read += length
        # Seek and return data
        self.dds.seek(offset)
        return self.dds.read(length)

    def write(self):
        outfile = os.path.join(self.cache_dir, f"{self.row}_{self.col}_{self.maptype}_{self.tilename_zoom}_{self.tilename_zoom}.dds")
        self.ready.clear()
        self.dds.write(outfile)
        self.ready.set()
        return outfile

    def get_header(self):
        outfile = os.path.join(self.cache_dir, f"{self.row}_{self.col}_{self.maptype}_{self.tilename_zoom}_{self.tilename_zoom}.dds")
        
        self.ready.clear()
        self.dds.write(outfile)
        self.ready.set()
        return outfile

    @locked
    def get_img(self, mipmap, startrow=0, endrow=None, maxwait=5, min_zoom=None, time_budget=None):
        #
        # Get an image for a particular mipmap
        #
        
        # === TIME BUDGET INITIALIZATION ===
        # The time budget is created lazily here on first get_img call, NOT when X-Plane
        # first requests the tile. This ensures queue wait time doesn't count against the budget.
        # Only actual processing time (chunk downloads, composition, compression) is measured.
        #
        # The budget is stored on self._tile_time_budget so it's shared across all mipmap
        # builds for this tile. If time_budget is explicitly provided (e.g., for pre-builds),
        # use that instead.
        
        if time_budget is not None:
            # Explicit budget provided (e.g., recursive pre-build calls) - use it
            log.debug(f"GET_IMG: Using provided budget (elapsed={time_budget.elapsed:.2f}s, remaining={time_budget.remaining:.2f}s)")
        elif self._tile_time_budget is not None:
            # Budget already created for this tile - reuse it
            time_budget = self._tile_time_budget
            log.debug(f"GET_IMG: Using existing tile budget (elapsed={time_budget.elapsed:.2f}s, remaining={time_budget.remaining:.2f}s)")
        else:
            # First get_img call for this tile - create the budget NOW (processing begins)
            use_time_budget = getattr(CFG.autoortho, 'use_time_budget', True)
            if isinstance(use_time_budget, str):
                use_time_budget = use_time_budget.lower() in ('true', '1', 'yes', 'on')
            
            if use_time_budget:
                base_budget = float(getattr(CFG.autoortho, 'tile_time_budget', 5.0))
                if CFG.autoortho.suspend_maxwait and not datareftracker.connected:
                    effective_budget = base_budget * 10.0
                    log.debug(f"GET_IMG: Startup mode - creating tile budget {effective_budget:.1f}s")
                else:
                    effective_budget = base_budget
                    log.debug(f"GET_IMG: Creating tile budget {effective_budget:.1f}s (processing begins now)")
                time_budget = TimeBudget(effective_budget)
            else:
                # Legacy mode: create budget from maxwait parameter
                effective_maxwait = self.get_maxwait() if maxwait == 5 else maxwait
                time_budget = TimeBudget(effective_maxwait)
                log.debug(f"GET_IMG: Legacy mode - budget {effective_maxwait:.1f}s")
            
            # Store on tile for sharing across all mipmap builds
            self._tile_time_budget = time_budget

        # === FALLBACK CONFIGURATION ===
        # Get fallback settings for use by the fallback chain in process_chunk().
        # 
        # NOTE: Pre-building of lower mipmaps was removed due to severe diminishing returns:
        # - Cost: Paid on 100% of new tiles (85 extra downloads + 1-3s delay)
        # - Benefit: Only helped when mipmap 0 chunks fail (<5% of cases)
        # - Alternative: Fallback 3 (Cascading Download) handles failures on-demand
        #   with much lower overhead since it only activates when needed.
        fallback_level = self.get_fallback_level()
        fallback_extends_budget = self.get_fallback_extends_budget()

        # Get effective zoom  
        zoom = min((self.max_zoom - mipmap), self.max_zoom)
        log.debug(f"GET_IMG: Default tile zoom: {self.tilename_zoom}, Requested Mipmap: {mipmap}, Requested mipmap zoom: {zoom}")
        col, row, width, height, zoom, zoom_diff = self._get_quick_zoom(zoom, min_zoom)
        log.debug(f"Will use:  Zoom: {zoom},  Zoom_diff: {zoom_diff}")        
        
        log.debug(f"GET_IMG: Final zoom {zoom} for mipmap {mipmap}, coords: {col}x{row}, size: {width}x{height}")
        
        # Do we already have this img?
        if mipmap in self.imgs:
            img_data = self.imgs[mipmap]
            # Unpack tuple format (new) or return directly (old format, backward compat)
            if isinstance(img_data, tuple):
                img = img_data[0]  # Extract just the image
                log.debug(f"GET_IMG: Found saved image: {img}")
                return img
            else:
                # Old format: just the image without metadata
                log.debug(f"GET_IMG: Found saved image (old format): {img_data}")
                return img_data

        log.debug(f"GET_IMG: MM List before { {x.idx:x.retrieved for x in self.dds.mipmap_list} }")
        if mipmap < len(self.dds.mipmap_list) and self.dds.mipmap_list[mipmap].retrieved:
            log.debug(f"GET_IMG: We already have mipmap {mipmap} for {self}")
            return

        if startrow == 0 and endrow is None:
            complete_img = True
        else:
            complete_img = False

        startchunk = 0
        endchunk = None
        # Determine start and end chunk based on the actual zoom level we're using
        # Use the width/height calculated for the capped zoom level
        chunks_per_row = width  # width already accounts for capping
        if startrow:
            startchunk = startrow * chunks_per_row
        if endrow is not None:
            endchunk = (endrow * chunks_per_row) + chunks_per_row
            
        log.debug(f"GET_IMG: Chunk indices - start: {startchunk}, end: {endchunk}, chunks_per_row: {chunks_per_row}")

        # Create chunks for the actual zoom level we'll download from
        self._create_chunks(zoom, min_zoom)
        chunks = self.chunks[zoom][startchunk:endchunk]
        log.debug(f"Start chunk: {startchunk}  End chunk: {endchunk}  Chunklen {len(self.chunks[zoom])} for zoom {zoom}")

        log.debug(f"GET_IMG: {self} : Retrieve mipmap for ZOOM: {zoom} MIPMAP: {mipmap}")
        data_updated = False
        log.debug(f"GET_IMG: {self} submitting chunks for zoom {zoom}.")
        for chunk in chunks:
            if not chunk.ready.is_set():
                log.debug(f"GET_IMG: Submitting chunk {chunk} for zoom {zoom}")
                # INVERTED: Lower detail (higher mipmap) = lower priority number (more urgent)
                # This ensures lower-detail tiles load first, minimizing missing tiles
                base_priority = self.max_mipmap - mipmap
                
                # During initial load (before flight starts), further deprioritize high-detail
                # to ensure lower mipmaps load completely first
                if CFG.autoortho.suspend_maxwait and not datareftracker.connected:
                    # Add penalty to high-detail mipmaps during initial load
                    # Mipmap 0 gets +20, mipmap 4 gets +0
                    initial_load_penalty = (self.max_mipmap - mipmap) * 5
                    base_priority = base_priority + initial_load_penalty
                
                # Apply spatial + predictive priority system
                # This considers distance from player and movement direction
                chunk.priority = _calculate_spatial_priority(
                    chunk.row, chunk.col, chunk.zoom, base_priority
                )
                
                chunk_getter.submit(chunk)
                data_updated = True
            else:
                log.debug(f"GET_IMG: Chunk {chunk} already ready - reusing for mipmap {mipmap}")

        # We've already determined this mipmap is not marked as 'retrieved' so we should create 
        # a new image, regardless here.
        #if not data_updated:
        #    log.info("No updates to chunks.  Exit.")
        #    return False

        # Calculate image dimensions based on the actual zoom level we're using for downloads
        # This creates smaller textures that save VRAM when zoom is capped
        img_width = 256 * width
        img_height = 256 * height
        
        log.debug(f"GET_IMG: Using download dimensions {width}x{height} chunks = {img_width}x{img_height} pixels")
        
        log.debug(f"GET_IMG: Create new image: Zoom: {zoom} | {(img_width, img_height)}")
        
        # === PRE-INITIALIZE BASE LAYER FOR MIPMAP 0 ===
        # When building mipmap 0 (highest detail), instead of starting with a blank 
        # missing_color image, we initialize with an upscaled lower mipmap.
        # This ensures any chunks that are skipped due to budget exhaustion still 
        # show a blurry-but-colored texture instead of visible holes.
        #
        # Why this works:
        # - Mipmaps are built 4->3->2->1->0, so lower mipmaps are usually available
        # - Chunks overwrite the base as they complete (progressive refinement)
        # - No hole detection needed - base layer guarantees coverage
        # - Single upscale operation (efficient) vs per-hole fills
        #
        # Fallback order: mipmap 1 (best quality) -> 2 -> 3 -> 4 -> missing_color
        
        new_im = None
        prefill_source_mm = None
        
        if mipmap == 0 and fallback_level >= 1:
            # Try to find the best available lower mipmap to use as base
            for source_mm in [1, 2, 3, 4]:
                if source_mm > self.max_mipmap:
                    continue
                    
                img_data = self.imgs.get(source_mm)
                if not img_data:
                    continue
                
                # Extract image from tuple format (image, col, row, zoom)
                if isinstance(img_data, tuple):
                    source_img = img_data[0]
                else:
                    source_img = img_data
                
                if source_img is None or getattr(source_img, '_freed', False):
                    continue
                
                # Calculate scale factor: mipmap 1->0 = 2x, mipmap 2->0 = 4x, etc.
                scale_factor = 1 << source_mm  # 2^source_mm
                
                # Verify source image dimensions match expected size
                expected_source_width = img_width >> source_mm
                expected_source_height = img_height >> source_mm
                
                if source_img._width != expected_source_width or source_img._height != expected_source_height:
                    log.debug(f"GET_IMG: Prefill skipping mipmap {source_mm} - size mismatch: "
                             f"got {source_img._width}x{source_img._height}, expected {expected_source_width}x{expected_source_height}")
                    continue
                
                # Upscale to mipmap 0 size
                log.debug(f"GET_IMG: Pre-initializing mipmap 0 base from mipmap {source_mm} "
                         f"({source_img._width}x{source_img._height} -> {img_width}x{img_height}, scale={scale_factor}x)")
                
                try:
                    new_im = source_img.scale(scale_factor)
                    if new_im is not None and new_im._width == img_width and new_im._height == img_height:
                        prefill_source_mm = source_mm
                        bump(f'mipmap0_prefill_from_mm{source_mm}')
                        log.debug(f"GET_IMG: Successfully pre-initialized base from mipmap {source_mm}")
                        break
                    else:
                        # Scale failed or wrong size - clean up and try next
                        if new_im is not None:
                            try:
                                new_im.close()
                            except Exception:
                                pass
                        new_im = None
                        log.debug(f"GET_IMG: Prefill scale from mipmap {source_mm} failed or wrong size")
                except Exception as e:
                    log.warning(f"GET_IMG: Prefill scale from mipmap {source_mm} exception: {e}")
                    new_im = None
        
        # Fall back to missing_color if no lower mipmap available or prefill disabled
        if new_im is None:
            new_im = AoImage.new(
                "RGBA",
                (img_width, img_height),
                (
                    CFG.autoortho.missing_color[0],
                    CFG.autoortho.missing_color[1],
                    CFG.autoortho.missing_color[2],
                ),
            )
            if mipmap == 0 and fallback_level >= 1:
                log.debug(f"GET_IMG: No lower mipmap available for prefill, using missing_color")
        else:
            log.debug(f"GET_IMG: Using prefilled base from mipmap {prefill_source_mm}")

        log.debug(f"GET_IMG: Will use image {new_im}")

        # Check if we have any chunks to process
        if len(chunks) == 0:
            log.warning(f"GET_IMG: No chunks created for zoom {zoom}, mipmap {mipmap}")
            # Return the missing_color filled image we already created
            # This ensures consistency - X-Plane sees missing_color, not arbitrary gray
            return new_im
            
        def process_chunk(chunk, skip_download_wait=False):
            """Process a single chunk and return (chunk, chunk_img, start_x, start_y)"""
            # Calculate position using native chunk size (no scaling)
            start_x = int(chunk.width * (chunk.col - col))
            start_y = int(chunk.height * (chunk.row - row))
            
            # PHASE 2 FIX #6 & #8: Validate coordinates before use
            # Negative coordinates indicate logic error or data corruption
            if start_x < 0 or start_y < 0:
                log.error(f"GET_IMG: Invalid negative coordinates: start_x={start_x}, start_y={start_y}, chunk.col={chunk.col}, chunk.row={chunk.row}, base col={col}, base row={row}")
                # Return placeholder to prevent crash
                return (chunk, None, 0, 0)
            
            # Check if coordinates would extend beyond image bounds
            if start_x + chunk.width > img_width or start_y + chunk.height > img_height:
                log.error(f"GET_IMG: Coordinates extend beyond image: pos=({start_x},{start_y}), size=({chunk.width}x{chunk.height}), image=({img_width}x{img_height})")
                # Return placeholder to prevent crash
                return (chunk, None, 0, 0)
            
            # === TIME BUDGET CHECK ===
            # Early exit if the time budget is exhausted. This is the key improvement:
            # instead of each chunk waiting maxwait seconds, we check the shared budget.
            if time_budget.exhausted:
                log.debug(f"Time budget exhausted (elapsed={time_budget.elapsed:.2f}s), skipping chunk {chunk}")
                time_budget.record_chunk_skipped()
                bump('chunk_budget_skipped')
                return (chunk, None, start_x, start_y)
            
            # Track if this chunk permanently failed
            is_permanent_failure = chunk.permanent_failure
            if is_permanent_failure:
                log.debug(f"Chunk {chunk} is permanently failed ({chunk.failure_reason}), will attempt fallbacks")
                bump(f'chunk_permanent_fail_{chunk.failure_reason}')
            
            # Smart timeout: only wait for download, not decode
            # If skip_download_wait is True, go straight to fallbacks (for chunks that never started)
            if skip_download_wait:
                chunk_ready = False
                log.debug(f"Chunk {chunk} never started downloading, skipping wait and going to fallbacks")
            elif chunk.ready.is_set():
                # Download already complete, just needs decode
                chunk_ready = True
                log.debug(f"Chunk {chunk} already downloaded, proceeding to decode")
            else:
                # Download in progress - use BOTH time budget AND per-chunk maxwait
                # This respects the total tile budget while also limiting per-chunk waits
                # The wait ends when either: budget exhausted, maxwait reached, or event set
                chunk_ready = time_budget.wait_with_budget(chunk.ready, max_single_wait=maxwait)
                if not chunk_ready:
                    log.debug(f"Chunk {chunk} wait ended (budget remaining={time_budget.remaining:.2f}s, maxwait={maxwait:.1f}s)")

            chunk_img = None
            decode_failed = False
            if chunk_ready and chunk.data:
                # We returned and have data!
                log.debug(f"GET_IMG(process_chunk(tid={threading.get_ident()})): Ready and found chunk data.")
                try:
                    with _decode_sem:
                        chunk_img = AoImage.load_from_memory(chunk.data)
                        if chunk_img is None:
                            log.warning(f"GET_IMG: load_from_memory returned None for {chunk}")
                            decode_failed = True
                except Exception as _e:
                    log.error(f"GET_IMG: load_from_memory exception for {chunk}: {_e}")
                    chunk_img = None
                    decode_failed = True
            elif chunk_ready and not chunk.data:
                # Download "succeeded" but returned empty data
                log.debug(f"GET_IMG: Chunk {chunk} ready but has no data")
                decode_failed = True
            
            # Determine if we need fallbacks:
            # - chunk didn't download in time (not chunk_ready)
            # - chunk is permanently failed (404, etc)
            # - chunk downloaded but decode failed (decode_failed)
            needs_fallback = not chunk_ready or is_permanent_failure or decode_failed
            
            # FALLBACK CHAIN (in order of preference):
            # Each fallback only runs if previous ones failed and fallback_level allows it.
            # For permanent failures (404, etc), we ALWAYS try fallbacks to get lower-zoom alternatives.
            #
            # fallback_level controls which fallbacks are enabled:
            #   0 = None: Skip all fallbacks (fastest, may have missing tiles)
            #   1 = Cache-only: Fallback 1 (disk cache) + Fallback 2 (built mipmaps)
            #   2 = Full: All fallbacks including Fallback 3 (network downloads)
            
            # Fallback 1: Search disk cache for lower-zoom JPEGs (enabled if fallback_level >= 1)
            if not chunk_img and needs_fallback and fallback_level >= 1:
                log.debug(f"GET_IMG(process_chunk): Fallback 1 - searching disk cache for backup chunk.")
                chunk_img = self.get_best_chunk(chunk.col, chunk.row, mipmap, zoom)
                # get_best_chunk bumps 'upscaled_from_jpeg_count' if successful
            
            # === LAZY BUILD TRIGGER ===
            # If Fallback 1 failed and we're building mipmap 0, trigger a lazy build
            # of a lower mipmap (mipmap 2) to enable Fallback 2 for subsequent failures.
            # This is on-demand (only when failures occur) unlike the removed pre-building.
            if not chunk_img and mipmap == 0 and fallback_level >= 1:
                if not self._lazy_build_attempted:
                    log.debug(f"GET_IMG(process_chunk): Triggering lazy build after first failure")
                    self._try_lazy_build_fallback_mipmap(time_budget)
            
            # Fallback 2: Scale from already-built mipmaps (enabled if fallback_level >= 1)
            # Now may have something to use thanks to lazy build above
            if not chunk_img and fallback_level >= 1:
                log.debug(f"GET_IMG(process_chunk): Fallback 2 - scaling from built mipmaps.")
                chunk_img = self.get_downscaled_from_higher_mipmap(mipmap, chunk.col, chunk.row, zoom)
                # Note: scaling function bumps its own counters (upscaled_chunk_count or downscaled_chunk_count)
            
            # Fallback 3: On-demand download of lower-detail chunks (enabled if fallback_level >= 2)
            # This is the expensive network fallback - only use when quality is prioritized
            if not chunk_img and needs_fallback and fallback_level >= 2:
                # Determine whether to respect or ignore the exhausted budget:
                # - If fallback_extends_budget is True: ignore exhausted budget (quality priority)
                # - If fallback_extends_budget is False: respect budget strictly (speed priority)
                if time_budget.exhausted and not fallback_extends_budget:
                    log.debug(f"GET_IMG(process_chunk): Skipping Fallback 3 - budget exhausted and fallback_extends_budget=False")
                else:
                    log.debug(f"GET_IMG(process_chunk): Fallback 3 - cascading fallback "
                             f"(budget_exhausted={time_budget.exhausted}, extends_budget={fallback_extends_budget}).")
                    # When extends_budget is True and budget is exhausted, pass time_budget=None
                    # so the cascading function uses its legacy fixed timeout instead of immediately giving up.
                    cascade_budget = None if (time_budget.exhausted and fallback_extends_budget) else time_budget
                    chunk_img = self.get_or_build_lower_mipmap_chunk(mipmap, chunk.col, chunk.row, zoom, 
                                                                      time_budget=cascade_budget)

            if not chunk_ready and not chunk_img and not is_permanent_failure:
                # === FINAL RETRY (OPTIMIZED) ===
                # Only worth retrying if the chunk download is still pending.
                # Pending means: in queue, in flight, or already completed (ready).
                # If none of these are true, the download failed and won't be retried
                # by the worker - so waiting would be pointless.
                download_pending = chunk.in_flight or chunk.in_queue or chunk.ready.is_set()
                
                if time_budget.exhausted:
                    log.debug(f"GET_IMG: Skipping final retry for {chunk} - budget exhausted")
                    time_budget.record_chunk_skipped()
                    bump('chunk_budget_skipped')
                elif not download_pending:
                    # Download failed completely and won't be retried - skip waiting
                    log.debug(f"GET_IMG: Skipping retry for {chunk} - download not pending "
                             f"(in_flight={chunk.in_flight}, in_queue={chunk.in_queue})")
                    bump('chunk_retry_skipped_not_pending')
                else:
                    # Download still in progress or just completed - worth checking
                    log.debug(f"GET_IMG: Final retry for {chunk} "
                             f"(in_flight={chunk.in_flight}, in_queue={chunk.in_queue}, "
                             f"budget remaining={time_budget.remaining:.2f}s)")
                    bump('retry_chunk_count')
                    
                    if chunk.ready.is_set():
                        # Completed during fallback attempts
                        chunk_ready = True
                        log.debug(f"Chunk {chunk} completed during fallbacks, proceeding to decode")
                    else:
                        # Still in progress - wait with reduced timeout (1s max for retry)
                        chunk_ready = time_budget.wait_with_budget(chunk.ready, max_single_wait=1.0)
                
                if chunk_ready and chunk.data:
                    log.debug(f"GET_IMG(process_chunk(tid={threading.get_ident()})): Final retry for {chunk}, SUCCESS!")
                    try:
                        with _decode_sem:
                            chunk_img = AoImage.load_from_memory(chunk.data)
                            if chunk_img is None:
                                log.warning(f"GET_IMG: load_from_memory returned None on retry for {chunk}")
                    except Exception as _e:
                        log.error(f"GET_IMG: load_from_memory exception on retry for {chunk}: {_e}")
                        chunk_img = None

            if not chunk_img:
                log.debug(f"GET_IMG(process_chunk(tid={threading.get_ident()})): Empty chunk data.  Skip.")
                bump('chunk_missing_count')
                if is_permanent_failure:
                    log.info(f"Chunk {chunk} permanently failed and all fallbacks exhausted")
            else:
                # Successfully processed this chunk within budget
                time_budget.record_chunk_processed()
                
            return (chunk, chunk_img, start_x, start_y)
        
        # OPTIMIZATION: Submit all chunks to decode executor immediately
        # Instead of polling for download_started, submit all chunks upfront.
        # Each worker will wait on its chunk's ready event, sleeping efficiently.
        # This removes polling overhead (~25ms per iteration) and lets the executor
        # manage scheduling. The _decode_sem already limits concurrent decodes.
        max_pool_workers = min(CURRENT_CPU_COUNT, len(chunks), _MAX_DECODE)
        
        total_chunks = len(chunks)
        completed = 0
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_pool_workers)
        active_futures = {}
        
        try:
            # Submit all chunks immediately (except permanently failed ones)
            for chunk in chunks:
                if chunk.permanent_failure:
                    bump('chunk_missing_count')
                    continue
                future = executor.submit(process_chunk, chunk)
                active_futures[future] = chunk
            
            log.debug(f"GET_IMG: Submitted {len(active_futures)} chunks to decode executor")
            
            # Collect results as they complete, respecting time budget
            while active_futures:
                # === TIME BUDGET CHECK ===
                if time_budget.exhausted:
                    remaining = len(active_futures)
                    if remaining > 0:
                        bump('chunk_budget_exhausted', remaining)
                    log.info(f"Time budget exhausted after {time_budget.elapsed:.2f}s for mipmap {mipmap}: "
                            f"processed {time_budget.chunks_processed}, skipped {time_budget.chunks_skipped}, "
                            f"remaining {remaining}/{total_chunks}")
                    break
                
                # Wait for completions with a short timeout to allow budget checks
                done, pending = concurrent.futures.wait(
                    active_futures.keys(), 
                    timeout=0.05,  # Check budget every 50ms
                    return_when=concurrent.futures.FIRST_COMPLETED
                )
                
                for future in done:
                    try:
                        chunk, chunk_img, start_x, start_y = future.result()
                        completed += 1
                        if chunk_img:
                            _safe_paste(new_im, chunk_img, start_x, start_y)
                    except Exception as exc:
                        log.error(f"Chunk processing failed: {exc}")
                    finally:
                        del active_futures[future]
                    # Progress logging
                    if total_chunks and (completed % max(1, total_chunks // 4) == 0):
                        log.debug(f"GET_IMG progress: {completed}/{total_chunks} chunks for mip {mipmap}")
            
            # Process any remaining active futures (when budget wasn't exhausted)
            # Budget exhaustion already logged above, so just drain remaining futures
            remaining_futures = list(active_futures.keys())
            if remaining_futures:
                # Use short timeout to avoid blocking forever
                timeout = max(0.5, time_budget.remaining) if not time_budget.exhausted else 0.5
                try:
                    for future in concurrent.futures.as_completed(remaining_futures, timeout=timeout):
                        try:
                            chunk, chunk_img, start_x, start_y = future.result()
                            if chunk_img:
                                _safe_paste(new_im, chunk_img, start_x, start_y)
                        except Exception as exc:
                            log.error(f"Chunk processing failed: {exc}")
                        finally:
                            if future in active_futures:
                                del active_futures[future]
                except TimeoutError:
                    unfinished = len([f for f in remaining_futures if not f.done()])
                    log.debug(f"Timeout waiting for {unfinished} remaining chunks")
                    bump('chunk_missing_count', unfinished)
                    # Cancel remaining futures to free resources
                    for future in remaining_futures:
                        if not future.done():
                            future.cancel()
                    # Clear remaining references
                    active_futures.clear()
        finally:
            # Use wait=False to avoid blocking on cancelled/timed-out futures
            # The futures will complete in the background but we won't wait for them
            # Since process_chunk checks time_budget.exhausted, they should exit quickly
            executor.shutdown(wait=False, cancel_futures=True)

        # Determine if we need to cache this image for fallback/upscaling
        should_cache = complete_img and mipmap <= self.max_mipmap
        
        if should_cache:
            log.debug(f"GET_IMG: Save complete image for later...")
            # Store image with metadata (col, row, zoom) for coordinate mapping in upscaling
            self.imgs[mipmap] = (new_im, col, row, zoom)

        log.debug(f"GET_IMG: DONE!  IMG created {new_im}")

        # OPTIMIZATION: In-place desaturation when image isn't cached
        # Only copy when image was saved to cache AND needs desaturation
        if seasons_enabled:
            saturation = 0.01 * season_saturation_locked(self.row, self.col, self.tilename_zoom)
            if saturation < 1.0:    # desaturation is expensive
                if should_cache:
                    # Must copy because original is cached for fallback use
                    new_im = new_im.copy().desaturate(saturation)
                else:
                    # In-place desaturation - assign result for error handling
                    new_im = new_im.desaturate(saturation)
        
        # Return image along with mipmap and zoom level this was created at
        return new_im

    def _try_lazy_build_fallback_mipmap(self, time_budget=None):
        """
        Lazy build a lower-detail mipmap when the first chunk failure occurs.
        
        This is triggered on-demand (not proactively like pre-building) to provide
        Fallback 2 support. When a mipmap 0 chunk fails, we quickly build mipmap 2
        (which has only 16 chunks at ZL14 vs 256 at ZL16), enabling subsequent
        failing chunks to upscale from the built image.
        
        Key differences from pre-building:
        - Only runs when a failure actually occurs (not on every tile)
        - Only runs once per tile (tracked by _lazy_build_attempted)
        - Uses reduced time budget to avoid blocking
        - Builds mipmap 2 (faster than mipmap 1, still useful for upscaling)
        
        Returns:
            True if a mipmap was successfully built, False otherwise
        """
        # Only attempt once per tile
        if self._lazy_build_attempted:
            return False
        
        self._lazy_build_attempted = True
        
        # Choose which mipmap to build:
        # - Mipmap 2 (ZL14): 16 chunks, fast to build, 4x upscale to mipmap 0
        # - Mipmap 3 (ZL13): 4 chunks, very fast, 8x upscale (lower quality)
        # We use mipmap 2 as a balance between speed and quality
        target_mipmap = min(2, self.max_mipmap)
        
        # Skip if target_mipmap is 0 - we're already building mipmap 0,
        # so building it again as a "fallback" is redundant
        if target_mipmap == 0:
            log.debug("Lazy build: max_mipmap is 0, no lower-detail mipmap available")
            return False
        
        # Skip if we somehow already have this mipmap
        if target_mipmap in self.imgs:
            log.debug(f"Lazy build: mipmap {target_mipmap} already exists")
            return True
        
        # Calculate available budget for lazy build
        # Use at most 1.5 seconds or 30% of remaining budget
        if time_budget and not time_budget.exhausted:
            max_lazy_budget = min(1.5, time_budget.remaining * 0.3)
            if max_lazy_budget < 0.3:
                log.debug(f"Lazy build: insufficient budget ({max_lazy_budget:.2f}s), skipping")
                return False
            lazy_budget = TimeBudget(max_lazy_budget)
        else:
            # No budget or exhausted - use a small fixed budget
            lazy_budget = TimeBudget(1.0)
        
        log.debug(f"Lazy build: triggering build of mipmap {target_mipmap} "
                 f"(budget={lazy_budget.max_seconds:.2f}s)")
        bump('lazy_build_triggered')
        
        try:
            # Build the lower mipmap
            # This will populate self.imgs[target_mipmap] for Fallback 2 to use
            result = self.get_img(
                target_mipmap, 
                startrow=0, 
                endrow=None, 
                maxwait=1.0,
                time_budget=lazy_budget
            )
            
            if result and target_mipmap in self.imgs:
                log.debug(f"Lazy build: successfully built mipmap {target_mipmap}")
                bump('lazy_build_success')
                return True
            else:
                log.debug(f"Lazy build: mipmap {target_mipmap} build returned but not in imgs")
                return False
                
        except Exception as e:
            log.debug(f"Lazy build: failed to build mipmap {target_mipmap}: {e}")
            bump('lazy_build_failed')
            return False

    def _get_shared_fallback_chunk(self, col, row, zoom):
        """
        Get or create a fallback chunk from the shared pool.
        
        This enables chunk sharing: when multiple high-zoom chunks fail and need
        the same parent chunk, they share a single download instead of each
        downloading it independently.
        
        Thread-safe: Uses a lock to prevent race conditions.
        
        Args:
            col, row, zoom: Coordinates of the fallback chunk to get/create
            
        Returns:
            Chunk object (may already be ready, in-flight, or newly created)
        """
        key = (col, row, zoom)
        
        with self._fallback_pool_lock:
            if key in self._fallback_chunk_pool:
                chunk = self._fallback_chunk_pool[key]
                log.debug(f"Reusing shared fallback chunk {chunk} from pool")
                bump('fallback_chunk_pool_hit')
                return chunk
            
            # Create new chunk and add to pool
            chunk = Chunk(col, row, self.maptype, zoom, cache_dir=self.cache_dir)
            self._fallback_chunk_pool[key] = chunk
            bump('fallback_chunk_pool_miss')
            
            # Check cache - if hit, it's ready immediately
            if chunk.ready.is_set():
                log.debug(f"Created shared fallback chunk {chunk} - already cached")
            else:
                log.debug(f"Created shared fallback chunk {chunk} - needs download")
            
            return chunk

    def get_or_build_lower_mipmap_chunk(self, target_mipmap, col, row, zoom, time_budget=None):
        """
        Cascading fallback: Try to get/build progressively lower-detail mipmaps.
        Only downloads chunks on-demand when needed (lazy evaluation).
        
        OPTIMIZED: Uses shared chunk pool to prevent duplicate downloads.
        When multiple chunks fail and need the same parent, they share one download.
        
        Args:
            target_mipmap: The mipmap level we need (e.g., 0)
            col, row, zoom: Chunk coordinates at target zoom
            time_budget: Optional TimeBudget to respect (for budget-aware network ops)
        
        Returns:
            Upscaled AoImage or None
        """
        # Early exit if budget is already exhausted
        if time_budget and time_budget.exhausted:
            log.debug(f"Cascading fallback: skipping - time budget exhausted")
            return None
        
        # Try each progressively lower-detail mipmap
        for fallback_mipmap in range(target_mipmap + 1, self.max_mipmap + 1):
            # Check budget at each iteration
            if time_budget and time_budget.exhausted:
                log.debug(f"Cascading fallback: stopping at mipmap {fallback_mipmap} - budget exhausted")
                break
            
            log.debug(f"Cascading fallback: trying mipmap {fallback_mipmap} for failed mipmap {target_mipmap}")
            
            # Calculate coordinates at fallback mipmap level
            mipmap_diff = fallback_mipmap - target_mipmap
            fallback_col = col >> mipmap_diff
            fallback_row = row >> mipmap_diff
            fallback_zoom = zoom - mipmap_diff
            
            # OPTIMIZED: Use shared pool to prevent duplicate downloads
            # Multiple failing chunks that need the same parent share one download
            fallback_chunk = self._get_shared_fallback_chunk(fallback_col, fallback_row, fallback_zoom)
            
            # If not ready, submit for download and wait
            if not fallback_chunk.ready.is_set():
                # Submit if not already in flight or queue
                if not fallback_chunk.in_flight and not fallback_chunk.in_queue:
                    chunk_getter.submit(fallback_chunk)
                
                # Wait for download with budget awareness
                if time_budget:
                    fallback_chunk_ready = time_budget.wait_with_budget(fallback_chunk.ready)
                else:
                    fallback_timeout = float(getattr(CFG.autoortho, 'fallback_timeout', 3.0))
                    fallback_chunk_ready = fallback_chunk.ready.wait(timeout=fallback_timeout)
                
                if not fallback_chunk_ready:
                    log.debug(f"Cascading fallback: mipmap {fallback_mipmap} timed out "
                             f"(budget_exhausted={time_budget.exhausted if time_budget else 'N/A'})")
                    # Don't close - shared pool manages lifecycle
                    continue
            
            # Chunk is ready - check if we have valid data
            if not fallback_chunk.data:
                log.debug(f"Cascading fallback: chunk {fallback_chunk} ready but no data")
                continue
            
            # Decode and upscale
            try:
                with _decode_sem:
                    fallback_img = AoImage.load_from_memory(fallback_chunk.data)
                    if fallback_img is None:
                        log.warning(f"Cascading fallback: load_from_memory returned None for {fallback_chunk}")
                        continue
                
                # Calculate which portion to extract and upscale
                scale_factor = 1 << mipmap_diff
                offset_col = col % scale_factor
                offset_row = row % scale_factor
                
                log.debug(f"CASCADE DEBUG: target=({col},{row}), fallback_chunk=({fallback_col},{fallback_row}), "
                         f"offset=({offset_col},{offset_row}), scale={scale_factor}")
                
                # Pixel position in fallback image
                pixel_x = offset_col * (256 // scale_factor)
                pixel_y = offset_row * (256 // scale_factor)
                crop_size = 256 // scale_factor
                
                # Upscale to 256x256
                upscaled = fallback_img.crop_and_upscale(
                    pixel_x, pixel_y, crop_size, crop_size, scale_factor
                )
                
                log.debug(f"Cascading fallback SUCCESS: upscaled mipmap {fallback_mipmap} -> {target_mipmap} "
                         f"at {col}x{row} (scale {scale_factor}x)")
                bump('upscaled_chunk_count')
                bump('chunk_from_cascade_fallback')
                
                # Don't close fallback_chunk - shared pool manages lifecycle
                # Other threads may still be using this chunk
                return upscaled
                
            except Exception as e:
                log.warning(f"Cascading fallback: failed to upscale from mipmap {fallback_mipmap}: {e}")
                continue
        
        log.debug(f"Cascading fallback: all mipmaps failed for {col}x{row} at mipmap {target_mipmap}")
        return None

    def get_downscaled_from_higher_mipmap(self, target_mipmap, col, row, zoom):
        """
        Try to scale from already-built mipmaps to fill missing chunk.
        Checks both downscaling (from higher-detail) and upscaling (from lower-detail).
        
        Args:
            target_mipmap: The mipmap level we need (e.g., 0 or 3)
            col, row, zoom: Chunk coordinates at target zoom
        
        Returns:
            Scaled AoImage or None
        """
        for higher_mipmap in range(target_mipmap):
            if higher_mipmap not in self.imgs:
                continue  # Haven't built this mipmap yet
            
            img_data = self.imgs[higher_mipmap]
            if not img_data:
                continue
            
            # Unpack metadata (or use old format for backward compat)
            if isinstance(img_data, tuple):
                higher_img = img_data[0]  # Extract image from tuple
            else:
                higher_img = img_data  # Old format: just the image
            
            if higher_img is None:
                continue
            
            # Calculate scale factor
            scale_factor = 1 << (target_mipmap - higher_mipmap)
            
            chunk_offset_x = (col % scale_factor) * 256
            chunk_offset_y = (row % scale_factor) * 256
            
            try:
                # Crop scale_factor*256 region, then downscale to 256
                crop_size = 256 * scale_factor
                cropped = AoImage.new('RGBA', (crop_size, crop_size), (0,0,0,0))
                
                # PHASE 2 FIX #6 & #8: Validate crop coordinates
                if chunk_offset_x < 0 or chunk_offset_y < 0:
                    log.warning(f"GET_IMG: Negative crop offset: ({chunk_offset_x},{chunk_offset_y}), skipping fallback")
                    continue
                
                # Check bounds
                higher_width, higher_height = higher_img.size
                if chunk_offset_x + crop_size > higher_width or chunk_offset_y + crop_size > higher_height:
                    log.warning(f"GET_IMG: Crop extends beyond image: pos=({chunk_offset_x},{chunk_offset_y}), size={crop_size}, image=({higher_width}x{higher_height})")
                    continue
                
                if not higher_img.crop(cropped, (chunk_offset_x, chunk_offset_y)):
                    log.warning(f"GET_IMG: crop() failed for fallback at ({chunk_offset_x},{chunk_offset_y})")
                    continue
                
                downscale_steps = int(math.log2(scale_factor))
                downscaled = cropped.reduce_2(downscale_steps)
                
                log.debug(f"Downscaled mipmap {higher_mipmap} to fill missing mipmap {target_mipmap} chunk at {col}x{row}")
                bump('downscaled_chunk_count')
                return downscaled
            except Exception as e:
                log.debug(f"Failed to downscale from mipmap {higher_mipmap}: {e}")
                continue
        
        # Search for lower-detail mipmaps (higher mipmap numbers) to upscale
        for lower_mipmap in range(target_mipmap + 1, self.max_mipmap + 1):
            if lower_mipmap not in self.imgs:
                continue
            
            img_data = self.imgs[lower_mipmap]
            if not img_data:
                continue
            
            # Unpack metadata
            if isinstance(img_data, tuple):
                lower_img, base_col, base_row, base_zoom = img_data
            else:
                continue  # Old format without metadata, skip
            
            # Calculate scale factor and relative position
            scale_factor = 1 << (lower_mipmap - target_mipmap)
            
            # Map requested chunk to position in lower-detail image
            # lower_img was built from chunks starting at (base_col, base_row) at base_zoom
            # We need chunk (col, row) at zoom
            
            # Convert requested chunk coords to the zoom level of the lower image
            zoom_diff = zoom - base_zoom
            if zoom_diff >= 0:
                # Requested chunk is at higher or same zoom as image base
                # Calculate which parent chunk at base_zoom level
                parent_col = col >> zoom_diff
                parent_row = row >> zoom_diff
                
                # Relative to base position (which parent chunk in the image)
                rel_col = parent_col - base_col
                rel_row = parent_row - base_row
                
                # Calculate sub-chunk offset within that parent (0 to 2^zoom_diff-1)
                sub_col = col & ((1 << zoom_diff) - 1)  # Equivalent to col % (1 << zoom_diff)
                sub_row = row & ((1 << zoom_diff) - 1)
            else:
                # Requested chunk is at lower zoom (shouldn't happen but handle it)
                parent_col = col << (-zoom_diff)
                parent_row = row << (-zoom_diff)
                rel_col = parent_col - base_col
                rel_row = parent_row - base_row
                sub_col = 0
                sub_row = 0
            
            # Size to crop from lower image (will be upscaled by scale_factor)
            crop_size = 256 // scale_factor
            
            # Each parent chunk in the image is 256px wide
            # We need to find the sub-chunk within that 256x256 parent
            # Each sub-chunk occupies (256 // scale_factor) pixels in the parent
            sub_chunk_size_in_parent = crop_size  # Same as 256 // scale_factor
            pixel_x = rel_col * 256 + sub_col * sub_chunk_size_in_parent
            pixel_y = rel_row * 256 + sub_row * sub_chunk_size_in_parent
            
            log.debug(f"MIPMAP UPSCALE DEBUG: target=({col},{row},z{zoom}) base=({base_col},{base_row},z{base_zoom}) parent=({parent_col},{parent_row}) sub=({sub_col},{sub_row}) pixel=({pixel_x},{pixel_y}) crop={crop_size}")
            
            # Bounds check
            img_width, img_height = lower_img.size
            if (pixel_x < 0 or pixel_y < 0 or 
                pixel_x + crop_size > img_width or 
                pixel_y + crop_size > img_height):
                log.debug(f"Upscale bounds check failed: pos ({pixel_x},{pixel_y}) crop {crop_size} img ({img_width}x{img_height})")
                continue
            
            try:
                upscaled = lower_img.crop_and_upscale(
                    pixel_x, pixel_y, crop_size, crop_size, scale_factor
                )
                log.debug(f"Upscaled mipmap {lower_mipmap} (zoom {base_zoom}) to fill mipmap {target_mipmap} chunk at {col}x{row}")
                bump('upscaled_chunk_count')
                return upscaled
            except Exception as e:
                log.debug(f"Failed to upscale from mipmap {lower_mipmap}: {e}")
                continue
        
        return None

    def get_best_chunk(self, col, row, mm, zoom):
        """
        Search disk cache for lower-zoom JPEG chunks and upscale to fill missing chunk.
        
        OPTIMIZED: Uses direct filesystem checks instead of creating Chunk objects.
        This avoids the overhead of creating threading primitives (Event objects)
        for each cache lookup. Only reads file data when cache hit is confirmed.
        
        Args:
            col, row: Chunk coordinates at target zoom
            mm: Target mipmap level
            zoom: Target zoom level
            
        Returns:
            Upscaled AoImage or None if no cached fallback found
        """
        max_search_zoom = self.max_zoom

        for i in range(mm + 1, self.max_mipmap + 1):
            # Difference between requested mm and found image mm level
            diff = i - mm
            
            # Equivalent col, row, zl at lower zoom
            col_p = col >> diff
            row_p = row >> diff
            zoom_p = zoom - i
            
            # Don't search beyond our detected actual_max_zoom
            if zoom_p > max_search_zoom:
                continue

            scalefactor = min(1 << diff, 16)

            # OPTIMIZED: Direct cache path check without Chunk object creation
            # This avoids creating threading.Event objects for each lookup
            chunk_id = f"{col_p}_{row_p}_{zoom_p}_{self.maptype}"
            cache_path = os.path.join(self.cache_dir, f"{chunk_id}.jpg")
            
            # Fast existence check - avoids Chunk object overhead
            if not os.path.isfile(cache_path):
                bump('chunk_miss')
                continue
            
            # Cache file exists - read and validate
            bump('chunk_hit')
            log.debug(f"Found cache file for {chunk_id}, reading...")
            
            # Read file data with retry for Windows file locking
            data = None
            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                try:
                    data = Path(cache_path).read_bytes()
                    break
                except PermissionError:
                    if attempt < max_attempts:
                        time.sleep(0.02 * attempt)
                        continue
                    log.debug(f"Permission denied reading cache {cache_path}")
                    break
                except FileNotFoundError:
                    # File was deleted between check and read (race condition)
                    break
                except OSError as e:
                    winerr = getattr(e, 'winerror', None)
                    if winerr in (5, 32, 33) and attempt < max_attempts:
                        time.sleep(0.02 * attempt)
                        continue
                    log.debug(f"OSError reading cache {cache_path}: {e}")
                    break
            
            if not data:
                continue
            
            # Validate JPEG header
            if not _is_jpeg(data[:3]):
                log.debug(f"Cache file {cache_path} not a valid JPEG")
                continue
            
            # Touch the file to update mtime for LRU cache management
            try:
                os.utime(cache_path, None)
            except (FileNotFoundError, PermissionError):
                pass
        
            log.debug(f"Found cached JPEG for {col}x{row}x{zoom} (mm{mm}) at {col_p}x{row_p}x{zoom_p} (mm{i}), upscaling {scalefactor}x")
            
            # Offset into chunk
            col_offset = col % scalefactor
            row_offset = row % scalefactor

            log.debug(f"UPSCALE DEBUG: col={col}, row={row}, col_p={col_p}, row_p={row_p}, col_offset={col_offset}, row_offset={row_offset}, scalefactor={scalefactor}")

            # Pixel dimensions to extract from source
            w_p = max(1, 256 >> diff)
            h_p = max(1, 256 >> diff)

            log.debug(f"Pixel Size: {w_p}x{h_p}")

            # Load image to crop
            try:
                img_p = AoImage.load_from_memory(data)
            except Exception as e:
                log.error(f"Exception loading cached JPEG {chunk_id} into memory: {e}")
                continue
            
            if not img_p:
                log.warning(f"Failed to load cached JPEG {chunk_id} into memory (returned None).")
                continue

            # Crop the relevant portion
            crop_img = AoImage.new('RGBA', (w_p, h_p), (0, 255, 0))
            img_p.crop(crop_img, (col_offset * w_p, row_offset * h_p))
            chunk_img = crop_img.scale(scalefactor)

            # Free source image memory
            try:
                img_p.close()
            except Exception:
                pass
            
            # Track upscaling from cached JPEGs
            bump('upscaled_from_jpeg_count')
            return chunk_img

        log.debug(f"No best chunk found for {col}x{row}x{zoom}!")
        return None

    def get_maxwait(self):
        effective_maxwait = self.maxchunk_wait
        if CFG.autoortho.suspend_maxwait and not datareftracker.connected:
            effective_maxwait = 20
        return effective_maxwait

    def get_fallback_level(self):
        """Get the fallback level as an integer.
        
        Converts string config values to integer:
        - 'none' -> 0 (skip all fallbacks)
        - 'cache' -> 1 (disk cache and pre-built mipmaps only)
        - 'full' -> 2 (all fallbacks including network downloads)
        
        For backwards compatibility, also accepts integer strings '0', '1', '2'
        and boolean True/False (legacy config parsing artifacts).
        """
        fb_value = getattr(CFG.autoortho, 'fallback_level', 'cache')
        
        # Handle string values
        if isinstance(fb_value, str):
            fb_lower = fb_value.lower().strip()
            if fb_lower == 'none':
                return 0
            elif fb_lower == 'cache':
                return 1
            elif fb_lower == 'full':
                return 2
            else:
                # Try parsing as integer for backwards compatibility
                try:
                    # Clamp to valid range 0-2
                    return max(0, min(2, int(fb_value)))
                except ValueError:
                    return 1  # Default to cache
        # Handle boolean (from SectionParser when value was '0' or '1')
        elif isinstance(fb_value, bool):
            return 2 if fb_value else 0
        # Handle integer
        elif isinstance(fb_value, int):
            return max(0, min(2, fb_value))
        else:
            return 1  # Default to cache

    def get_fallback_extends_budget(self):
        """Check if fallbacks should extend beyond the time budget.
        
        When True and fallback_level is 'full', network fallbacks will continue
        even after the tile time budget is exhausted (prioritizing quality over timing).
        
        When False, fallbacks respect the time budget strictly (prioritizing timing over quality).
        
        Returns:
            bool: True if fallbacks should ignore budget exhaustion
        """
        fb_value = getattr(CFG.autoortho, 'fallback_extends_budget', False)
        
        # Handle string values from config
        if isinstance(fb_value, str):
            return fb_value.lower().strip() in ('true', '1', 'yes', 'on')
        # Handle boolean directly
        elif isinstance(fb_value, bool):
            return fb_value
        else:
            return False  # Default: respect budget

    #@profile
    @locked
    def get_mipmap(self, mipmap=0):
        #
        # Protect this method to avoid simultaneous threads attempting mm builds at the same time.
        # Otherwise we risk contention such as waiting get_img call attempting to build an image as 
        # another thread closes chunks.
        #
        
        # Start timing FULL tile creation (download + compose + compress)
        # Use monotonic() for consistent interval measurement (immune to clock adjustments)
        tile_creation_start = time.monotonic()

        log.debug(f"GET_MIPMAP: {self}")

        if mipmap > self.max_mipmap:
            mipmap = self.max_mipmap
        
        # === BUDGET TIMING ===
        # The tile budget starts when get_img() is first called, NOT when X-Plane first
        # requests the tile. This means queue wait time doesn't count against the budget.
        # Only actual processing time (chunk downloads, composition, compression) counts.
        #
        # NOTE: We intentionally do NOT skip get_img/gen_mipmaps when budget is exhausted.
        # Even if the budget is exhausted, we must still:
        # 1. Call get_img() to create an image filled with missing_color
        # 2. Call gen_mipmaps() to compress it into the DDS buffer
        # This ensures X-Plane receives a valid DDS with missing_color tiles instead of
        # black/uninitialized data. Once X-Plane reads a DDS, it's cached and won't refresh.
        # The get_img() method handles budget exhaustion gracefully by skipping chunk downloads
        # but still returning a valid (missing_color filled) image for compression.
        
        if self._tile_time_budget and self._tile_time_budget.exhausted:
            log.debug(f"GET_MIPMAP: Tile budget exhausted, will build mipmap {mipmap} with missing_color (no new downloads)")

        # We can have multiple threads wait on get_img ...
        log.debug(f"GET_MIPMAP: Next call is get_img which may block!.............")
        # Pass the tile-level budget to get_img so it's shared across all mipmap builds
        new_im = self.get_img(mipmap, maxwait=self.get_maxwait(), time_budget=self._tile_time_budget)
        if not new_im:
            log.debug("GET_MIPMAP: No updates, so no image generated")
            return True

        self.ready.clear()
        compress_start_time = time.monotonic()
        try:
            if mipmap == 0:
                self.dds.gen_mipmaps(new_im, mipmap, 0) 
            else:
                self.dds.gen_mipmaps(new_im, mipmap) 
        finally:
            pass
            #new_im.close()

        compress_end_time = time.monotonic()
        self.ready.set()

        # Calculate timing metrics
        zoom = self.max_zoom - mipmap
        compress_time = compress_end_time - compress_start_time
        total_creation_time = compress_end_time - tile_creation_start
        
        # Track compression time (legacy stat)
        mm_stats.set(mipmap, compress_time)
        
        # Track FULL tile creation time (new stat for tuning tile_time_budget)
        tile_creation_stats.set(mipmap, total_creation_time)

        # Record per-mipmap stats via counters for aggregation
        try:
            bump_many({
                f"mm_count:{mipmap}": 1,
                f"mm_compress_time_ms:{mipmap}": int(compress_time * 1000),
                f"tile_create_time_ms:{mipmap}": int(total_creation_time * 1000),
            })
        except Exception:
            pass
        
        # Only report tile completion stats when mipmap 0 is done (full tile delivered to X-Plane)
        # This tracks the same time window as TimeBudget: from first request to tile release
        if mipmap == 0 and not self._completion_reported:
            self._completion_reported = True
            # Calculate time from first X-Plane request to completion
            if self.first_request_time is not None:
                tile_completion_time = time.monotonic() - self.first_request_time
            else:
                # Fallback: use the mipmap creation time if first_request_time wasn't set
                tile_completion_time = total_creation_time
            
            try:
                bump_many({
                    "tile_create_count": 1,
                    "tile_create_time_total_ms": int(tile_completion_time * 1000),
                })
            except Exception:
                pass
            
            log.debug(f"GET_MIPMAP: Tile {self} COMPLETED in {tile_completion_time:.2f}s "
                     f"(mipmap 0 done, time from first request)")
        
        # Log per-mipmap creation time for visibility
        log.debug(f"GET_MIPMAP: Tile {self} mipmap {mipmap} created in {total_creation_time:.2f}s "
                 f"(download+compose: {total_creation_time - compress_time:.2f}s, compress: {compress_time:.2f}s)")

        # Don't close all chunks since we don't gen all mipmaps 
        if mipmap == 0:
            log.debug("GET_MIPMAP: Will close all chunks.")
            for z,chunks in self.chunks.items():
                for chunk in chunks:
                    chunk.close()
            self.chunks = {}
                    #del(chunk.data)
                    #del(chunk.img)
        #return outfile
        log.debug("Results:")
        log.debug(self.dds.mipmap_list)
        return True


    def should_close(self):
        if self.dds.mipmap_list[0].retrieved:
            if self.bytes_read < self.dds.mipmap_list[0].length:
                log.warning(f"TILE: {self} retrieved mipmap 0, but only read {self.bytes_read}. Lowest offset: {self.lowest_offset}")
                return False
            else:
                #log.info(f"TILE: {self} retrieved mipmap 0, full read of mipmap! {self.bytes_read}.")
                return True
        else:
            return True


    def close(self):
        log.debug(f"Closing {self}")

        # Check refs first - if still referenced, don't close yet
        if self.refs > 0:
            log.warning(f"TILE: Trying to close, but has refs: {self.refs}")
            return

        # Log mipmap retrieval status (safely check dds first)
        if self.dds is not None:
            try:
                if self.dds.mipmap_list and self.dds.mipmap_list[0].retrieved:
                    if self.bytes_read < self.dds.mipmap_list[0].length:
                        log.warning(f"TILE: {self} retrieved mipmap 0, but only read {self.bytes_read}. Lowest offset: {self.lowest_offset}")
                    else:
                        log.debug(f"TILE: {self} retrieved mipmap 0, full read of mipmap! {self.bytes_read}.")
            except (AttributeError, IndexError):
                pass  # DDS structure incomplete, continue with cleanup

        # ------------------------------------------------------------------
        # Memory-reclamation additions
        # ------------------------------------------------------------------

        # 1) Free any cached AoImage instances (RGBA pixel buffers)
        try:
            for img_data in list(self.imgs.values()):
                # Handle both tuple format (new) and plain image (old)
                if isinstance(img_data, tuple):
                    im = img_data[0]  # Extract image from tuple
                else:
                    im = img_data
                
                if im is not None and hasattr(im, "close"):
                    try:
                        im.close()
                    except Exception:
                        pass
        except Exception:
            pass
        finally:
            self.imgs.clear()

        # 2) Release DDS mip-map ByteIO buffers so the underlying bytes
        #    are no longer referenced from Python.
        if self.dds is not None:
            try:
                for mm in getattr(self.dds, "mipmap_list", []):
                    mm.databuffer = None
            except Exception:
                pass
            # Drop the DDS object reference itself
            self.dds = None

        # 3) Close all chunks
        try:
            for chunks in self.chunks.values():
                for chunk in chunks:
                    try:
                        chunk.close()
                    except Exception:
                        pass
        except Exception:
            pass
        self.chunks = {}
        
        # 4) Close all fallback chunks in the shared pool
        try:
            with self._fallback_pool_lock:
                for chunk in self._fallback_chunk_pool.values():
                    try:
                        chunk.close()
                    except Exception:
                        pass
                self._fallback_chunk_pool.clear()
        except Exception:
            pass


class TileCacher(object):
    hits = 0
    misses = 0

    enable_cache = True
    cache_mem_lim = pow(2,30) * float(CFG.cache.cache_mem_limit)
    cache_tile_lim = 25

    def __init__(self, cache_dir='.cache'):
        if MEMTRACE:
            tracemalloc.start()

        self.tiles = OrderedDict()
        self.open_count = {}

        self.maptype_override = CFG.autoortho.maptype_override
        if self.maptype_override:
            log.info(f"Maptype override set to {self.maptype_override}")
            if self.maptype_override == "APPLE":
                apple_token_service.reset_apple_maps_token()
        else:
            log.info(f"Maptype override not set, will use default.")
        log.info(f"Will use Compressor: {CFG.pydds.compressor}")
        self.tc_lock = threading.RLock()
        self._pid = os.getpid()
        # Eviction behavior controls
        self.evict_hysteresis_frac = 0.10  # keep ~10% headroom below limit
        self.evict_headroom_min_bytes = 256 * 1048576  # at least 256MB headroom
        self.evict_leader_ttl_sec = 5  # seconds
        
        self.cache_dir = CFG.paths.cache_dir
        log.info(f"Cache dir: {self.cache_dir}")
        self.min_zoom = int(CFG.autoortho.min_zoom)
        # Set target zoom level directly - much simpler than offset calculations
        self.target_zoom_level = int(CFG.autoortho.max_zoom)  # Direct zoom level target, regardless of tile name
        self.target_zoom_level_near_airports = int(CFG.autoortho.max_zoom_near_airports)
        log.info(f"Target zoom level set to ZL{self.target_zoom_level}")

        # Dynamic zoom configuration
        # When "dynamic", zoom level is computed based on predicted altitude at tile
        # When "fixed" (default), uses the configured max_zoom value
        self.max_zoom_mode = str(CFG.autoortho.max_zoom_mode).lower()
        self.dynamic_zoom_manager = DynamicZoomManager()
        if self.max_zoom_mode == "dynamic":
            self.dynamic_zoom_manager.load_from_config(
                CFG.autoortho.dynamic_zoom_steps
            )
            step_count = len(self.dynamic_zoom_manager.get_steps())
            log.info(f"Dynamic zoom enabled with {step_count} quality step(s)")
            if step_count > 0:
                log.info(f"Dynamic zoom steps: {self.dynamic_zoom_manager.get_summary()}")
        else:
            log.info("Using fixed max zoom level")

        self.clean_t = threading.Thread(target=self.clean, daemon=True)
        self.clean_t.start()

        if system_type == 'windows':
            # Windows doesn't handle FS cache the same way so enable here.
            self.enable_cache = True
            self.cache_tile_lim = 50
    
    def _compute_dynamic_zoom(self, row: int, col: int, tile_zoom: int) -> int:
        """
        Compute dynamic zoom level based on predicted altitude at tile.

        Uses the aircraft's averaged flight data (heading, speed, vertical speed)
        to predict what altitude the aircraft will be at when it reaches the
        closest point to this tile. Then returns the appropriate zoom level
        from the configured quality steps.
        
        If SimBrief flight data is loaded and the "use_flight_data" toggle is enabled,
        and the aircraft is on-route, uses the flight plan altitude for that position
        instead of DataRef-based prediction.

        Args:
            row: Tile row coordinate
            col: Tile column coordinate
            tile_zoom: Default zoom level for this tile

        Returns:
            The computed max zoom level for this tile
        """
        # Get tile center coordinates (needed for both methods)
        tile_lat, tile_lon = _chunk_to_latlon(row, col, tile_zoom)
        
        # Check if SimBrief flight data should be used
        simbrief_altitude_agl = self._get_simbrief_altitude_for_tile(tile_lat, tile_lon)
        if simbrief_altitude_agl is not None:
            # Use SimBrief flight plan AGL altitude for zoom calculation
            log.debug(f"Using SimBrief AGL altitude {simbrief_altitude_agl}ft for tile at {tile_lat:.2f},{tile_lon:.2f}")
            if tile_zoom == 18 and not CFG.autoortho.using_custom_tiles:
                return self.dynamic_zoom_manager.get_airport_zoom_for_altitude(simbrief_altitude_agl)
            return self.dynamic_zoom_manager.get_zoom_for_altitude(simbrief_altitude_agl)
        
        # Fall back to DataRef-based calculation
        return self._compute_dynamic_zoom_from_datarefs(row, col, tile_zoom, tile_lat, tile_lon)
    
    def _get_simbrief_altitude_for_tile(self, tile_lat: float, tile_lon: float) -> Optional[int]:
        """
        Get conservative AGL altitude from SimBrief flight plan for a tile position.
        
        Returns the planned altitude Above Ground Level (AGL) if:
        - SimBrief flight data is loaded
        - The "use_flight_data" toggle is enabled
        - The aircraft is currently on-route (within deviation threshold)
        
        When multiple waypoints are within the consideration radius:
        - Uses the LOWEST flight altitude (MSL) - accounts for descent
        - Uses the HIGHEST ground elevation - accounts for mountains
        - Conservative AGL = lowest_MSL - highest_ground
        
        This ensures maximum detail when flying over areas with varied terrain
        (e.g., descending over mountains).
        
        AGL Calculation:
            AGL = MSL altitude - terrain elevation (ground_height from SimBrief)
            
            AGL is used because it represents the actual height above the terrain
            being viewed. This is more relevant for imagery quality:
            - 10,000 ft MSL over 5,000 ft mountains = 5,000 ft AGL (needs higher zoom)
            - 10,000 ft MSL over the ocean = 10,000 ft AGL (can use lower zoom)
        
        Args:
            tile_lat: Tile center latitude
            tile_lon: Tile center longitude
            
        Returns:
            Conservative AGL altitude in feet if SimBrief data should be used, None otherwise
        """
        # Check if SimBrief integration is enabled
        if not hasattr(CFG, 'simbrief'):
            return None
        
        use_flight_data = getattr(CFG.simbrief, 'use_flight_data', False)
        if isinstance(use_flight_data, str):
            use_flight_data = use_flight_data.lower() in ('true', '1', 'yes', 'on')
        
        if not use_flight_data:
            return None
        
        # Check if flight data is loaded
        if not simbrief_flight_manager.is_loaded:
            return None
        
        # Get aircraft position to check if on-route
        with datareftracker._lock:
            if not datareftracker.data_valid:
                return None
            aircraft_lat = datareftracker.lat
            aircraft_lon = datareftracker.lon
        
        # Get deviation threshold from config
        deviation_threshold = float(getattr(CFG.simbrief, 'route_deviation_threshold_nm', 40))
        
        # Check if aircraft is on-route
        if not simbrief_flight_manager.is_on_route(aircraft_lat, aircraft_lon, deviation_threshold):
            log.debug(f"Aircraft deviated from route, using DataRef-based zoom calculation")
            return None
        
        # Get consideration radius from config
        consideration_radius = float(getattr(CFG.simbrief, 'route_consideration_radius_nm', 50))
        
        # Get AGL altitude at tile position (uses lowest AGL of fixes within radius)
        # use_agl=True returns Above Ground Level altitude for better terrain awareness
        altitude_agl = simbrief_flight_manager.get_altitude_at_position(
            tile_lat, tile_lon, consideration_radius, use_agl=True
        )
        
        return altitude_agl
    
    def _compute_dynamic_zoom_from_datarefs(self, row: int, col: int, tile_zoom: int,
                                             tile_lat: float, tile_lon: float) -> int:
        """
        Compute dynamic zoom level using DataRef-based altitude prediction.
        
        This is the fallback method when SimBrief data is not available or
        when the aircraft has deviated from the planned route.
        """
        # Get flight averages for prediction
        averages = datareftracker.get_flight_averages()

        # If no valid averages, try to use current altitude (AGL)
        if averages is None:
            with datareftracker._lock:
                if datareftracker.data_valid and datareftracker.alt_agl_ft > 0:
                    return self.dynamic_zoom_manager.get_zoom_for_altitude(
                        datareftracker.alt_agl_ft
                    )
            # Fall back to base step or fixed zoom
            base = self.dynamic_zoom_manager.get_base_step()
            return base.zoom_level if base else self.target_zoom_level

        # Get current position (with lock for thread safety)
        with datareftracker._lock:
            if not datareftracker.data_valid:
                base = self.dynamic_zoom_manager.get_base_step()
                return base.zoom_level if base else self.target_zoom_level

            aircraft_lat = datareftracker.lat
            aircraft_lon = datareftracker.lon
            aircraft_alt_ft = datareftracker.alt_agl_ft

        # Predict altitude at closest approach (using AGL for terrain-aware calculations)
        predicted_alt, will_approach = predict_altitude_at_closest_approach(
            aircraft_lat=aircraft_lat,
            aircraft_lon=aircraft_lon,
            aircraft_alt_ft=aircraft_alt_ft,
            aircraft_hdg=averages['heading'],
            aircraft_speed_mps=averages['ground_speed_mps'],
            vertical_speed_fpm=averages['vertical_speed_fpm'],
            tile_lat=tile_lat,
            tile_lon=tile_lon
        )

        # Get zoom level for predicted altitude
        # Use airport zoom level when near airports (tile_zoom == 18)
        if tile_zoom == 18 and not CFG.autoortho.using_custom_tiles:
            return self.dynamic_zoom_manager.get_airport_zoom_for_altitude(predicted_alt)
        return self.dynamic_zoom_manager.get_zoom_for_altitude(predicted_alt)

    def _get_target_zoom_level(self, default_zoom: int, row: int = None, col: int = None) -> int:
        """
        Get target zoom level for a tile.

        In fixed mode: Uses the configured max_zoom (current behavior)
        In dynamic mode: Computes zoom based on predicted altitude at tile

        Args:
            default_zoom: The default/base zoom level for this tile
            row: Tile row coordinate (required for dynamic mode)
            col: Tile column coordinate (required for dynamic mode)

        Returns:
            The target zoom level, capped to tile's max supported zoom
        """
        # Dynamic mode - compute based on altitude prediction
        if self.max_zoom_mode == "dynamic" and row is not None and col is not None:
            dynamic_zoom = self._compute_dynamic_zoom(row, col, default_zoom)
            # Still cap to tile's max supported zoom (default + 1 is X-Plane's limit)
            return min(default_zoom + 1, dynamic_zoom)

        # Fixed mode - existing behavior
        if CFG.autoortho.using_custom_tiles:
            uncapped_target_zoom = self.target_zoom_level
        else:
            uncapped_target_zoom = (
                self.target_zoom_level_near_airports
                if default_zoom == 18
                else self.target_zoom_level
            )
        return min(default_zoom + 1, uncapped_target_zoom)

    def _to_tile_id(self, row, col, map_type, zoom):
        if self.maptype_override:
            map_type = self.maptype_override
        tile_id = f"{row}_{col}_{map_type}_{zoom}"
        return tile_id

    def show_stats(self):
        process = psutil.Process(os.getpid())
        cur_mem = process.memory_info().rss
        # Report per-process memory to shared store; parent will aggregate
        update_process_memory_stat()
        #set_stat('tile_mem_open', len(self.tiles))
        if self.enable_cache:
            #set_stat('tile_mem_miss', self.misses)
            #set_stat('tile_mem_hits', self.hits)
            log.debug(f"TILE CACHE:  MISS: {self.misses}  HIT: {self.hits}")
        log.debug(f"NUM OPEN TILES: {len(self.tiles)}.  TOTAL MEM: {cur_mem//1048576} MB")

    # -----------------------------
    # LRU helpers and leader logic
    # -----------------------------
    def _touch_tile(self, idx, tile):
        try:
            # Move to MRU position
            self.tiles.move_to_end(idx, last=True)
        except Exception:
            pass

    def _lru_candidates(self):
        try:
            # Keys iterate from LRU -> MRU after move_to_end
            return list(self.tiles.keys())
        except Exception:
            return list(self.tiles.keys())

    def _has_shared_store(self) -> bool:
        return bool(getattr(STATS, "_remote", None) or os.getenv("AO_STATS_ADDR"))

    def _try_acquire_evict_leader(self) -> bool:
        if not self._has_shared_store():
            return True
        now = int(time.time())
        try:
            leader_until = get_stat('evict_leader_until') or 0
            leader_pid = get_stat('evict_leader_pid') or 0
        except Exception:
            leader_until = 0
            leader_pid = 0

        if int(leader_until) < now:
            # Try to become leader
            try:
                set_stat('evict_leader_pid', self._pid)
                set_stat('evict_leader_until', now + self.evict_leader_ttl_sec)
                return True
            except Exception:
                return False

        # Renew if we are already leader
        if int(leader_pid) == self._pid:
            try:
                set_stat('evict_leader_until', now + self.evict_leader_ttl_sec)
                return True
            except Exception:
                return False

        return False

    def _renew_evict_leader(self) -> None:
        if not self._has_shared_store():
            return
        try:
            now = int(time.time())
            if int(get_stat('evict_leader_pid') or 0) == self._pid:
                set_stat('evict_leader_until', now + self.evict_leader_ttl_sec)
        except Exception:
            pass

    def _evict_batch(self, max_to_evict: int) -> int:
        evicted = 0
        # Prefer strict LRU order: left to right
        for idx in list(self._lru_candidates()):
            if evicted >= max_to_evict:
                break
            t = self.tiles.get(idx)
            if not t:
                continue
            if t.refs > 0:
                continue
            # Evict this tile
            try:
                t = self.tiles.pop(idx)
            except KeyError:
                continue
            try:
                t.close()
            except Exception:
                pass
            finally:
                t = None
                evicted += 1
        return evicted

    def clean(self):
        log.info(f"Started tile clean thread.  Mem limit {self.cache_mem_lim}")
        # Faster cadence when a shared stats store is present (macOS parent)
        fast_mode = self._has_shared_store()
        poll_interval = 3 if fast_mode else 15
        
        # Maximum tile count before forced eviction (prevents memory bloat from tile object overhead)
        # Each tile object costs ~10-50KB in overhead even without loaded data
        max_tile_count = 5000

        while True:
            process = psutil.Process(os.getpid())
            cur_mem = process.memory_info().rss

            # Publish this process heartbeat + RSS so the parent can aggregate
            try:
                update_process_memory_stat()
            except Exception:
                pass

            self.show_stats()

            if not self.enable_cache:
                time.sleep(poll_interval)
                continue

            # Use aggregated memory across all workers when available; otherwise local RSS
            try:
                global_mem_mb = get_stat('cur_mem_mb')
                global_mem_bytes = int(global_mem_mb) * 1048576 if isinstance(global_mem_mb, (int, float)) else 0
            except Exception:
                global_mem_bytes = 0

            effective_mem = global_mem_bytes or cur_mem

            # Hysteresis target: evict down to limit - headroom
            headroom = max(int(self.cache_mem_lim * self.evict_hysteresis_frac), self.evict_headroom_min_bytes)
            target_bytes = max(0, int(self.cache_mem_lim) - headroom)
            
            # Check if we need to evict based on tile count (prevents unbounded tile accumulation)
            tile_count = len(self.tiles)
            need_tile_count_eviction = tile_count > max_tile_count

            # Leader election: only one worker performs eviction when shared store is present
            need_mem_eviction = effective_mem > self.cache_mem_lim
            if need_mem_eviction or need_tile_count_eviction:
                if not self._try_acquire_evict_leader():
                    time.sleep(poll_interval)
                    continue
                    
            # Evict if too many tiles (regardless of memory)
            if need_tile_count_eviction:
                target_tile_count = int(max_tile_count * 0.8)  # Evict down to 80% of max
                tiles_to_evict = tile_count - target_tile_count
                if tiles_to_evict > 0:
                    log.info(f"Tile count eviction: {tile_count} tiles, evicting {tiles_to_evict} to reach {target_tile_count}")
                    with self.tc_lock:
                        self._evict_batch(tiles_to_evict)

            # Evict while above target using adaptive batch sizing
            while self.tiles and effective_mem > target_bytes:
                over_bytes = max(0, effective_mem - target_bytes)
                ratio = min(1.0, over_bytes / max(1, self.cache_mem_lim))
                adaptive = max(20, int(len(self.tiles) * min(0.10, ratio)))
                with self.tc_lock:
                    evicted = self._evict_batch(adaptive)
                if evicted == 0:
                    break

                # Recompute local RSS and, if available, the aggregated RSS
                cur_mem = process.memory_info().rss
                try:
                    global_mem_mb = get_stat('cur_mem_mb')
                    global_mem_bytes = int(global_mem_mb) * 1048576 if isinstance(global_mem_mb, (int, float)) else 0
                except Exception:
                    global_mem_bytes = 0
                effective_mem = global_mem_bytes or cur_mem

                # Renew leadership and publish heartbeat after an eviction batch
                try:
                    self._renew_evict_leader()
                    update_process_memory_stat()
                except Exception:
                    pass

            if MEMTRACE:
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')

                log.info("[ Top 10 ]")
                for stat in top_stats[:10]:
                        log.info(stat)

            time.sleep(poll_interval)

    def _get_tile(self, row, col, map_type, zoom):
        
        idx = self._to_tile_id(row, col, map_type, zoom)
        with self.tc_lock:
            tile = self.tiles.get(idx)
            if not tile:
                tile = self._open_tile(row, col, map_type, zoom)
            else:
                # Touch LRU on hit
                self._touch_tile(idx, tile)
        return tile

    def _open_tile(self, row, col, map_type, zoom):
        if self.maptype_override and self.maptype_override != "Use tile default":
            map_type = self.maptype_override
        idx = self._to_tile_id(row, col, map_type, zoom)

        log.debug(f"Get_tile: {idx}")
        with self.tc_lock:
            tile = self.tiles.get(idx)
            if not tile:
                self.misses += 1
                bump('tile_mem_miss')
                # Use target zoom level - supports both fixed and dynamic modes
                # Pass row/col for dynamic zoom computation based on predicted altitude
                tile = Tile(
                    col, row, map_type, zoom, 
                    cache_dir=self.cache_dir,
                    min_zoom=self.min_zoom,
                    max_zoom=self._get_target_zoom_level(zoom, row=row, col=col),
                )
                self.tiles[idx] = tile
                # New tile becomes MRU
                self._touch_tile(idx, tile)
                self.open_count[idx] = self.open_count.get(idx, 0) + 1
                if self.open_count[idx] > 1:
                    log.debug(f"Tile: {idx} opened for the {self.open_count[idx]} time.")
            elif tile.refs <= 0:
                # Only in this case would this cache have made a difference
                self.hits += 1
                bump('tile_mem_hits')

            tile.refs += 1
        return tile

    
    def _close_tile(self, row, col, map_type, zoom):
        tile_id = self._to_tile_id(row, col, map_type, zoom)
        with self.tc_lock:
            t = self.tiles.get(tile_id)
            if not t:
                log.warning(f"Attmpted to close unknown tile {tile_id}!")
                return False

            t.refs -= 1

            if self.enable_cache: # and not t.should_close():
                log.debug(f"Cache enabled.  Delay tile close for {tile_id}")
                return True

            if t.refs <= 0:
                log.debug(f"No more refs for {tile_id} closing...")
                t = self.tiles.pop(tile_id)
                t.close()
                t = None
                del(t)
            else:
                log.debug(f"Still have {t.refs} refs for {tile_id}")

        return True

# ============================================================
# Module-level cleanup helpers
# ============================================================

def shutdown():
    """Free network pools, worker threads and cached tiles to minimise RSS
    just before interpreter exit. Safe to call multiple times."""

    global chunk_getter

    # 1. Stop background download threads
    try:
        if chunk_getter is not None:
            chunk_getter.stop()
            chunk_getter = None
    except Exception as _err:
        log.debug(f"ChunkGetter stop error: {_err}")

    # 2. Iterate over every TileCacher instance still alive and flush
    #    its caches.  We avoid importing autoortho_fuse to prevent cycles; instead
    #    we search the GC list.
    import gc
    for obj in gc.get_objects():
        try:
            if isinstance(obj, TileCacher):
                with obj.tc_lock:
                    for tile in list(obj.tiles.values()):
                        tile.close()
                    obj.tiles.clear()
        except Exception:
            # Ignore any edge-case failures during shutdown
            pass

    try:
        if stats_batcher:
            stats_batcher.stop()
    except Exception:
        pass

    try:
        clear_process_memory_stat()
    except Exception:
        pass

    log.info("autoortho.getortho shutdown complete")
