"""
bundle_consolidator.py - Background worker for consolidating JPEGs into AOB2 bundles.

This module provides background processing to consolidate individual JPEG
cache files into AOB2 bundles, significantly improving read performance.

Features:
- Background consolidation (doesn't block downloads)
- Priority queue for user-visible vs prefetch tiles
- Debouncing to batch consolidation work
- Verification before deleting source JPEGs
- Compaction worker for bundle maintenance

Usage:
    from autoortho.aopipeline.bundle_consolidator import BundleConsolidator
    
    consolidator = BundleConsolidator(cache_dir)
    consolidator.schedule(row=456, col=123, maptype="BI", zoom=16)
    
    # ... later ...
    consolidator.shutdown()
"""

import logging
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)


class ConsolidationPriority(IntEnum):
    """Priority levels for consolidation tasks."""
    HIGH = 0       # User-visible tiles (immediate need)
    NORMAL = 1     # Prefetch tiles
    LOW = 2        # Background consolidation
    COMPACTION = 3 # Compaction tasks (lowest priority)


@dataclass(order=True)
class ConsolidationTask:
    """A consolidation task with priority ordering."""
    priority: int
    row: int = field(compare=False)  # Chunk grid origin row (at zoom level)
    col: int = field(compare=False)  # Chunk grid origin col (at zoom level)
    maptype: str = field(compare=False)
    zoom: int = field(compare=False)  # Chunk zoom level (effective_zoom)
    timestamp: float = field(default_factory=time.time, compare=False)
    chunks_per_side: int = field(default=16, compare=False)
    # Optional: JPEG data passed directly (avoids disk read)
    # List of bytes or None for each chunk position
    jpeg_datas: Optional[List[Optional[bytes]]] = field(default=None, compare=False)
    # Tile ID coordinates for bundle path (at tilename_zoom)
    # If None, falls back to using row/col/zoom (legacy behavior)
    tile_row: Optional[int] = field(default=None, compare=False)
    tile_col: Optional[int] = field(default=None, compare=False)
    tile_zoom: Optional[int] = field(default=None, compare=False)


class BundleConsolidator:
    """
    Background worker that consolidates JPEGs into bundles.
    
    Runs in a dedicated thread pool (1-2 workers) to avoid blocking
    download threads. Uses priority queue to process user-visible
    tiles before prefetch tiles.
    
    Thread Safety: All public methods are thread-safe.
    """
    
    # Default configuration
    DEFAULT_WORKERS = 1
    DEBOUNCE_DELAY_MS = 500  # Wait 500ms after scheduling (cache writes are now synchronous)
    RETRY_DELAY_MS = 500  # Retry delay if no files found (fallback for edge cases)
    MAX_RETRIES = 1  # Single retry for edge cases (sync writes should make this rare)
    COMPACTION_CHECK_INTERVAL = 300  # Check for compaction every 5 minutes
    COMPACTION_THRESHOLD = 0.30  # 30% garbage triggers compaction
    
    def __init__(
        self,
        cache_dir: str,
        delete_jpegs: bool = True,
        max_workers: int = DEFAULT_WORKERS,
        enabled: bool = True
    ):
        """
        Initialize the consolidator.
        
        Args:
            cache_dir: Base cache directory
            delete_jpegs: If True, delete source JPEGs after successful bundle creation
            max_workers: Number of worker threads (1-2 recommended)
            enabled: If False, consolidation is disabled (no-op)
        """
        self.cache_dir = cache_dir
        self.delete_jpegs = delete_jpegs
        self.enabled = enabled
        
        # Task queue and tracking
        self._task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._pending: Dict[str, ConsolidationTask] = {}  # tile_key -> task
        self._pending_lock = threading.Lock()
        
        # Debouncing: track last chunk arrival per tile
        self._last_chunk_time: Dict[str, float] = {}
        self._debounce_lock = threading.Lock()
        
        # Worker pool
        self._executor: Optional[ThreadPoolExecutor] = None
        self._shutdown_event = threading.Event()
        self._workers_started = False
        
        if enabled and max_workers > 0:
            self._executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="bundle_consolidator"
            )
            self._start_workers()
        
        # Statistics
        self._stats = {
            'bundles_created': 0,
            'bundles_updated': 0,
            'jpegs_consolidated': 0,
            'bytes_saved': 0,
            'errors': 0,
            'in_memory_consolidations': 0,  # Used in-memory data (no disk read)
            'disk_read_consolidations': 0,  # Had to read from disk
        }
        self._stats_lock = threading.Lock()
        
        # Callbacks
        self._on_bundle_created: Optional[Callable] = None
        self._on_error: Optional[Callable] = None
    
    def _start_workers(self):
        """Start background worker threads."""
        if self._workers_started:
            return
        
        self._workers_started = True
        
        # Main consolidation worker
        if self._executor:
            self._executor.submit(self._worker_loop)
    
    def _worker_loop(self):
        """Main worker loop - processes consolidation tasks."""
        log.debug("Bundle consolidator worker started")
        
        # Track retry counts per tile
        retry_counts: Dict[str, int] = {}
        
        while not self._shutdown_event.is_set():
            try:
                # Get task with timeout to allow periodic shutdown check
                try:
                    task = self._task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                tile_key = self._get_tile_key(task.row, task.col, task.maptype)
                
                # Check debounce - wait if more chunks might be coming
                with self._debounce_lock:
                    last_chunk = self._last_chunk_time.get(tile_key, 0)
                    
                    # If chunk arrived recently, requeue with delay
                    time_since_last = (time.time() - last_chunk) * 1000
                    if time_since_last < self.DEBOUNCE_DELAY_MS:
                        # Requeue for later
                        self._task_queue.put(task)
                        time.sleep(0.05)  # Brief sleep to avoid busy loop
                        continue
                
                # Get retry count for this tile
                retry_count = retry_counts.get(tile_key, 0)
                
                # Process the task
                try:
                    success = self._consolidate_tile(task, retry_count)
                    
                    if not success and retry_count < self.MAX_RETRIES:
                        # Retry - wait for async writes to complete
                        # IMPORTANT: Clear jpeg_datas before retry to avoid memory bloat
                        # (retry will read from disk anyway since in-memory failed)
                        task.jpeg_datas = None
                        retry_counts[tile_key] = retry_count + 1
                        time.sleep(self.RETRY_DELAY_MS / 1000.0)
                        self._task_queue.put(task)  # Requeue for retry
                        continue  # Don't remove from pending yet
                    
                    # Done with this tile (success or max retries reached)
                    retry_counts.pop(tile_key, None)
                    
                except Exception as e:
                    log.error(f"Error consolidating tile {tile_key}: {e}")
                    with self._stats_lock:
                        self._stats['errors'] += 1
                    if self._on_error:
                        self._on_error(tile_key, e)
                    retry_counts.pop(tile_key, None)
                
                # CRITICAL: Free memory immediately after processing
                # With 300k+ JPEGs/session, holding references causes memory bloat
                task.jpeg_datas = None
                
                # Remove from pending
                with self._pending_lock:
                    self._pending.pop(tile_key, None)
                    
            except Exception as e:
                log.exception(f"Unexpected error in consolidator worker: {e}")
        
        log.debug("Bundle consolidator worker stopped")
    
    def _get_tile_key(self, row: int, col: int, maptype: str) -> str:
        """Generate unique key for a tile."""
        return f"{row}_{col}_{maptype}"
    
    def schedule(
        self,
        row: int,
        col: int,
        maptype: str,
        zoom: int,
        priority: ConsolidationPriority = ConsolidationPriority.NORMAL,
        chunks_per_side: int = 16,
        jpeg_datas: Optional[List[Optional[bytes]]] = None,
        tile_row: Optional[int] = None,
        tile_col: Optional[int] = None,
        tile_zoom: Optional[int] = None
    ) -> bool:
        """
        Schedule tile for bundle consolidation.
        
        Args:
            row: Chunk grid origin row (at zoom level)
            col: Chunk grid origin col (at zoom level)
            maptype: Map type identifier
            zoom: Zoom level of the chunks
            priority: Task priority
            chunks_per_side: Chunks per side
            jpeg_datas: Optional list of JPEG bytes for each chunk position.
                        If provided, consolidator uses this data directly (no disk read).
                        If None, consolidator reads from cache files on disk.
                        Length must be chunks_per_side * chunks_per_side.
            tile_row: Tile ID row at tilename_zoom (for bundle path)
            tile_col: Tile ID col at tilename_zoom (for bundle path)
            tile_zoom: Tile's tilename_zoom (for bundle path)
        
        Returns:
            True if scheduled, False if consolidation disabled or already pending
        """
        if not self.enabled:
            return False
        
        # Use tile_row/tile_col for the tile key if provided, otherwise use chunk row/col
        key_row = tile_row if tile_row is not None else row
        key_col = tile_col if tile_col is not None else col
        tile_key = self._get_tile_key(key_row, key_col, maptype)
        
        with self._pending_lock:
            # Skip if already pending
            if tile_key in self._pending:
                # Update priority if higher
                existing = self._pending[tile_key]
                if priority < existing.priority:
                    existing.priority = priority
                # Don't replace jpeg_datas - first submission wins
                return False
            
            # Create task with optional in-memory JPEG data
            task = ConsolidationTask(
                priority=priority,
                row=row,
                col=col,
                maptype=maptype,
                zoom=zoom,
                chunks_per_side=chunks_per_side,
                jpeg_datas=jpeg_datas,
                tile_row=tile_row,
                tile_col=tile_col,
                tile_zoom=tile_zoom
            )
            
            self._pending[tile_key] = task
            self._task_queue.put(task)
        
        return True
    
    def notify_chunk_complete(self, row: int, col: int, maptype: str):
        """
        Notify that a chunk has been downloaded (for debouncing).
        
        Call this when a chunk download completes to reset the debounce timer.
        """
        tile_key = self._get_tile_key(row, col, maptype)
        with self._debounce_lock:
            self._last_chunk_time[tile_key] = time.time()
    
    def _consolidate_tile(self, task: ConsolidationTask, retry_count: int = 0) -> bool:
        """
        Actual consolidation work.
        
        1. Use in-memory JPEG data if provided, otherwise read from disk
        2. Create/update bundle atomically
        3. Verify bundle integrity
        4. Delete source JPEGs if configured
        
        CRITICAL: Bundle paths are based on TILE ID coordinates (tile_row, tile_col at tile_zoom),
        NOT the chunk grid coordinates (task.row, task.col at task.zoom)!
        This ensures bundles can be found when looking up by tile ID.
        
        Returns:
            True if consolidation succeeded or should not retry
            False if no JPEGs found and should retry
        """
        # Handle imports for both frozen (PyInstaller) and direct Python execution
        try:
            from autoortho.utils.bundle_paths import get_bundle2_path, ensure_bundle2_dir
            from autoortho.aopipeline import AoBundle2
        except ImportError:
            from utils.bundle_paths import get_bundle2_path, ensure_bundle2_dir
            from aopipeline import AoBundle2
        
        # Use tile ID coordinates for bundle path if provided, otherwise fall back to chunk coords
        path_row = task.tile_row if task.tile_row is not None else task.row
        path_col = task.tile_col if task.tile_col is not None else task.col
        path_zoom = task.tile_zoom if task.tile_zoom is not None else task.zoom
        
        tile_key = self._get_tile_key(path_row, path_col, task.maptype)
        chunk_count = task.chunks_per_side * task.chunks_per_side
        
        # Determine if we have in-memory data or need to read from disk
        use_memory_data = (task.jpeg_datas is not None and 
                          len(task.jpeg_datas) == chunk_count)
        
        if use_memory_data:
            # FAST PATH: Use in-memory JPEG data (avoids 256 disk reads per tile)
            jpeg_datas = task.jpeg_datas
            log.debug(f"Consolidating tile {tile_key} at ZL{task.zoom} (in-memory data)")
            with self._stats_lock:
                self._stats['in_memory_consolidations'] += 1
        else:
            # FALLBACK PATH: Read from disk cache files
            log.debug(f"Consolidating tile {tile_key} at ZL{task.zoom} (disk read, attempt {retry_count + 1})")
            with self._stats_lock:
                self._stats['disk_read_consolidations'] += 1
            jpeg_datas = []
            
            for i in range(chunk_count):
                chunk_row = i // task.chunks_per_side
                chunk_col = i % task.chunks_per_side
                # task.col/row ARE the starting coordinates of the chunk grid (scaled coords)
                abs_col = task.col + chunk_col
                abs_row = task.row + chunk_row
                
                path = os.path.join(
                    self.cache_dir, 
                    f"{abs_col}_{abs_row}_{task.zoom}_{task.maptype}.jpg"
                )
                
                try:
                    with open(path, 'rb') as f:
                        jpeg_datas.append(f.read())
                except FileNotFoundError:
                    jpeg_datas.append(None)
        
        # Build paths list for deletion (always needed, even with in-memory data)
        jpeg_paths: List[Optional[str]] = []
        for i in range(chunk_count):
            chunk_row = i // task.chunks_per_side
            chunk_col = i % task.chunks_per_side
            abs_col = task.col + chunk_col
            abs_row = task.row + chunk_row
            path = os.path.join(
                self.cache_dir,
                f"{abs_col}_{abs_row}_{task.zoom}_{task.maptype}.jpg"
            )
            jpeg_paths.append(path if jpeg_datas[i] is not None else None)
        
        # Count valid chunks
        valid_count = sum(1 for d in jpeg_datas if d is not None)
        if valid_count == 0:
            if not use_memory_data and retry_count < self.MAX_RETRIES:
                log.debug(f"No JPEGs found for tile {tile_key} at ZL{task.zoom}, will retry. "
                         f"First path: {self.cache_dir}/{task.col}_{task.row}_{task.zoom}_{task.maptype}.jpg")
                return False  # Signal to retry
            else:
                log.debug(f"No JPEGs found for tile {tile_key} (in-memory={use_memory_data}), giving up.")
                return True  # Don't retry anymore
        
        log.debug(f"Found {valid_count}/{chunk_count} JPEGs for tile {tile_key} at ZL{task.zoom}")
        
        # CRITICAL: Use tile ID coordinates (path_row, path_col, path_zoom) for bundle path,
        # NOT chunk grid coordinates! This ensures bundles match tile ID naming.
        bundle_dir = ensure_bundle2_dir(self.cache_dir, path_row, path_col, path_zoom)
        bundle_path = get_bundle2_path(self.cache_dir, path_row, path_col, task.maptype, path_zoom)
        
        # Check if bundle exists (update vs create)
        bundle_exists = os.path.exists(bundle_path)
        
        try:
            if AoBundle2.is_available():
                # Use native implementation
                AoBundle2.create_bundle_from_data(
                    task.row, task.col, task.maptype, task.zoom,
                    jpeg_datas, bundle_path
                )
            else:
                # Use pure Python fallback
                AoBundle2.create_bundle_from_data_python(
                    task.row, task.col, task.maptype, task.zoom,
                    jpeg_datas, bundle_path
                )
        except Exception as e:
            log.error(f"Failed to create bundle for {tile_key}: {e}")
            raise
        
        # Verify bundle
        try:
            if AoBundle2.is_available():
                if not AoBundle2.validate(bundle_path):
                    raise RuntimeError("Bundle validation failed")
            else:
                # Python validation - just try to parse it
                bundle = AoBundle2.Bundle2Python(bundle_path)
                if bundle.get_chunk_count(task.zoom) != chunk_count:
                    raise RuntimeError("Bundle chunk count mismatch")
        except Exception as e:
            log.error(f"Bundle verification failed for {tile_key}: {e}")
            # Try to remove corrupted bundle
            try:
                os.remove(bundle_path)
            except:
                pass
            raise
        
        # Delete source JPEGs if configured
        if self.delete_jpegs:
            for path in jpeg_paths:
                if path:
                    # Retry deletion with small delays to handle Windows file locking
                    # WinError 32 occurs when another thread still has the file open
                    max_delete_attempts = 3
                    for attempt in range(max_delete_attempts):
                        try:
                            os.remove(path)
                            break  # Success
                        except FileNotFoundError:
                            break  # Already deleted, that's fine
                        except PermissionError as e:
                            # WinError 32: File in use by another process
                            if attempt < max_delete_attempts - 1:
                                time.sleep(0.1 * (attempt + 1))  # 100ms, 200ms delays
                            else:
                                log.debug(f"Could not delete {path} after {max_delete_attempts} attempts: {e}")
                        except Exception as e:
                            log.warning(f"Failed to delete {path}: {e}")
                            break
        
        # Update stats
        with self._stats_lock:
            if bundle_exists:
                self._stats['bundles_updated'] += 1
            else:
                self._stats['bundles_created'] += 1
            self._stats['jpegs_consolidated'] += valid_count
        
        # Callback
        if self._on_bundle_created:
            self._on_bundle_created(tile_key, bundle_path)
        
        log.debug(f"Consolidated {valid_count} JPEGs into bundle for {tile_key}")
        return True  # Success
    
    def get_stats(self) -> dict:
        """Get consolidation statistics."""
        with self._stats_lock:
            return self._stats.copy()
    
    def get_pending_count(self) -> int:
        """Get number of pending consolidation tasks."""
        with self._pending_lock:
            return len(self._pending)
    
    def is_pending(self, row: int, col: int, maptype: str) -> bool:
        """Check if a tile is pending consolidation."""
        tile_key = self._get_tile_key(row, col, maptype)
        with self._pending_lock:
            return tile_key in self._pending
    
    def set_callbacks(
        self,
        on_bundle_created: Optional[Callable[[str, str], None]] = None,
        on_error: Optional[Callable[[str, Exception], None]] = None
    ):
        """Set callback functions for events."""
        self._on_bundle_created = on_bundle_created
        self._on_error = on_error
    
    def shutdown(self, wait: bool = True):
        """
        Shutdown the consolidator.
        
        Args:
            wait: If True, wait for pending tasks to complete
        """
        log.debug("Shutting down bundle consolidator")
        self._shutdown_event.set()
        
        if self._executor:
            self._executor.shutdown(wait=wait)
            self._executor = None
        
        self._workers_started = False


class CompactionWorker:
    """
    Periodic compaction worker for fragmented bundles.
    
    Runs during idle time (no active downloads) to reclaim space
    from garbage data in bundles.
    """
    
    FRAGMENTATION_THRESHOLD = 0.30  # 30% garbage triggers compaction
    CHECK_INTERVAL = 300  # Check every 5 minutes
    
    def __init__(
        self,
        cache_dir: str,
        threshold: float = FRAGMENTATION_THRESHOLD,
        enabled: bool = True
    ):
        """
        Initialize compaction worker.
        
        Args:
            cache_dir: Base cache directory
            threshold: Fragmentation threshold for compaction
            enabled: If False, compaction is disabled
        """
        self.cache_dir = cache_dir
        self.threshold = threshold
        self.enabled = enabled
        
        self._shutdown_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._idle_event = threading.Event()
        self._idle_event.set()  # Start as idle
        
        # Stats
        self._stats = {
            'bundles_checked': 0,
            'bundles_compacted': 0,
            'bytes_reclaimed': 0,
        }
        self._stats_lock = threading.Lock()
    
    def start(self):
        """Start the compaction worker."""
        if not self.enabled or self._thread is not None:
            return
        
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="bundle_compaction",
            daemon=True
        )
        self._thread.start()
    
    def stop(self, wait: bool = True):
        """Stop the compaction worker."""
        self._shutdown_event.set()
        if self._thread and wait:
            self._thread.join(timeout=5.0)
        self._thread = None
    
    def set_busy(self):
        """Signal that the system is busy (don't compact)."""
        self._idle_event.clear()
    
    def set_idle(self):
        """Signal that the system is idle (can compact)."""
        self._idle_event.set()
    
    def _worker_loop(self):
        """Main compaction loop."""
        log.debug("Compaction worker started")
        
        while not self._shutdown_event.is_set():
            # Wait for idle state
            if not self._idle_event.wait(timeout=60.0):
                continue
            
            # Check shutdown
            if self._shutdown_event.is_set():
                break
            
            # Run compaction check
            try:
                self._check_and_compact()
            except Exception as e:
                log.error(f"Error during compaction check: {e}")
            
            # Sleep until next check
            self._shutdown_event.wait(timeout=self.CHECK_INTERVAL)
        
        log.debug("Compaction worker stopped")
    
    def _check_and_compact(self):
        """Check all bundles and compact if needed."""
        # Handle imports for both frozen (PyInstaller) and direct Python execution
        try:
            from autoortho.utils.bundle_paths import enumerate_bundles
            from autoortho.aopipeline import AoBundle2
        except ImportError:
            from utils.bundle_paths import enumerate_bundles
            from aopipeline import AoBundle2
        
        bundles = enumerate_bundles(self.cache_dir)
        
        for bundle_info in bundles:
            if self._shutdown_event.is_set() or not self._idle_event.is_set():
                break
            
            path = bundle_info['path']
            
            with self._stats_lock:
                self._stats['bundles_checked'] += 1
            
            try:
                if AoBundle2.is_available():
                    if AoBundle2.needs_compaction(path, self.threshold):
                        bytes_reclaimed = AoBundle2.compact(path)
                        if bytes_reclaimed > 0:
                            with self._stats_lock:
                                self._stats['bundles_compacted'] += 1
                                self._stats['bytes_reclaimed'] += bytes_reclaimed
                            log.info(f"Compacted {path}, reclaimed {bytes_reclaimed} bytes")
            except Exception as e:
                log.warning(f"Failed to compact {path}: {e}")
    
    def get_stats(self) -> dict:
        """Get compaction statistics."""
        with self._stats_lock:
            return self._stats.copy()


# ============================================================================
# Convenience Functions
# ============================================================================

_default_consolidator: Optional[BundleConsolidator] = None
_consolidator_lock = threading.Lock()


def get_consolidator(cache_dir: str = None) -> Optional[BundleConsolidator]:
    """Get or create the default consolidator."""
    global _default_consolidator
    
    with _consolidator_lock:
        if _default_consolidator is None and cache_dir:
            _default_consolidator = BundleConsolidator(cache_dir)
        return _default_consolidator


def schedule_consolidation(
    row: int,
    col: int,
    maptype: str,
    zoom: int,
    cache_dir: str = None,
    priority: ConsolidationPriority = ConsolidationPriority.NORMAL
) -> bool:
    """
    Convenience function to schedule consolidation.
    
    Uses the default consolidator if cache_dir matches.
    """
    consolidator = get_consolidator(cache_dir)
    if consolidator:
        return consolidator.schedule(row, col, maptype, zoom, priority)
    return False


def shutdown_consolidator():
    """Shutdown the default consolidator."""
    global _default_consolidator
    
    with _consolidator_lock:
        if _default_consolidator:
            _default_consolidator.shutdown()
            _default_consolidator = None
