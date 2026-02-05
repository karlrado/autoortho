import os
import time
import threading
import collections
from collections.abc import MutableMapping
from multiprocessing.managers import BaseManager
import psutil

# Handle imports for both frozen (PyInstaller) and direct Python execution
try:
    from autoortho.aoconfig import CFG
except ImportError:
    from aoconfig import CFG
import logging
log = logging.getLogger(__name__)


STATS = {}
_local_lock = threading.RLock()
_local_stats = {}

_manager = None
_store = None


def _connect_manager_from_env():
    addr = os.getenv("AO_STATS_ADDR")
    auth = os.getenv("AO_STATS_AUTH")
    if not addr or not auth:
        return None, None
    host, port = addr.split(":")

    class _StatsManager(BaseManager):
        pass
    _StatsManager.register(
        'get_store',
        exposed=['inc', 'inc_many', 'set', 'get', 'delete', 'keys', 'snapshot']
    )
    m = _StatsManager(address=(host, int(port)), authkey=auth.encode('utf-8'))
    m.connect()
    return m, m.get_store()


def bind_local_store(store):
    """
    In the parent process, bind the module to an already-created StatsStore
    instance (not a proxy). All helpers (get/set/inc) and STATS[...] will
    use this store from now on.
    """
    global _store, STATS
    _store = store
    STATS = _StatsMapping(_store)
    log.info("aostats: bound to shared StatsStore")


try:
    _manager, _store = _connect_manager_from_env()
except Exception as e:
    _manager, _store = None, None
    log.debug(f"aostats: manager connect failed: {e}")


class _StatsMapping(MutableMapping):
    def __init__(self, remote_store=None):
        self._remote = remote_store

    def __getitem__(self, key):
        if self._remote:
            val = self._remote.get(key, None)
            if val is None: raise KeyError(key)
            return val
        with _local_lock:
            return _local_stats[key]

    def __setitem__(self, key, value):
        if self._remote:
            self._remote.set(key, value)
            return
        with _local_lock:
            _local_stats[key] = value

    def __delitem__(self, key):
        if self._remote:
            self._remote.delete(key)
            return
        with _local_lock:
            del _local_stats[key]

    def __iter__(self):
        if self._remote:
            # Remote has no native iter; snapshot for iteration
            return iter(self._remote.snapshot().keys())
        with _local_lock:
            return iter(dict(_local_stats).keys())

    def __len__(self):
        if self._remote:
            return len(self._remote.keys())
        with _local_lock:
            return len(_local_stats)

    # Provide dict-like helpers
    def get(self, key, default=None):
        if self._remote:
            val = self._remote.get(key, None)
            return default if val is None else val
        with _local_lock:
            return _local_stats.get(key, default)

    def items(self):
        if self._remote:
            return self._remote.snapshot().items()
        with _local_lock:
            return dict(_local_stats).items()


STATS = _StatsMapping(_store)


def set_stat(stat, value):
    if _store:
        _store.set(stat, value)
    else:
        with _local_lock:
            _local_stats[stat] = value


def get_stat(stat):
    if _store:
        return _store.get(stat, 0)
    with _local_lock:
        return _local_stats.get(stat, 0)


def inc_stat(stat, amount=1):
    if _store:
        return _store.inc(stat, amount)  # atomic on the shared store
    with _local_lock:
        _local_stats[stat] = _local_stats.get(stat, 0) + amount
        return _local_stats[stat]


def inc_many(items: dict):
    """Batch increment for workers; falls back to local dict when no manager."""
    if _store and hasattr(_store, "inc_many"):
        _store.inc_many(items)
    else:
        with _local_lock:
            for k, v in items.items():
                _local_stats[k] = _local_stats.get(k, 0) + int(v)


def delete_stat(stat: str):
    """Delete a stat from the shared store or local fallback."""
    if _store:
        try:
            _store.delete(stat)
        except Exception:
            pass
    else:
        with _local_lock:
            _local_stats.pop(stat, None)


def update_process_memory_stat():
    """Update this process's memory and heartbeat in the shared stats store.

    On macOS, Activity Monitor's "Memory" column can differ significantly from
    psutil's RSS due to shared libraries, memory-mapped files, and native 
    allocations. We try multiple approaches to get the most accurate value.
    
    Writes keys:
      - proc_mem_rss_bytes:<pid> = memory in bytes (best available metric)
      - proc_alive_ts:<pid> = unix timestamp of last heartbeat
      - proc_mem_mb:<pid> = human-readable MB value for debugging
    """
    try:
        import sys
        pid = os.getpid()
        proc = psutil.Process(pid)
        now_ts = int(time.time())
        
        # Get all available memory metrics
        mem_info = proc.memory_info()
        rss = mem_info.rss
        vms = getattr(mem_info, 'vms', 0)
        
        # On macOS, try to get 'footprint' if available (matches Activity Monitor better)
        # This requires macOS 10.9+ and psutil 5.7+
        footprint = 0
        if sys.platform == 'darwin':
            try:
                # Try to read footprint via memory_full_info if available
                full_info = proc.memory_full_info()
                # Some psutil versions provide 'uss' which is closer to footprint
                if hasattr(full_info, 'uss') and full_info.uss > 0:
                    footprint = full_info.uss
            except (AttributeError, psutil.AccessDenied, NotImplementedError, psutil.NoSuchProcess):
                pass
        
        # Use the best available metric:
        # 1. Footprint/USS if available (most accurate on macOS)
        # 2. Otherwise RSS
        if footprint > 0:
            mem_bytes = footprint
        else:
            mem_bytes = rss
        
        # Log detailed memory info periodically for debugging discrepancies
        # This helps diagnose cases where Activity Monitor shows different values
        if vms > rss * 3:
            log.debug(f"Memory detail PID={pid}: RSS={rss//(1024**2)}MB, VMS={vms//(1024**2)}MB, "
                     f"footprint={footprint//(1024**2)}MB, using={mem_bytes//(1024**2)}MB")
        
        set_stat(f"proc_mem_rss_bytes:{pid}", int(mem_bytes))
        set_stat(f"proc_alive_ts:{pid}", now_ts)
        # Also publish human-readable versions for debugging
        set_stat(f"proc_mem_mb:{pid}", int(mem_bytes // (1024 * 1024)))
        
        # Add thread count for debugging (high thread counts can explain memory discrepancies)
        try:
            thread_count = proc.num_threads()
            set_stat(f"proc_threads:{pid}", thread_count)
        except Exception:
            pass
    except Exception as _err:
        # Best-effort; ignore failures
        log.debug(f"update_process_memory_stat: {_err}")


def clear_process_memory_stat():
    """Remove this process's memory/heartbeat keys from the stats store."""
    pid = os.getpid()
    delete_stat(f"proc_mem_rss_bytes:{pid}")
    delete_stat(f"proc_alive_ts:{pid}")


def update_decode_pool_stats():
    """
    Publish decode pool statistics to the stats store.
    
    This is called periodically from stats loops to track native
    JPEG decode buffer pool usage, which is critical for diagnosing
    memory issues.
    
    Published stats:
    - decode_pool_fixed: Number of fixed pool buffers
    - decode_pool_available: Available fixed buffers
    - decode_pool_acquired: Buffers currently in use from fixed pool
    - decode_pool_overflow: Overflow buffers allocated via malloc
    - decode_pool_overflow_mb: Overflow memory in MB
    - decode_pool_limit_mb: Memory limit in MB
    """
    try:
        # Import here to avoid circular imports
        try:
            from autoortho.aopipeline import AoDDS
        except ImportError:
            from aopipeline import AoDDS
        
        stats = AoDDS.get_decode_pool_stats()
        if stats:
            set_stat('decode_pool_fixed', stats['total'])
            set_stat('decode_pool_available', stats['available'])
            set_stat('decode_pool_acquired', stats['acquired'])
            set_stat('decode_pool_overflow', stats['overflow_count'])
            overflow_mb = stats['overflow_bytes'] // (1024 * 1024)
            limit_mb = stats['memory_limit'] // (1024 * 1024)
            set_stat('decode_pool_overflow_mb', overflow_mb)
            set_stat('decode_pool_limit_mb', limit_mb)

            # Log warning if overflow is growing
            if stats['overflow_count'] > 0:
                log.debug(f"Decode pool overflow: {stats['overflow_count']} "
                          f"buffers, {overflow_mb} MB")
    except Exception as e:
        # Best-effort; ignore failures
        log.debug(f"update_decode_pool_stats: {e}")


class AOStats(object):
    def __init__(self):
        self.running = False
        self._t = threading.Thread(daemon=True, target=self.show)

    def start(self):
        self.running = True
        self._t.start()

    def stop(self):
        self.running = False
        self._t.join()

    def show(self):
        while self.running:
            time.sleep(10)
            try:
                snap = _store.snapshot() if _store else dict(STATS.items())
                log.info(f"STATS: {snap}")
                        
            except Exception as e:
                log.debug(f"aostats.show error: {e}")


class StatTracker(object):

    def __init__(self, start=None, end=None, default=-1, maxlen=None):
        self.fetch_times = {}
        self.averages = {}
        self.counts = {}
        self.maxlen = 25

        if maxlen:
            self.maxlen = maxlen

        if start is not None and end is not None:
            if end < start:
                inc = -1
            else:
                inc = 1

            for i in range(start, end, inc):
                self.averages[i] = default
                self.counts[i] = default

    def set(self, key, value):
        self.counts[key] = self.counts.get(key, 0) + 1
        self.fetch_times.setdefault(key, collections.deque(maxlen=self.maxlen)).append(value)
        self.averages[key] = round(sum(self.fetch_times.get(key))/len(self.fetch_times.get(key)), 3)


class StatsStore:
    def __init__(self):
        self._lock = threading.RLock()
        self._data = {}  # dict[str, any]

    # Atomic increment
    def inc(self, key, amount=1):
        with self._lock:
            self._data[key] = self._data.get(key, 0) + amount
            return self._data[key]

    def inc_many(self, items):
        # items: dict[str, int]
        with self._lock:
            for k, a in items.items():
                self._data[k] = self._data.get(k, 0) + a

    def set(self, key, value):
        with self._lock:
            self._data[key] = value

    def get(self, key, default=0):
        with self._lock:
            return self._data.get(key, default)

    def delete(self, key):
        with self._lock:
            self._data.pop(key, None)

    def keys(self):
        with self._lock:
            return list(self._data.keys())

    def snapshot(self):
        with self._lock:
            return dict(self._data)


class StatsBatcher:
    def __init__(self, flush_interval=0.05, max_items=200):
        self._buf = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._flush_interval = flush_interval
        self._max_items = max_items
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def add(self, key: str, amount: int = 1):
        with self._lock:
            self._buf[key] = self._buf.get(key, 0) + int(amount)
            if len(self._buf) >= self._max_items:
                self._flush_unlocked()

    def add_many(self, items: dict):
        with self._lock:
            for k, v in items.items():
                self._buf[k] = self._buf.get(k, 0) + int(v)
            if len(self._buf) >= self._max_items:
                self._flush_unlocked()

    def flush(self):
        with self._lock:
            self._flush_unlocked()

    def stop(self):
        # Flush BEFORE setting stop flag, otherwise _flush_unlocked() will
        # detect the stop flag and discard the buffer instead of flushing
        self.flush()
        self._stop.set()

    def _flush_unlocked(self):
        if not self._buf:
            return
        # If stopping, don't bother flushing - manager may already be gone
        if self._stop.is_set():
            self._buf = {}
            return
        payload, self._buf = self._buf, {}
        try:
            inc_many(payload)
        except (ConnectionResetError, BrokenPipeError, EOFError, OSError):
            # Manager already shut down - discard stats silently
            pass
        except Exception:
            # As a safety net, fall back to single increments.
            # But check if we're stopping to avoid cascade errors
            if self._stop.is_set():
                return
            for k, v in payload.items():
                try:
                    inc_stat(k, v)
                except (ConnectionResetError, BrokenPipeError, EOFError, OSError):
                    # Manager shut down mid-fallback - stop trying
                    break

    def _loop(self):
        while not self._stop.wait(self._flush_interval):
            try:
                self.flush()
            except (ConnectionResetError, BrokenPipeError, EOFError, OSError):
                # Manager shut down - exit loop silently
                break
