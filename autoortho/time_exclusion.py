#!/usr/bin/env python3

"""
Time Exclusion Manager for AutoOrtho.

This module provides time-based exclusion functionality that allows users to
disable AutoOrtho's scenery during specific time ranges in the simulator.
When active, AutoOrtho's DSF files are hidden from X-Plane, causing it to
fall back to default scenery.

The manager ensures safe transitions by:
- Tracking which DSF files are currently in use
- Not hiding DSFs that are actively being read
- Only applying exclusions to new DSF accesses

Usage:
    from time_exclusion import time_exclusion_manager

    # Check if a DSF path should be hidden
    if time_exclusion_manager.should_hide_dsf(path):
        # Return file not found or hide from directory listing
        pass

    # Register DSF usage (call when DSF is opened)
    time_exclusion_manager.register_dsf_open(path)

    # Unregister DSF usage (call when DSF is closed)
    time_exclusion_manager.register_dsf_close(path)
"""

import threading
import time
import logging

# Handle imports for both frozen (PyInstaller) and direct Python execution
try:
    from autoortho.aoconfig import CFG
except ImportError:
    from aoconfig import CFG

log = logging.getLogger(__name__)


def parse_time_string(time_str):
    """
    Parse a time string in HH:MM format to seconds since midnight.
    
    Args:
        time_str: Time string in HH:MM format (e.g., "22:00", "06:30")
        
    Returns:
        int: Seconds since midnight, or None if parsing fails
    """
    try:
        time_str = str(time_str).strip()
        if ':' in time_str:
            parts = time_str.split(':')
            hours = int(parts[0])
            minutes = int(parts[1]) if len(parts) > 1 else 0
            if 0 <= hours <= 23 and 0 <= minutes <= 59:
                return hours * 3600 + minutes * 60
        else:
            # Try parsing as raw seconds
            return int(float(time_str))
    except (ValueError, TypeError) as e:
        log.warning(f"Failed to parse time string '{time_str}': {e}")
        return None


def format_time_from_seconds(seconds):
    """
    Format seconds since midnight to HH:MM string.
    
    Args:
        seconds: Seconds since midnight
        
    Returns:
        str: Time string in HH:MM format
    """
    if seconds < 0:
        return "??:??"
    hours = int(seconds // 3600) % 24
    minutes = int((seconds % 3600) // 60)
    return f"{hours:02d}:{minutes:02d}"


class TimeExclusionManager:
    """
    Manages time-based exclusion of AutoOrtho scenery.
    
    This class monitors the simulator's local time and determines when
    AutoOrtho's DSF files should be hidden from X-Plane. It safely handles
    transitions by tracking which DSFs are currently in use.
    """
    
    def __init__(self):
        """Initialize the time exclusion manager."""
        self._lock = threading.RLock()
        
        # Track DSF files currently in use (path -> open count)
        self._active_dsfs = {}
        
        # Cache for exclusion state
        self._exclusion_active = False
        self._last_check_time = 0
        self._check_interval = 1.0  # Check every second
        
        # Track if sim time has become available (for one-time logging)
        self._sim_time_was_available = False
        
        # Reference to dataref tracker (set later to avoid circular import)
        self._dataref_tracker = None
        
        # Background monitoring thread
        self._monitor_thread = None
        self._monitor_running = False
        
        log.info("TimeExclusionManager initialized")
    
    def set_dataref_tracker(self, tracker):
        """
        Set the dataref tracker reference and start background monitoring.
        
        Args:
            tracker: DatarefTracker instance
        """
        self._dataref_tracker = tracker
        self._start_monitor()
    
    def _start_monitor(self):
        """Start the background monitoring thread."""
        if self._monitor_running:
            return
        
        self._monitor_running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="TimeExclusionMonitor"
        )
        self._monitor_thread.start()
        log.debug("Time exclusion monitor thread started")
    
    def _monitor_loop(self):
        """
        Background loop that periodically checks exclusion state.
        
        This ensures the state is updated even when no DSF files are
        being accessed (e.g., when exclusion is active and hiding all DSFs).
        """
        import time as time_module
        
        while self._monitor_running:
            try:
                # Check if time exclusion is enabled
                enabled, _, _, _ = self._get_config()
                if enabled:
                    # Trigger a state check (this updates the cache and logs changes)
                    self.is_exclusion_active()
            except Exception as e:
                log.debug(f"Time exclusion monitor error: {e}")
            
            # Sleep for the check interval
            time_module.sleep(self._check_interval)
    
    def stop(self):
        """Stop the background monitoring thread."""
        self._monitor_running = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        log.debug("Time exclusion monitor stopped")
    
    def _get_config(self):
        """
        Get current time exclusion configuration.
        
        Returns:
            tuple: (enabled, start_seconds, end_seconds, default_to_exclusion) 
                   or (False, None, None, False) if disabled/invalid
        """
        try:
            enabled = getattr(CFG.time_exclusion, 'enabled', False)
            if not enabled:
                return (False, None, None, False)
            
            start_str = getattr(CFG.time_exclusion, 'start_time', '22:00')
            end_str = getattr(CFG.time_exclusion, 'end_time', '06:00')
            default_to_exclusion = getattr(CFG.time_exclusion, 'default_to_exclusion', False)
            
            start_sec = parse_time_string(start_str)
            end_sec = parse_time_string(end_str)
            
            if start_sec is None or end_sec is None:
                log.warning(f"Invalid time exclusion config: start={start_str}, end={end_str}")
                return (False, None, None, False)
            
            return (True, start_sec, end_sec, default_to_exclusion)
        except Exception as e:
            log.debug(f"Error reading time exclusion config: {e}")
            return (False, None, None, False)
    
    def _is_time_in_range(self, current_time_sec, start_sec, end_sec):
        """
        Check if current time falls within the exclusion range.
        
        Handles overnight ranges (e.g., 22:00 to 06:00).
        
        Args:
            current_time_sec: Current time in seconds since midnight
            start_sec: Start of exclusion range (seconds since midnight)
            end_sec: End of exclusion range (seconds since midnight)
            
        Returns:
            bool: True if current time is within the exclusion range
        """
        if start_sec <= end_sec:
            # Normal range (e.g., 09:00 to 17:00)
            return start_sec <= current_time_sec < end_sec
        else:
            # Overnight range (e.g., 22:00 to 06:00)
            # Time is in range if it's >= start OR < end
            return current_time_sec >= start_sec or current_time_sec < end_sec
    
    def _check_exclusion_state(self):
        """
        Check if time exclusion should be active based on current sim time.
        
        Returns:
            bool: True if exclusion should be active
        """
        enabled, start_sec, end_sec, default_to_exclusion = self._get_config()
        
        if not enabled:
            return False
        
        # Get current sim time from dataref tracker
        if self._dataref_tracker is None:
            # No dataref tracker available - use default behavior
            return default_to_exclusion
        
        current_time = self._dataref_tracker.get_local_time_sec()
        
        if current_time < 0:
            # No valid sim time available - reset the flag so we log again next time
            if self._sim_time_was_available:
                self._sim_time_was_available = False
                log.debug("Sim time no longer available")
            # Use configured default
            if default_to_exclusion:
                log.debug("Sim time not available - defaulting to exclusion active")
            return default_to_exclusion
        
        # Log once when sim time first becomes available
        if not self._sim_time_was_available:
            self._sim_time_was_available = True
            log.info(f"Sim time now available: {format_time_from_seconds(current_time)}")
        
        return self._is_time_in_range(current_time, start_sec, end_sec)
    
    def is_exclusion_active(self):
        """
        Check if time exclusion is currently active.
        
        This method caches the result and only rechecks periodically
        for performance.
        
        Returns:
            bool: True if exclusion is active
        """
        current_time = time.time()
        
        # Use cached result if recent
        if current_time - self._last_check_time < self._check_interval:
            return self._exclusion_active
        
        with self._lock:
            # Double-check after acquiring lock
            if current_time - self._last_check_time < self._check_interval:
                return self._exclusion_active
            
            old_state = self._exclusion_active
            self._exclusion_active = self._check_exclusion_state()
            self._last_check_time = current_time
            
            # Log state changes
            if old_state != self._exclusion_active:
                if self._exclusion_active:
                    enabled, start_sec, end_sec, default_to_excl = self._get_config()
                    current_time_sec = self._dataref_tracker.get_local_time_sec() if self._dataref_tracker else -1
                    if current_time_sec < 0:
                        log.info(
                            f"Time exclusion ACTIVATED (sim time not yet available, using default) "
                            f"(exclusion range: {format_time_from_seconds(start_sec)} - {format_time_from_seconds(end_sec)})"
                        )
                    else:
                        log.info(
                            f"Time exclusion ACTIVATED at sim time {format_time_from_seconds(current_time_sec)} "
                            f"(exclusion range: {format_time_from_seconds(start_sec)} - {format_time_from_seconds(end_sec)})"
                        )
                else:
                    log.info("Time exclusion DEACTIVATED")
            
            return self._exclusion_active
    
    def register_dsf_open(self, path):
        """
        Register that a DSF file has been opened.
        
        This is called when X-Plane opens a DSF file. The DSF will not
        be hidden while it's registered as open, even if exclusion becomes
        active.
        
        Args:
            path: Path to the DSF file
        """
        with self._lock:
            normalized_path = self._normalize_path(path)
            if normalized_path in self._active_dsfs:
                self._active_dsfs[normalized_path] += 1
            else:
                self._active_dsfs[normalized_path] = 1
            log.debug(f"DSF opened: {normalized_path} (count: {self._active_dsfs[normalized_path]})")
    
    def register_dsf_close(self, path):
        """
        Register that a DSF file has been closed.
        
        Args:
            path: Path to the DSF file
        """
        with self._lock:
            normalized_path = self._normalize_path(path)
            if normalized_path in self._active_dsfs:
                self._active_dsfs[normalized_path] -= 1
                if self._active_dsfs[normalized_path] <= 0:
                    del self._active_dsfs[normalized_path]
                    log.debug(f"DSF closed (fully): {normalized_path}")
                else:
                    log.debug(f"DSF closed: {normalized_path} (count: {self._active_dsfs[normalized_path]})")
    
    def is_dsf_in_use(self, path):
        """
        Check if a DSF file is currently in use.
        
        Args:
            path: Path to the DSF file
            
        Returns:
            bool: True if the DSF is currently open
        """
        with self._lock:
            normalized_path = self._normalize_path(path)
            return normalized_path in self._active_dsfs
    
    def should_hide_dsf(self, path):
        """
        Determine if a DSF file should be hidden from X-Plane.
        
        A DSF is hidden if:
        1. Time exclusion is enabled and currently active
        2. The DSF is NOT currently in use
        
        This ensures safe transitions - DSFs that are already being used
        will continue to be served until they are released.
        
        Args:
            path: Path to the DSF file
            
        Returns:
            bool: True if the DSF should be hidden
        """
        if not self.is_exclusion_active():
            return False
        
        # Don't hide DSFs that are currently in use
        if self.is_dsf_in_use(path):
            return False
        
        return True
    
    def _normalize_path(self, path):
        """
        Normalize a path for consistent comparison.
        
        Args:
            path: File path
            
        Returns:
            str: Normalized path
        """
        if path is None:
            return ""
        # Convert to forward slashes and lowercase for consistent comparison
        return str(path).replace('\\', '/').lower()
    
    def get_status(self):
        """
        Get current status information for display.
        
        Returns:
            dict: Status information including enabled state, time range,
                  current exclusion state, and active DSF count
        """
        with self._lock:
            enabled, start_sec, end_sec, default_to_excl = self._get_config()
            current_time = -1
            if self._dataref_tracker:
                current_time = self._dataref_tracker.get_local_time_sec()
            
            return {
                'enabled': enabled,
                'start_time': format_time_from_seconds(start_sec) if start_sec is not None else "N/A",
                'end_time': format_time_from_seconds(end_sec) if end_sec is not None else "N/A",
                'default_to_exclusion': default_to_excl,
                'exclusion_active': self._exclusion_active,
                'current_sim_time': format_time_from_seconds(current_time) if current_time >= 0 else "N/A",
                'sim_time_available': current_time >= 0,
                'active_dsf_count': len(self._active_dsfs),
            }


# Singleton instance
time_exclusion_manager = TimeExclusionManager()

