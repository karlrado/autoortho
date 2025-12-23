#!/usr/bin/env python3

"""
DatarefTracker: Thread-safe X-Plane dataref collection via UDP.

This module provides a singleton DatarefTracker instance that connects
to X-Plane via UDP to receive real-time flight data (position, heading,
speed, altitude).

Connection State:
    The tracker automatically connects when X-Plane is running a flight
    and reconnects if the connection is lost. The 'connected' and
    'data_valid' flags indicate the current state.
"""

import binascii
import math
import time
import struct
import socket
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional

from aoconfig import CFG
import logging

log = logging.getLogger(__name__)
UDP_IP = "127.0.0.1"


# =============================================================================
# Flight Data Averaging
# =============================================================================
# Maintains a rolling 60-second window of flight samples for computing
# smoothed/averaged values used in predictive tile loading calculations.
# This reduces jitter from instantaneous readings and provides better
# estimates for vertical speed and heading trends.
# =============================================================================


@dataclass
class FlightSample:
    """
    A single flight data sample with timestamp.
    
    Used by FlightDataAverager to maintain a history of recent
    flight data for computing rolling averages.
    """
    timestamp: float    # Time when sample was recorded (time.time())
    lat: float          # Latitude in degrees
    lon: float          # Longitude in degrees
    alt_ft: float       # Pressure altitude in feet
    hdg: float          # Heading in degrees (0-360)
    spd: float          # Ground speed in m/s


class FlightDataAverager:
    """
    Maintains a 60-second rolling window of flight samples
    and computes averaged values for predictive calculations.
    
    The averager provides smoothed values for:
    - Vertical speed (computed from altitude change over time)
    - Heading (circular average to handle 359->1 wraparound)
    - Ground speed (simple arithmetic average)
    
    Thread Safety:
        All public methods are thread-safe. The averager uses its own
        lock separate from DatarefTracker to minimize contention.
    
    Usage:
        averager = FlightDataAverager()
        averager.add_sample(lat, lon, alt_ft, hdg, spd)
        averages = averager.get_averages()
        if averages:
            vs = averages['vertical_speed_fpm']
    """
    
    # Configuration constants
    WINDOW_SEC = 60.0       # Rolling window duration in seconds
    MIN_SAMPLES = 3         # Minimum samples needed for valid average
    MAX_SAMPLES = 600       # Maximum samples to store (~10 samples/sec max)
    MIN_TIME_DELTA = 0.01   # Minimum time delta in minutes (~0.6 seconds)
    
    def __init__(self):
        """Initialize with empty sample buffer."""
        self._samples: deque = deque(maxlen=self.MAX_SAMPLES)
        self._lock = threading.Lock()
        
        # Cached averages (updated on each add_sample call)
        self._avg_vertical_speed_fpm = 0.0  # feet per minute
        self._avg_heading = 0.0             # degrees (0-360)
        self._avg_ground_speed_mps = 0.0    # m/s
        self._is_valid = False
    
    def add_sample(self, lat: float, lon: float, alt_ft: float,
                   hdg: float, spd: float) -> None:
        """
        Add a new flight data sample and update cached averages.
        
        This method is called from the UDP listener thread each time
        valid flight data is received from X-Plane. Samples older than
        WINDOW_SEC are automatically pruned.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            alt_ft: Pressure altitude in feet
            hdg: Heading in degrees (0-360)
            spd: Ground speed in m/s
        """
        now = time.time()
        sample = FlightSample(
            timestamp=now,
            lat=lat,
            lon=lon,
            alt_ft=alt_ft,
            hdg=hdg,
            spd=spd
        )
        
        with self._lock:
            self._samples.append(sample)
            self._prune_old_samples(now)
            self._update_averages()
    
    def _prune_old_samples(self, now: float) -> None:
        """
        Remove samples older than the window.
        
        Called internally while holding the lock.
        
        Args:
            now: Current timestamp for comparison
        """
        cutoff = now - self.WINDOW_SEC
        while self._samples and self._samples[0].timestamp < cutoff:
            self._samples.popleft()
    
    def _update_averages(self) -> None:
        """
        Compute rolling averages from current samples.
        
        Called internally while holding the lock after each sample
        is added. Updates the cached average values.
        """
        if len(self._samples) < self.MIN_SAMPLES:
            self._is_valid = False
            return
        
        samples = list(self._samples)
        
        # --- Vertical Speed ---
        # Computed as (last_alt - first_alt) / time_delta
        # This gives average climb/descent rate over the window
        first, last = samples[0], samples[-1]
        time_delta_min = (last.timestamp - first.timestamp) / 60.0
        
        if time_delta_min > self.MIN_TIME_DELTA:
            alt_delta_ft = last.alt_ft - first.alt_ft
            self._avg_vertical_speed_fpm = alt_delta_ft / time_delta_min
        else:
            self._avg_vertical_speed_fpm = 0.0
        
        # --- Ground Speed ---
        # Simple arithmetic average of all samples
        self._avg_ground_speed_mps = sum(s.spd for s in samples) / len(samples)
        
        # --- Heading ---
        # Circular average to handle wraparound (e.g., 359° -> 1°)
        # Uses vector addition: average of unit vectors in heading direction
        sin_sum = sum(math.sin(math.radians(s.hdg)) for s in samples)
        cos_sum = sum(math.cos(math.radians(s.hdg)) for s in samples)
        self._avg_heading = math.degrees(math.atan2(sin_sum, cos_sum)) % 360
        
        self._is_valid = True
    
    def get_averages(self) -> Optional[dict]:
        """
        Get current averaged values.
        
        Returns:
            dict with keys:
                - 'vertical_speed_fpm': Average climb/descent rate (ft/min)
                - 'heading': Average heading (degrees, 0-360)
                - 'ground_speed_mps': Average ground speed (m/s)
            Returns None if not enough samples are available.
        """
        with self._lock:
            if not self._is_valid:
                return None
            return {
                'vertical_speed_fpm': self._avg_vertical_speed_fpm,
                'heading': self._avg_heading,
                'ground_speed_mps': self._avg_ground_speed_mps,
            }
    
    def get_vertical_speed_fpm(self) -> Optional[float]:
        """
        Get the averaged vertical speed.
        
        Returns:
            Vertical speed in feet per minute, or None if not valid.
            Positive = climbing, Negative = descending
        """
        with self._lock:
            if not self._is_valid:
                return None
            return self._avg_vertical_speed_fpm
    
    def clear(self) -> None:
        """
        Clear all samples and reset state.
        
        Called when connection to X-Plane is lost to ensure stale
        data isn't used when connection is re-established.
        """
        with self._lock:
            self._samples.clear()
            self._avg_vertical_speed_fpm = 0.0
            self._avg_heading = 0.0
            self._avg_ground_speed_mps = 0.0
            self._is_valid = False
    
    def is_valid(self) -> bool:
        """Check if averages are currently valid."""
        with self._lock:
            return self._is_valid
    
    def sample_count(self) -> int:
        """Get the current number of samples in the buffer."""
        with self._lock:
            return len(self._samples)
    
    def get_window_duration(self) -> float:
        """
        Get the actual time span covered by current samples.
        
        Returns:
            Duration in seconds, or 0 if fewer than 2 samples.
        """
        with self._lock:
            if len(self._samples) < 2:
                return 0.0
            return self._samples[-1].timestamp - self._samples[0].timestamp


class DatarefTracker(object):
    """Thread-safe tracker for X-Plane flight data via UDP."""

    # Protocol constants
    RREF_HEADER = b"RREF"
    RREF_CMD = b"RREF\x00"
    PACKET_HEADER_SIZE = 5
    VALUE_SIZE = 8  # idx (4 bytes) + value (4 bytes)
    DATAREF_STRING_SIZE = 400
    MESSAGE_SIZE = 413  # 5 (cmd) + 4 (freq) + 4 (idx) + 400 (string)
    RECV_BUFFER_SIZE = 1024
    DEFAULT_FREQ_HZ = 1
    RECONNECT_DELAY_SEC = 5
    UDP_TIMEOUT_SEC = 5
    THREAD_JOIN_TIMEOUT_SEC = 10

    # List of datarefs to request.
    # fmt:off
    datarefs = [
        # (dataref, unit, description, decimals for formatted output)
        ("sim/flightmodel/position/latitude", "°N",
         "The latitude of the aircraft", 6),
        ("sim/flightmodel/position/longitude", "°E",
         "The longitude of the aircraft", 6),
        ("sim/flightmodel/position/y_agl", "m",
         "AGL", 0),
        ("sim/flightmodel/position/mag_psi", "°",
         "The real magnetic heading of the aircraft", 0),
        ("sim/flightmodel/position/groundspeed", "m/s",
         "The ground speed of the aircraft", 0),
        ("sim/time/local_time_sec", "s",
         "Local time (seconds since midnight)", 0),
        ("sim/flightmodel2/position/pressure_altitude", "ft",
         "Pressure altitude in standard atmosphere", 0),
    ]
    # fmt:on

    def __init__(self):
        """
        Initialize DatarefTracker with thread-safe state and pre-built
        messages.
        """
        # Thread synchronization
        self._lock = threading.Lock()
        self._shutdown_flag = threading.Event()

        # Flight data (protected by _lock)
        self.lat = -1.0
        self.lon = -1.0
        self.alt = -1.0
        self.hdg = -1.0
        self.spd = -1.0
        self.local_time_sec = -1.0  # Local time (seconds since midnight)
        self.pressure_alt = -1.0  # Pressure altitude in feet
        self.connected = False
        self.data_valid = False

        # Thread management
        self.t = None
        self.running = False

        # Flight data averaging for predictive calculations
        # Maintains a 60-second rolling window of samples
        self.flight_averager = FlightDataAverager()

        # Socket
        self.sock = socket.socket(
            socket.AF_INET,  # Internet
            socket.SOCK_DGRAM,  # UDP
        )
        self.sock.settimeout(self.UDP_TIMEOUT_SEC)

        # Pre-build UDP messages for performance
        # (avoid repeated encoding/packing)
        self._build_messages()

        log.info("Dataref Tracker instance created")

    def _build_messages(self):
        """
        Pre-build subscribe and unsubscribe messages for all datarefs.
        """
        self.subscribe_messages = []
        self.unsubscribe_messages = []

        for idx, dataref in enumerate(self.datarefs):
            # Pre-encode the dataref string
            encoded = dataref[0].encode()

            # Pre-build subscribe message (freq=DEFAULT_FREQ_HZ)
            sub_msg = struct.pack(
                "<5sii400s",
                self.RREF_CMD,
                self.DEFAULT_FREQ_HZ,
                idx,
                encoded
            )
            self.subscribe_messages.append(sub_msg)

            # Pre-build unsubscribe message (freq=0)
            unsub_msg = struct.pack(
                "<5sii400s",
                self.RREF_CMD,
                0,
                idx,
                encoded
            )
            self.unsubscribe_messages.append(unsub_msg)

    def get_flight_data(self):
        """
        Thread-safe getter for all flight data.

        Returns:
            dict: Flight data with keys 'lat', 'lon', 'alt', 'hdg',
                  'spd', 'local_time_sec', 'pressure_alt', 'connected',
                  'data_valid', 'timestamp'.
                  Returns None if not connected or data is invalid.
        """
        with self._lock:
            if not self.connected or not self.data_valid:
                return None
            return {
                'lat': self.lat,
                'lon': self.lon,
                'alt': self.alt,
                'hdg': self.hdg,
                'spd': self.spd,
                'local_time_sec': self.local_time_sec,
                'pressure_alt': self.pressure_alt,
                'connected': self.connected,
                'data_valid': self.data_valid,
                'timestamp': time.time()
            }

    def get_local_time_sec(self):
        """
        Thread-safe getter for local sim time.

        Returns:
            float: Seconds since midnight in sim time, or -1 if not available.
        """
        with self._lock:
            if not self.connected or not self.data_valid:
                return -1.0
            return self.local_time_sec

    def get_pressure_alt(self):
        """
        Thread-safe getter for pressure altitude.

        Returns:
            float: Pressure altitude in feet, or -1 if not available.
        """
        with self._lock:
            if not self.connected or not self.data_valid:
                return -1.0
            return self.pressure_alt

    def start(self):
        """Start the UDP listening thread."""
        if self.running:
            return
        log.info("Starting UDP listening thread")
        self._shutdown_flag.clear()
        self.running = True
        self.t = threading.Thread(target=self._dataref_tracker_udp_listen)
        self.t.daemon = True  # Ensure thread doesn't prevent exit
        self.t.start()

    def stop(self):
        """Stop the UDP listening thread and unsubscribe from datarefs."""
        log.info("Dataref Tracker shutdown requested.")

        # Signal shutdown to prevent race conditions
        self._shutdown_flag.set()
        self.running = False

        # Unsubscribe from datarefs
        # X-Plane might already be shut down, so ignore errors
        try:
            self._request_datarefs(subscribe=False)
        except (Exception, OSError):
            pass

        # Wait for thread to terminate with timeout
        if self.t and self.t.is_alive():
            self.t.join(timeout=self.THREAD_JOIN_TIMEOUT_SEC)
            if self.t.is_alive():
                log.error(
                    "UDP listener thread did not terminate gracefully"
                )

        # Close socket
        try:
            self.sock.close()
        except Exception:
            pass

        log.info("Dataref Tracker stopped.")

    def _validate_position(self, lat, lon, alt):
        """
        Validate that position data is within reasonable ranges.

        Args:
            lat (float): Latitude in degrees
            lon (float): Longitude in degrees
            alt (float): Altitude in meters AGL

        Returns:
            bool: True if all values are valid, False otherwise
        """
        if not (-90 <= lat <= 90):
            log.warning(f"Invalid latitude: {lat}")
            return False
        if not (-180 <= lon <= 180):
            log.warning(f"Invalid longitude: {lon}")
            return False
        # Reasonable altitude range: -500m to 50000m AGL
        # (allowing negative for below ground detection)
        if not (-500 <= alt <= 50000):
            log.warning(f"Invalid altitude: {alt}")
            return False
        return True

    # We're really only interested in whether or not X-Plane is able
    # to send us datarefs, which it only does when a flight is active.
    # Note that a flight is not considered active when X-Plane is
    # displaying the "Reading new scenery files" splash screen before
    # a flight.
    #
    # - For the feature where we increase maxwait during the initial
    #   scenery load, we just need to know if the flight is active
    #   (could be flying). When it is not active (e.g., at a splash
    #   screen before the flight), we know we are not flying and can
    #   increase maxwait without the risk of a stutter.
    #
    # - For the predictive preload feature, we need to know if the
    #   flight is active so we can get good datarefs to predict the
    #   flight path.
    #
    # We don't really need to know if X-Plane is simply up and running.
    # So there really isn't any need to use the "beacon" feature to
    # "find" a running instance of X-Plane. If we ever have a need to
    # do so, here are some resources:
    # https://github.com/charlylima/XPlaneUDP/blob/master/XPlaneUdp.py
    # https://gitlab.bliesener.com/jbliesener/PiDisplay/-/...
    #                                   .../blob/master/XPlaneUDP.py
    #
    # It therefore seems like the best way to detect when we can get
    # datarefs is to just periodically subscribe for datarefs and see
    # if a response comes back in a reasonable time. If it does not,
    # clear/reset the socket and try again after a short wait.
    def _dataref_tracker_udp_listen(self):
        """Main UDP listening loop (runs in separate thread)."""
        log.info("UDP listening thread started and listening!")

        # Subscribe to dataref updates at the default rate
        self._request_datarefs(subscribe=True)

        while self.running and not self._shutdown_flag.is_set():
            try:
                data, addr = self.sock.recvfrom(self.RECV_BUFFER_SIZE)
            except Exception:
                # All exceptions including ConnectionResetError and
                # socket.timeout are handled in the same way.
                # This can happen when X-Plane is not running yet or
                # has stopped. For now, reset the socket, wait a bit
                # (quietly), and retry.

                # Update connection state (thread-safe)
                with self._lock:
                    if self.connected:
                        log.info("Flight has stopped.")
                    self.connected = False
                    self.data_valid = False

                # Clear flight averager when connection is lost
                # This ensures stale data isn't used when reconnecting
                self.flight_averager.clear()

                # Check if we're shutting down before recreating socket
                if self._shutdown_flag.is_set():
                    break

                # Recreate socket
                try:
                    self.sock.close()
                except Exception:
                    pass

                self.sock = socket.socket(
                    socket.AF_INET,  # Internet
                    socket.SOCK_DGRAM,  # UDP
                )
                self.sock.settimeout(self.UDP_TIMEOUT_SEC)

                # Wait before retrying
                time.sleep(self.RECONNECT_DELAY_SEC)

                # Check again if we're shutting down
                if self._shutdown_flag.is_set():
                    break

                # Re-subscribe to datarefs
                self._request_datarefs(subscribe=True)
                continue

            # Socket read was successful - decode packet
            values = self._decode_packet(data)

            # Update flight data (thread-safe, atomic update)
            with self._lock:
                if not self.connected:
                    log.info("Flight is starting.")
                    self.connected = True

                # Accept 5, 6, or 7 values for backward compatibility
                # (6th value is local_time_sec, 7th is pressure_alt)
                if len(values) >= 5:
                    lat = values[0]
                    lon = values[1]
                    alt = values[2]
                    hdg = values[3]
                    spd = values[4]
                    # local_time is optional (6th value) for backward compat
                    local_time = values[5] if len(values) >= 6 else None
                    # pressure_alt is optional (7th value)
                    pressure_alt = values[6] if len(values) >= 7 else None

                    # Validate position data
                    if self._validate_position(lat, lon, alt):
                        self.lat = lat
                        self.lon = lon
                        self.alt = alt
                        self.hdg = hdg
                        self.spd = spd
                        # Only update local_time_sec if we received it
                        if local_time is not None:
                            self.local_time_sec = local_time
                        # Only update pressure_alt if we received it
                        if pressure_alt is not None:
                            self.pressure_alt = pressure_alt
                        self.data_valid = True

                        # Feed sample to flight averager for predictive calculations
                        # Only add sample if we have valid pressure altitude
                        if self.pressure_alt > 0:
                            self.flight_averager.add_sample(
                                lat=self.lat,
                                lon=self.lon,
                                alt_ft=self.pressure_alt,
                                hdg=self.hdg,
                                spd=self.spd
                            )
                    else:
                        self.data_valid = False
                else:
                    # Not enough values - log and mark invalid
                    log.debug(f"Incomplete packet: got {len(values)}, need 5+")
                    self.data_valid = False

            # Debug logging (uncomment if needed)
            # log.info(
            #     f"Lat: {self.lat:.6f}, Lon: {self.lon:.6f}, "
            #     f"Alt: {self.alt:.0f}, Hdg: {self.hdg:.0f}, "
            #     f"Spd: {self.spd:.0f}"
            # )

        # No longer in running state
        log.info("UDP listen thread exiting...")
        # Let the thread exit

    def _request_datarefs(self, subscribe=True):
        """
        Send dataref subscription requests to X-Plane using pre-built
        messages.

        Args:
            subscribe (bool): True to subscribe (freq=1), False to
                              unsubscribe (freq=0)
        """
        messages = (self.subscribe_messages if subscribe
                    else self.unsubscribe_messages)
        udp_port = CFG.flightdata.xplane_udp_port

        for message in messages:
            try:
                self.sock.sendto(message, (UDP_IP, int(udp_port)))
            except Exception as e:
                log.warning(f"Failed to send dataref request: {e}")
                raise

    def _decode_packet(self, data):
        """
        Decode RREF packet from X-Plane.

        Args:
            data (bytes): Raw UDP packet data

        Returns:
            list: List of float values in the order they were
                  requested, or empty list if packet is invalid
        """
        try:
            # Validate packet length
            if len(data) < self.PACKET_HEADER_SIZE:
                log.warning(f"Packet too short: {len(data)} bytes")
                return []

            # Validate header
            header = data[0:4]
            if header != self.RREF_HEADER:
                log.warning(
                    f"Unknown packet header: {binascii.hexlify(header)}"
                )
                return []

            # Extract values
            values_data = data[self.PACKET_HEADER_SIZE:]

            # Validate payload length
            if len(values_data) % self.VALUE_SIZE != 0:
                log.warning(
                    f"Invalid packet length: {len(values_data)} not "
                    f"divisible by {self.VALUE_SIZE}"
                )
                return []

            # Decode all values
            num_values = len(values_data) // self.VALUE_SIZE
            retvalues = []

            for i in range(num_values):
                offset = self.PACKET_HEADER_SIZE + (i * self.VALUE_SIZE)
                singledata = data[offset:offset + self.VALUE_SIZE]

                try:
                    (idx, value) = struct.unpack("<if", singledata)
                    retvalues.append(value)
                except struct.error as e:
                    log.error(
                        f"Struct unpacking error at index {i}: {e}"
                    )
                    return []

            return retvalues

        except Exception as e:
            log.error(f"Unexpected error decoding packet: {e}")
            return []

    # Legacy method for backwards compatibility (if needed externally)
    def RequestDataRefs(self, sock, UDP_PORT=49000, REQ_FREQ=1):
        """
        Legacy method for requesting datarefs.
        Prefer using _request_datarefs() instead.
        """
        self._request_datarefs(subscribe=(REQ_FREQ > 0))

    def DecodePacket(self, data):
        """
        Legacy method for decoding packets.
        Prefer using _decode_packet() instead.

        Returns:
            dict: Dictionary mapping idx to (value, unit, dataref_name)
                  tuples
        """
        values = self._decode_packet(data)
        retvalues = {}
        for idx, value in enumerate(values):
            if idx < len(self.datarefs):
                retvalues[idx] = (
                    value,
                    self.datarefs[idx][1],
                    self.datarefs[idx][0]
                )
        return retvalues

    def get_flight_averages(self) -> Optional[dict]:
        """
        Get 60-second averaged flight data for predictive calculations.

        This provides smoothed values suitable for predicting future
        aircraft position and altitude. Used by the dynamic zoom system
        to determine appropriate zoom levels based on predicted altitude
        at closest approach to tiles.

        Returns:
            dict with keys:
                - 'vertical_speed_fpm': Average climb/descent rate (ft/min)
                - 'heading': Average heading (degrees, 0-360)
                - 'ground_speed_mps': Average ground speed (m/s)
            Returns None if not enough samples are available or if
            not connected to X-Plane.
        """
        return self.flight_averager.get_averages()


dt = DatarefTracker()
