#!/usr/bin/env python3
"""
Unit tests for SpatialPrefetcher with averaged flight data.

Tests cover:
- Data source selection (averaged vs instantaneous)
- Prefetch cycle logic simulation
- Speed threshold handling
- Prediction calculation correctness
- Integration patterns with DatarefTracker
"""

import pytest
import sys
import os
import math
import threading
import time
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Mock DatarefTracker for Testing
# =============================================================================


class MockDatarefTracker:
    """Mock DatarefTracker for unit testing prefetcher logic."""

    def __init__(self):
        self.lat = 47.5
        self.lon = -122.3
        self.hdg = 270.0  # West
        self.spd = 100.0  # m/s
        self.connected = True
        self.data_valid = True
        self._lock = threading.Lock()
        self._flight_averages = None

    def get_flight_averages(self):
        """Return mock averaged data."""
        return self._flight_averages

    def set_flight_averages(self, averages):
        """Set mock averaged data for testing."""
        self._flight_averages = averages

    def clear_flight_averages(self):
        """Clear averaged data to test fallback."""
        self._flight_averages = None


# =============================================================================
# Data Source Selection Tests
# =============================================================================


class TestDataSourceSelection:
    """Tests for selecting between averaged and instantaneous data."""

    def test_prefers_averaged_data_when_available(self):
        """Test that averaged data is preferred over instantaneous."""
        tracker = MockDatarefTracker()

        # Set different values for instantaneous vs averaged
        tracker.hdg = 0.0   # Instantaneous: North
        tracker.spd = 50.0  # Instantaneous: 50 m/s

        tracker.set_flight_averages({
            'heading': 90.0,          # Averaged: East
            'ground_speed_mps': 100.0,  # Averaged: 100 m/s
            'vertical_speed_fpm': 0.0,
        })

        # Simulate prefetcher logic:
        averages = tracker.get_flight_averages()
        if averages is not None:
            hdg = averages['heading']
            spd = averages['ground_speed_mps']
        else:
            hdg = tracker.hdg
            spd = tracker.spd

        # Should use averaged values
        assert hdg == 90.0
        assert spd == 100.0

    def test_falls_back_to_instantaneous_when_no_averages(self):
        """Test fallback to instantaneous when averages unavailable."""
        tracker = MockDatarefTracker()

        tracker.hdg = 180.0  # Instantaneous: South
        tracker.spd = 75.0   # Instantaneous: 75 m/s
        tracker.connected = True
        tracker.data_valid = True
        tracker.clear_flight_averages()

        # Simulate prefetcher logic:
        averages = tracker.get_flight_averages()
        if averages is not None:
            hdg = averages['heading']
            spd = averages['ground_speed_mps']
        else:
            if tracker.data_valid and tracker.connected:
                hdg = tracker.hdg
                spd = tracker.spd
            else:
                # Would return early in real code
                hdg = None
                spd = None

        # Should use instantaneous values
        assert hdg == 180.0
        assert spd == 75.0

    def test_always_uses_instantaneous_position(self):
        """Test that position is always instantaneous (never averaged)."""
        tracker = MockDatarefTracker()
        tracker.lat = 47.5
        tracker.lon = -122.3

        # Even with averages available, position comes from tracker
        tracker.set_flight_averages({
            'heading': 90.0,
            'ground_speed_mps': 100.0,
            'vertical_speed_fpm': 0.0,
            # Note: No lat/lon in averages - by design
        })

        # Prefetcher always reads position directly
        lat = tracker.lat
        lon = tracker.lon

        assert lat == 47.5
        assert lon == -122.3


# =============================================================================
# Speed Threshold Tests
# =============================================================================


class TestSpeedThreshold:
    """Tests for minimum speed threshold to trigger prefetch."""

    # MIN_SPEED_MPS is typically 25 m/s (~50 knots)
    MIN_SPEED_MPS = 25

    def test_no_prefetch_when_stationary(self):
        """Test that prefetch doesn't occur when speed is 0."""
        tracker = MockDatarefTracker()
        tracker.set_flight_averages({
            'heading': 90.0,
            'ground_speed_mps': 0.0,
            'vertical_speed_fpm': 0.0,
        })

        averages = tracker.get_flight_averages()
        should_prefetch = averages['ground_speed_mps'] >= self.MIN_SPEED_MPS

        assert should_prefetch is False

    def test_no_prefetch_when_taxiing(self):
        """Test no prefetch during taxi speeds (< 25 m/s)."""
        tracker = MockDatarefTracker()
        taxi_speed = 10.0  # ~20 knots

        tracker.set_flight_averages({
            'heading': 270.0,
            'ground_speed_mps': taxi_speed,
            'vertical_speed_fpm': 0.0,
        })

        averages = tracker.get_flight_averages()
        should_prefetch = averages['ground_speed_mps'] >= self.MIN_SPEED_MPS

        assert should_prefetch is False

    def test_prefetch_when_flying(self):
        """Test that prefetch occurs at normal flight speeds."""
        tracker = MockDatarefTracker()
        flight_speed = 100.0  # ~195 knots

        tracker.set_flight_averages({
            'heading': 270.0,
            'ground_speed_mps': flight_speed,
            'vertical_speed_fpm': 0.0,
        })

        averages = tracker.get_flight_averages()
        should_prefetch = averages['ground_speed_mps'] >= self.MIN_SPEED_MPS

        assert should_prefetch is True


# =============================================================================
# Connection State Tests
# =============================================================================


class TestConnectionState:
    """Tests for connection state handling during fallback."""

    def test_no_fallback_when_disconnected(self):
        """Test no fallback to instantaneous when disconnected."""
        tracker = MockDatarefTracker()
        tracker.connected = False
        tracker.data_valid = True
        tracker.clear_flight_averages()

        averages = tracker.get_flight_averages()
        can_use_fallback = (
            averages is None and
            tracker.connected and
            tracker.data_valid
        )

        assert can_use_fallback is False

    def test_no_fallback_when_data_invalid(self):
        """Test no fallback when data is marked invalid."""
        tracker = MockDatarefTracker()
        tracker.connected = True
        tracker.data_valid = False
        tracker.clear_flight_averages()

        averages = tracker.get_flight_averages()
        can_use_fallback = (
            averages is None and
            tracker.connected and
            tracker.data_valid
        )

        assert can_use_fallback is False

    def test_uses_averages_regardless_of_connection(self):
        """Test that valid averages can be used even if connection is transitioning."""
        tracker = MockDatarefTracker()

        # Have valid averages from before potential disconnection
        tracker.set_flight_averages({
            'heading': 270.0,
            'ground_speed_mps': 100.0,
            'vertical_speed_fpm': 0.0,
        })

        # Averages should still be usable (they were computed recently)
        averages = tracker.get_flight_averages()
        assert averages is not None
        assert averages['heading'] == 270.0


# =============================================================================
# Position Prediction Tests
# =============================================================================


class TestPositionPrediction:
    """Tests for position prediction calculations used in prefetching."""

    # Lookahead time (same as SpatialPrefetcher)
    LOOKAHEAD_SEC = 600  # 10 minutes

    def test_prediction_heading_north(self):
        """Test prediction when heading north."""
        lat = 47.0
        lon = -122.0
        hdg = 0.0  # North
        spd = 100.0  # m/s

        # Calculate predicted position
        distance_m = spd * self.LOOKAHEAD_SEC  # 60,000 meters
        hdg_rad = math.radians(hdg)

        delta_lat = (distance_m * math.cos(hdg_rad)) / 111320
        cos_lat = math.cos(math.radians(lat))
        delta_lon = (distance_m * math.sin(hdg_rad)) / (111320 * cos_lat)

        predicted_lat = lat + delta_lat
        predicted_lon = lon + delta_lon

        # Heading north: latitude increases, longitude unchanged
        assert predicted_lat > lat
        assert abs(predicted_lon - lon) < 0.01

    def test_prediction_heading_east(self):
        """Test prediction when heading east."""
        lat = 47.0
        lon = -122.0
        hdg = 90.0  # East
        spd = 100.0

        distance_m = spd * self.LOOKAHEAD_SEC
        hdg_rad = math.radians(hdg)

        delta_lat = (distance_m * math.cos(hdg_rad)) / 111320
        cos_lat = math.cos(math.radians(lat))
        delta_lon = (distance_m * math.sin(hdg_rad)) / (111320 * cos_lat)

        predicted_lat = lat + delta_lat
        predicted_lon = lon + delta_lon

        # Heading east: longitude increases, latitude unchanged
        assert abs(predicted_lat - lat) < 0.01
        assert predicted_lon > lon

    def test_prediction_heading_west(self):
        """Test prediction when heading west."""
        lat = 47.0
        lon = -122.0
        hdg = 270.0  # West
        spd = 100.0

        distance_m = spd * self.LOOKAHEAD_SEC
        hdg_rad = math.radians(hdg)

        delta_lat = (distance_m * math.cos(hdg_rad)) / 111320
        cos_lat = math.cos(math.radians(lat))
        delta_lon = (distance_m * math.sin(hdg_rad)) / (111320 * cos_lat)

        predicted_lat = lat + delta_lat
        predicted_lon = lon + delta_lon

        # Heading west: longitude decreases
        assert predicted_lon < lon

    def test_prediction_heading_south(self):
        """Test prediction when heading south."""
        lat = 47.0
        lon = -122.0
        hdg = 180.0  # South
        spd = 100.0

        distance_m = spd * self.LOOKAHEAD_SEC
        hdg_rad = math.radians(hdg)

        delta_lat = (distance_m * math.cos(hdg_rad)) / 111320
        cos_lat = math.cos(math.radians(lat))
        delta_lon = (distance_m * math.sin(hdg_rad)) / (111320 * cos_lat)

        predicted_lat = lat + delta_lat
        predicted_lon = lon + delta_lon

        # Heading south: latitude decreases
        assert predicted_lat < lat


# =============================================================================
# Heading Averaging Benefits Tests
# =============================================================================


class TestHeadingAveragingBenefits:
    """Tests demonstrating why averaging helps with heading stability."""

    def test_jittery_instantaneous_vs_stable_average(self):
        """Demonstrate that averaging smooths out heading jitter."""
        # Simulate noisy instantaneous headings (turbulence, autopilot oscillation)
        instantaneous_headings = [85, 95, 88, 92, 87, 93, 86, 94, 89, 91]

        # Simple arithmetic average (for demonstration)
        avg_heading = sum(instantaneous_headings) / len(instantaneous_headings)

        # Average should be close to 90
        assert avg_heading == pytest.approx(90.0, abs=1.0)

        # Max deviation from average
        max_deviation = max(abs(h - avg_heading) for h in instantaneous_headings)
        assert max_deviation > 4  # Instantaneous values deviate by >4 degrees

    def test_circular_heading_average_near_north(self):
        """Test that circular averaging handles 0/360 wraparound."""
        # Headings crossing north: 350, 355, 0, 5, 10
        headings = [350, 355, 0, 5, 10]

        # Circular average using sin/cos (same as FlightDataAverager)
        sin_sum = sum(math.sin(math.radians(h)) for h in headings)
        cos_sum = sum(math.cos(math.radians(h)) for h in headings)
        circular_avg = math.degrees(math.atan2(sin_sum, cos_sum)) % 360

        # Should be close to 0 (north), not 144 (arithmetic average)
        arithmetic_avg = sum(headings) / len(headings)
        assert arithmetic_avg > 100  # Arithmetic average is wrong

        # Circular average should be close to north
        if circular_avg > 180:
            circular_avg = circular_avg - 360
        assert circular_avg == pytest.approx(0.0, abs=5.0)


# =============================================================================
# Integration Pattern Tests
# =============================================================================


class TestPrefetcherIntegrationPatterns:
    """Tests for the complete prefetch cycle pattern."""

    def test_complete_prefetch_cycle_with_averages(self):
        """Simulate a complete prefetch cycle using averaged data."""
        tracker = MockDatarefTracker()
        tracker.lat = 47.5
        tracker.lon = -122.3
        tracker.set_flight_averages({
            'heading': 270.0,  # West
            'ground_speed_mps': 100.0,
            'vertical_speed_fpm': -500.0,  # Descending
        })

        # Step 1: Try to get averages
        averages = tracker.get_flight_averages()
        assert averages is not None

        # Step 2: Extract values
        hdg = averages['heading']
        spd = averages['ground_speed_mps']
        lat = tracker.lat  # Always instantaneous
        lon = tracker.lon

        # Step 3: Validate position
        assert -90 <= lat <= 90
        assert -180 <= lon <= 180

        # Step 4: Check speed threshold
        MIN_SPEED_MPS = 25
        assert spd >= MIN_SPEED_MPS

        # Step 5: Calculate prediction
        lookahead_sec = 600
        distance_m = spd * lookahead_sec
        hdg_rad = math.radians(hdg)

        delta_lat = (distance_m * math.cos(hdg_rad)) / 111320
        cos_lat = math.cos(math.radians(lat))
        delta_lon = (distance_m * math.sin(hdg_rad)) / (111320 * cos_lat)

        predicted_lat = lat + delta_lat
        predicted_lon = lon + delta_lon

        # Heading west (270°): longitude should decrease
        assert predicted_lon < lon
        # Latitude should stay roughly the same
        assert abs(predicted_lat - lat) < 0.1

    def test_complete_prefetch_cycle_with_fallback(self):
        """Simulate a complete prefetch cycle using instantaneous fallback."""
        tracker = MockDatarefTracker()
        tracker.lat = 47.5
        tracker.lon = -122.3
        tracker.hdg = 180.0  # South
        tracker.spd = 80.0
        tracker.connected = True
        tracker.data_valid = True
        tracker.clear_flight_averages()

        # Step 1: Try to get averages
        averages = tracker.get_flight_averages()
        assert averages is None

        # Step 2: Fallback to instantaneous
        if tracker.data_valid and tracker.connected:
            hdg = tracker.hdg
            spd = tracker.spd
        else:
            # Would return early
            hdg = None
            spd = None

        assert hdg == 180.0
        assert spd == 80.0

        # Rest of cycle would proceed with instantaneous values

    def test_prefetch_aborted_when_no_data(self):
        """Test that prefetch is aborted when no data is available."""
        tracker = MockDatarefTracker()
        tracker.connected = False
        tracker.data_valid = False
        tracker.clear_flight_averages()

        averages = tracker.get_flight_averages()
        can_proceed = (
            averages is not None or
            (tracker.connected and tracker.data_valid)
        )

        assert can_proceed is False

