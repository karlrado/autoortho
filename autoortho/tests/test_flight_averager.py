#!/usr/bin/env python3
"""
Unit tests for the FlightDataAverager class.

Tests cover:
- Basic sample addition and averaging
- Circular heading averaging (wraparound handling)
- Vertical speed computation
- Window pruning behavior
- Thread safety
- Edge cases and validation
"""

import pytest
import sys
import os
import time
import math
import threading
from unittest.mock import patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datareftrack import FlightSample, FlightDataAverager


# =============================================================================
# FlightSample Tests
# =============================================================================


class TestFlightSample:
    """Tests for FlightSample dataclass."""

    def test_basic_creation(self):
        """Test creating a FlightSample with valid values."""
        sample = FlightSample(
            timestamp=1000.0,
            lat=47.5,
            lon=-122.3,
            alt_ft=10000,
            hdg=180.0,
            spd=100.0
        )
        assert sample.timestamp == 1000.0
        assert sample.lat == 47.5
        assert sample.lon == -122.3
        assert sample.alt_ft == 10000
        assert sample.hdg == 180.0
        assert sample.spd == 100.0

    def test_negative_values(self):
        """Test that negative values are allowed."""
        sample = FlightSample(
            timestamp=0.0,
            lat=-45.0,
            lon=-180.0,
            alt_ft=-100,  # Below sea level
            hdg=0.0,
            spd=0.0
        )
        assert sample.lat == -45.0
        assert sample.alt_ft == -100


# =============================================================================
# FlightDataAverager Tests - Basic Operations
# =============================================================================


class TestFlightDataAveragerBasic:
    """Tests for basic FlightDataAverager operations."""

    def test_initialization(self):
        """Test that averager starts empty and invalid."""
        averager = FlightDataAverager()
        assert averager.sample_count() == 0
        assert not averager.is_valid()
        assert averager.get_averages() is None

    def test_add_single_sample(self):
        """Test adding a single sample."""
        averager = FlightDataAverager()
        averager.add_sample(lat=47.5, lon=-122.3, alt_ft=10000, hdg=180.0, spd=100.0)
        
        assert averager.sample_count() == 1
        # Still not valid - need MIN_SAMPLES
        assert not averager.is_valid()

    def test_add_minimum_samples(self):
        """Test adding minimum samples to become valid."""
        averager = FlightDataAverager()
        
        # Add MIN_SAMPLES samples
        for i in range(FlightDataAverager.MIN_SAMPLES):
            averager.add_sample(
                lat=47.5, lon=-122.3, alt_ft=10000, 
                hdg=180.0, spd=100.0
            )
            time.sleep(0.01)  # Small delay so timestamps differ
        
        assert averager.sample_count() == FlightDataAverager.MIN_SAMPLES
        assert averager.is_valid()

    def test_clear(self):
        """Test clearing all samples."""
        averager = FlightDataAverager()
        
        # Add samples
        for i in range(5):
            averager.add_sample(
                lat=47.5, lon=-122.3, alt_ft=10000, 
                hdg=180.0, spd=100.0
            )
            time.sleep(0.01)
        
        assert averager.sample_count() > 0
        
        # Clear
        averager.clear()
        
        assert averager.sample_count() == 0
        assert not averager.is_valid()


# =============================================================================
# FlightDataAverager Tests - Ground Speed Averaging
# =============================================================================


class TestFlightDataAveragerGroundSpeed:
    """Tests for ground speed averaging."""

    def test_constant_speed_average(self):
        """Test averaging constant ground speed."""
        averager = FlightDataAverager()
        
        for i in range(5):
            averager.add_sample(
                lat=47.5, lon=-122.3, alt_ft=10000, 
                hdg=180.0, spd=100.0  # Constant 100 m/s
            )
            time.sleep(0.01)
        
        averages = averager.get_averages()
        assert averages is not None
        assert averages['ground_speed_mps'] == pytest.approx(100.0, rel=0.01)

    def test_varying_speed_average(self):
        """Test averaging varying ground speed."""
        averager = FlightDataAverager()
        
        speeds = [80.0, 90.0, 100.0, 110.0, 120.0]  # Average = 100
        for spd in speeds:
            averager.add_sample(
                lat=47.5, lon=-122.3, alt_ft=10000, 
                hdg=180.0, spd=spd
            )
            time.sleep(0.01)
        
        averages = averager.get_averages()
        assert averages is not None
        assert averages['ground_speed_mps'] == pytest.approx(100.0, rel=0.01)


# =============================================================================
# FlightDataAverager Tests - Heading Averaging
# =============================================================================


class TestFlightDataAveragerHeading:
    """Tests for circular heading averaging."""

    def test_constant_heading(self):
        """Test averaging constant heading."""
        averager = FlightDataAverager()
        
        for i in range(5):
            averager.add_sample(
                lat=47.5, lon=-122.3, alt_ft=10000, 
                hdg=90.0,  # Due east
                spd=100.0
            )
            time.sleep(0.01)
        
        averages = averager.get_averages()
        assert averages is not None
        assert averages['heading'] == pytest.approx(90.0, abs=0.1)

    def test_north_heading_average(self):
        """Test averaging headings around north (0/360 wraparound)."""
        averager = FlightDataAverager()
        
        # Headings around north: 350, 355, 0, 5, 10
        # Should average to approximately 0 (north)
        headings = [350.0, 355.0, 0.0, 5.0, 10.0]
        for hdg in headings:
            averager.add_sample(
                lat=47.5, lon=-122.3, alt_ft=10000, 
                hdg=hdg, spd=100.0
            )
            time.sleep(0.01)
        
        averages = averager.get_averages()
        assert averages is not None
        # Should be close to 0 (or 360)
        heading = averages['heading']
        # Handle wraparound: 0 and 360 are equivalent
        if heading > 180:
            heading = heading - 360
        assert heading == pytest.approx(0.0, abs=5.0)

    def test_south_heading_average(self):
        """Test averaging headings around south (180)."""
        averager = FlightDataAverager()
        
        headings = [170.0, 175.0, 180.0, 185.0, 190.0]
        for hdg in headings:
            averager.add_sample(
                lat=47.5, lon=-122.3, alt_ft=10000, 
                hdg=hdg, spd=100.0
            )
            time.sleep(0.01)
        
        averages = averager.get_averages()
        assert averages is not None
        assert averages['heading'] == pytest.approx(180.0, abs=1.0)

    def test_heading_always_positive(self):
        """Test that heading is always in 0-360 range."""
        averager = FlightDataAverager()
        
        for hdg in [0.0, 90.0, 180.0, 270.0, 359.0]:
            averager.add_sample(
                lat=47.5, lon=-122.3, alt_ft=10000, 
                hdg=hdg, spd=100.0
            )
            time.sleep(0.01)
        
        averages = averager.get_averages()
        assert averages is not None
        assert 0 <= averages['heading'] < 360


# =============================================================================
# FlightDataAverager Tests - Vertical Speed
# =============================================================================


class TestFlightDataAveragerVerticalSpeed:
    """Tests for vertical speed computation."""

    def test_level_flight(self):
        """Test vertical speed is near zero in level flight."""
        averager = FlightDataAverager()
        
        for i in range(5):
            averager.add_sample(
                lat=47.5, lon=-122.3, 
                alt_ft=10000,  # Constant altitude
                hdg=180.0, spd=100.0
            )
            time.sleep(0.05)  # 50ms between samples
        
        averages = averager.get_averages()
        assert averages is not None
        # Should be approximately 0 fpm
        assert averages['vertical_speed_fpm'] == pytest.approx(0.0, abs=10.0)

    def test_climbing(self):
        """Test vertical speed calculation during climb."""
        averager = FlightDataAverager()
        
        # Simulate climb: 1000 ft over 1 second = 60000 fpm
        # But with our sample rate, we need appropriate time deltas
        base_alt = 10000
        for i in range(5):
            averager.add_sample(
                lat=47.5, lon=-122.3, 
                alt_ft=base_alt + (i * 100),  # +100 ft per sample
                hdg=180.0, spd=100.0
            )
            time.sleep(0.1)  # 100ms between samples
        
        averages = averager.get_averages()
        assert averages is not None
        # Should be positive (climbing)
        assert averages['vertical_speed_fpm'] > 0

    def test_descending(self):
        """Test vertical speed calculation during descent."""
        averager = FlightDataAverager()
        
        base_alt = 10000
        for i in range(5):
            averager.add_sample(
                lat=47.5, lon=-122.3, 
                alt_ft=base_alt - (i * 100),  # -100 ft per sample
                hdg=180.0, spd=100.0
            )
            time.sleep(0.1)
        
        averages = averager.get_averages()
        assert averages is not None
        # Should be negative (descending)
        assert averages['vertical_speed_fpm'] < 0

    def test_vertical_speed_getter(self):
        """Test the dedicated vertical speed getter method."""
        averager = FlightDataAverager()
        
        # Before valid
        assert averager.get_vertical_speed_fpm() is None
        
        # Add samples
        for i in range(5):
            averager.add_sample(
                lat=47.5, lon=-122.3, alt_ft=10000, 
                hdg=180.0, spd=100.0
            )
            time.sleep(0.01)
        
        vs = averager.get_vertical_speed_fpm()
        assert vs is not None
        assert isinstance(vs, float)


# =============================================================================
# FlightDataAverager Tests - Window Pruning
# =============================================================================


class TestFlightDataAveragerWindowPruning:
    """Tests for sample window pruning behavior."""

    def test_old_samples_pruned(self):
        """Test that samples older than WINDOW_SEC are pruned."""
        averager = FlightDataAverager()
        
        # Temporarily reduce window for faster testing
        original_window = FlightDataAverager.WINDOW_SEC
        FlightDataAverager.WINDOW_SEC = 0.5  # 500ms window
        
        try:
            # Add initial samples
            for i in range(3):
                averager.add_sample(
                    lat=47.5, lon=-122.3, alt_ft=10000, 
                    hdg=180.0, spd=100.0
                )
            
            initial_count = averager.sample_count()
            assert initial_count == 3
            
            # Wait for window to expire
            time.sleep(0.6)
            
            # Add new sample - should trigger pruning
            averager.add_sample(
                lat=47.5, lon=-122.3, alt_ft=10000, 
                hdg=180.0, spd=100.0
            )
            
            # Old samples should be pruned
            # Only the new sample should remain
            assert averager.sample_count() <= 2
            
        finally:
            FlightDataAverager.WINDOW_SEC = original_window

    def test_get_window_duration(self):
        """Test window duration calculation."""
        averager = FlightDataAverager()
        
        # Empty
        assert averager.get_window_duration() == 0.0
        
        # One sample
        averager.add_sample(
            lat=47.5, lon=-122.3, alt_ft=10000, 
            hdg=180.0, spd=100.0
        )
        assert averager.get_window_duration() == 0.0
        
        # Two samples with delay
        time.sleep(0.1)
        averager.add_sample(
            lat=47.5, lon=-122.3, alt_ft=10000, 
            hdg=180.0, spd=100.0
        )
        
        duration = averager.get_window_duration()
        assert duration >= 0.09  # At least 90ms


# =============================================================================
# FlightDataAverager Tests - Thread Safety
# =============================================================================


class TestFlightDataAveragerThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_reads_and_writes(self):
        """Test concurrent read/write access."""
        averager = FlightDataAverager()
        errors = []
        
        # Pre-populate with some samples
        for i in range(5):
            averager.add_sample(
                lat=47.5, lon=-122.3, alt_ft=10000, 
                hdg=180.0, spd=100.0
            )
            time.sleep(0.01)
        
        def writer():
            try:
                for i in range(50):
                    averager.add_sample(
                        lat=47.5 + i * 0.001, lon=-122.3, 
                        alt_ft=10000 + i, hdg=180.0, spd=100.0
                    )
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        def reader():
            try:
                for _ in range(100):
                    averager.get_averages()
                    averager.is_valid()
                    averager.sample_count()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        threads.append(threading.Thread(target=writer))
        for _ in range(3):
            threads.append(threading.Thread(target=reader))
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Thread safety errors: {errors}"

    def test_clear_during_access(self):
        """Test clearing while other threads are accessing."""
        averager = FlightDataAverager()
        errors = []
        
        def writer():
            try:
                for i in range(50):
                    averager.add_sample(
                        lat=47.5, lon=-122.3, alt_ft=10000, 
                        hdg=180.0, spd=100.0
                    )
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        def clearer():
            try:
                for _ in range(10):
                    time.sleep(0.005)
                    averager.clear()
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=clearer),
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0


# =============================================================================
# FlightDataAverager Tests - Edge Cases
# =============================================================================


class TestFlightDataAveragerEdgeCases:
    """Tests for edge cases."""

    def test_minimum_time_delta(self):
        """Test handling of very small time deltas."""
        averager = FlightDataAverager()
        
        # Add samples with no time delay (same timestamp effectively)
        for i in range(5):
            averager.add_sample(
                lat=47.5, lon=-122.3, 
                alt_ft=10000 + i * 100,  # Different altitudes
                hdg=180.0, spd=100.0
            )
            # No sleep - very small time delta
        
        # Should still work, vertical speed might be 0 or very large
        # depending on timing, but shouldn't crash
        averages = averager.get_averages()
        # May or may not be valid depending on MIN_SAMPLES and timing

    def test_zero_speed(self):
        """Test handling of zero ground speed (parked/hover)."""
        averager = FlightDataAverager()
        
        for i in range(5):
            averager.add_sample(
                lat=47.5, lon=-122.3, alt_ft=0, 
                hdg=180.0, spd=0.0  # Not moving
            )
            time.sleep(0.01)
        
        averages = averager.get_averages()
        assert averages is not None
        assert averages['ground_speed_mps'] == 0.0

    def test_extreme_altitude(self):
        """Test handling of extreme altitudes."""
        averager = FlightDataAverager()
        
        # Very high altitude (FL450 = 45000 ft)
        for i in range(5):
            averager.add_sample(
                lat=47.5, lon=-122.3, alt_ft=45000, 
                hdg=180.0, spd=250.0
            )
            time.sleep(0.01)
        
        averages = averager.get_averages()
        assert averages is not None

    def test_negative_altitude(self):
        """Test handling of negative altitude (below sea level)."""
        averager = FlightDataAverager()
        
        # Death Valley is ~-282 ft
        for i in range(5):
            averager.add_sample(
                lat=36.2, lon=-116.8, alt_ft=-200, 
                hdg=90.0, spd=50.0
            )
            time.sleep(0.01)
        
        averages = averager.get_averages()
        assert averages is not None


# =============================================================================
# Integration with DatarefTracker
# =============================================================================


class TestFlightDataAveragerIntegration:
    """Tests for integration with DatarefTracker."""

    def test_datareftracker_has_averager(self):
        """Test that DatarefTracker has flight_averager attribute."""
        from datareftrack import DatarefTracker
        
        dt = DatarefTracker()
        assert hasattr(dt, 'flight_averager')
        assert isinstance(dt.flight_averager, FlightDataAverager)

    def test_datareftracker_get_flight_averages(self):
        """Test that DatarefTracker exposes get_flight_averages method."""
        from datareftrack import DatarefTracker
        
        dt = DatarefTracker()
        assert hasattr(dt, 'get_flight_averages')
        
        # Should return None when no samples
        result = dt.get_flight_averages()
        assert result is None

    def test_flight_averages_after_manual_samples(self):
        """Test flight averages after manually adding samples."""
        from datareftrack import DatarefTracker
        
        dt = DatarefTracker()
        
        # Manually add samples to the averager
        for i in range(5):
            dt.flight_averager.add_sample(
                lat=47.5, lon=-122.3, alt_ft=10000, 
                hdg=180.0, spd=100.0
            )
            time.sleep(0.01)
        
        # Should now return valid averages
        result = dt.get_flight_averages()
        assert result is not None
        assert 'vertical_speed_fpm' in result
        assert 'heading' in result
        assert 'ground_speed_mps' in result


# =============================================================================
# Realistic Flight Scenarios
# =============================================================================


class TestFlightDataAveragerScenarios:
    """Tests simulating realistic flight scenarios."""

    def test_takeoff_scenario(self):
        """Simulate takeoff with increasing altitude and speed."""
        averager = FlightDataAverager()
        
        # Takeoff roll to climb
        altitudes = [0, 0, 100, 500, 1000, 2000, 3000]
        speeds = [0, 20, 50, 80, 100, 120, 140]
        
        for alt, spd in zip(altitudes, speeds):
            averager.add_sample(
                lat=47.5, lon=-122.3, alt_ft=alt, 
                hdg=270.0, spd=spd
            )
            time.sleep(0.05)
        
        averages = averager.get_averages()
        assert averages is not None
        
        # Should show positive climb rate
        assert averages['vertical_speed_fpm'] > 0
        # Heading should be west
        assert averages['heading'] == pytest.approx(270.0, abs=5.0)

    def test_cruise_scenario(self):
        """Simulate stable cruise flight."""
        averager = FlightDataAverager()
        
        # Stable cruise at FL350
        for i in range(10):
            averager.add_sample(
                lat=47.5 + i * 0.01,  # Moving north
                lon=-122.3, 
                alt_ft=35000 + (i % 2) * 50,  # Minor altitude variations
                hdg=0.0 + (i % 2),  # Minor heading variations
                spd=250.0
            )
            time.sleep(0.05)
        
        averages = averager.get_averages()
        assert averages is not None
        
        # Vertical speed should be near zero
        assert abs(averages['vertical_speed_fpm']) < 1000
        # Speed should be around 250 m/s
        assert averages['ground_speed_mps'] == pytest.approx(250.0, abs=1.0)

    def test_approach_scenario(self):
        """Simulate approach with descending altitude."""
        averager = FlightDataAverager()
        
        # Descending approach
        altitudes = [5000, 4500, 4000, 3500, 3000, 2500, 2000]
        speeds = [140, 130, 120, 120, 110, 100, 90]
        
        for alt, spd in zip(altitudes, speeds):
            averager.add_sample(
                lat=47.5, lon=-122.3, alt_ft=alt, 
                hdg=90.0, spd=spd
            )
            time.sleep(0.05)
        
        averages = averager.get_averages()
        assert averages is not None
        
        # Should show negative climb rate (descent)
        assert averages['vertical_speed_fpm'] < 0

    def test_holding_pattern(self):
        """Simulate holding pattern with changing heading."""
        averager = FlightDataAverager()
        
        # Holding pattern - full 360 turn
        headings = [0, 45, 90, 135, 180, 225, 270, 315, 360]
        
        for hdg in headings:
            averager.add_sample(
                lat=47.5, lon=-122.3, alt_ft=8000, 
                hdg=hdg % 360, spd=120.0
            )
            time.sleep(0.05)
        
        averages = averager.get_averages()
        assert averages is not None
        # In a perfect 360, all directions average out
        # The heading average depends on implementation

