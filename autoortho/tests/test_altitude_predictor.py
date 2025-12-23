#!/usr/bin/env python3
"""
Unit tests for the altitude prediction module.

Tests cover:
- Haversine distance calculations
- Bearing calculations
- Altitude prediction for approaching/receding tiles
- Edge cases (stationary, directly overhead, etc.)
- Integration with tile coordinate conversion
"""

import pytest
import sys
import os
import math

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.altitude_predictor import (
    predict_altitude_at_closest_approach,
    get_tile_center_coords,
    _haversine_distance,
    _bearing_to_point,
    _angular_difference,
    EARTH_RADIUS_M,
    MIN_ALTITUDE_FT,
    MAX_ALTITUDE_FT,
)


# =============================================================================
# Haversine Distance Tests
# =============================================================================


class TestHaversineDistance:
    """Tests for _haversine_distance function."""

    def test_same_point(self):
        """Distance from a point to itself should be zero."""
        dist = _haversine_distance(47.5, -122.3, 47.5, -122.3)
        assert dist == pytest.approx(0.0, abs=0.1)

    def test_known_distance_seattle_portland(self):
        """Test with known distance: Seattle to Portland (~233 km)."""
        # Seattle: 47.6062, -122.3321
        # Portland: 45.5152, -122.6784
        dist = _haversine_distance(47.6062, -122.3321, 45.5152, -122.6784)
        # Approximately 233 km
        assert dist == pytest.approx(233_000, rel=0.05)

    def test_known_distance_short(self):
        """Test short distance (1 degree latitude ≈ 111km)."""
        dist = _haversine_distance(47.0, -122.0, 48.0, -122.0)
        # 1 degree latitude ≈ 111.32 km
        assert dist == pytest.approx(111_320, rel=0.01)

    def test_equator_longitude(self):
        """Test longitude distance at equator (1 degree ≈ 111km)."""
        dist = _haversine_distance(0.0, 0.0, 0.0, 1.0)
        assert dist == pytest.approx(111_320, rel=0.01)

    def test_symmetry(self):
        """Distance A->B should equal B->A."""
        dist_ab = _haversine_distance(47.5, -122.3, 45.5, -122.7)
        dist_ba = _haversine_distance(45.5, -122.7, 47.5, -122.3)
        assert dist_ab == pytest.approx(dist_ba, rel=0.0001)


# =============================================================================
# Bearing Tests
# =============================================================================


class TestBearingToPoint:
    """Tests for _bearing_to_point function."""

    def test_due_north(self):
        """Bearing to point directly north should be ~0°."""
        bearing = _bearing_to_point(47.0, -122.0, 48.0, -122.0)
        assert bearing == pytest.approx(0.0, abs=1.0)

    def test_due_east(self):
        """Bearing to point directly east should be ~90°."""
        bearing = _bearing_to_point(47.0, -122.0, 47.0, -121.0)
        assert bearing == pytest.approx(90.0, abs=1.0)

    def test_due_south(self):
        """Bearing to point directly south should be ~180°."""
        bearing = _bearing_to_point(47.0, -122.0, 46.0, -122.0)
        assert bearing == pytest.approx(180.0, abs=1.0)

    def test_due_west(self):
        """Bearing to point directly west should be ~270°."""
        bearing = _bearing_to_point(47.0, -122.0, 47.0, -123.0)
        assert bearing == pytest.approx(270.0, abs=1.0)

    def test_bearing_range(self):
        """Bearing should always be in 0-360 range."""
        test_cases = [
            (47.0, -122.0, 48.0, -121.0),  # NE
            (47.0, -122.0, 46.0, -121.0),  # SE
            (47.0, -122.0, 46.0, -123.0),  # SW
            (47.0, -122.0, 48.0, -123.0),  # NW
        ]
        for lat1, lon1, lat2, lon2 in test_cases:
            bearing = _bearing_to_point(lat1, lon1, lat2, lon2)
            assert 0 <= bearing < 360


# =============================================================================
# Angular Difference Tests
# =============================================================================


class TestAngularDifference:
    """Tests for _angular_difference function."""

    def test_same_heading(self):
        """Difference between same heading should be 0."""
        assert _angular_difference(90.0, 90.0) == 0.0

    def test_opposite_heading(self):
        """Difference between opposite headings should be 180."""
        assert _angular_difference(0.0, 180.0) == 180.0
        assert _angular_difference(90.0, 270.0) == 180.0

    def test_wraparound(self):
        """Test 360/0 wraparound."""
        # 350 to 10 is 20 degrees, not 340
        assert _angular_difference(350.0, 10.0) == 20.0
        assert _angular_difference(10.0, 350.0) == 20.0

    def test_right_angle(self):
        """90 degree differences."""
        assert _angular_difference(0.0, 90.0) == 90.0
        assert _angular_difference(270.0, 0.0) == 90.0


# =============================================================================
# Altitude Prediction Tests - Basic Scenarios
# =============================================================================


class TestAltitudePredictionBasic:
    """Tests for basic altitude prediction scenarios."""

    def test_heading_directly_to_tile(self):
        """Aircraft heading directly toward tile."""
        # Aircraft at 10000 ft, level flight, heading north to tile north
        alt, approaching = predict_altitude_at_closest_approach(
            aircraft_lat=47.0,
            aircraft_lon=-122.0,
            aircraft_alt_ft=10000.0,
            aircraft_hdg=0.0,  # North
            aircraft_speed_mps=100.0,  # ~200 knots
            vertical_speed_fpm=0.0,  # Level
            tile_lat=47.1,  # North of aircraft
            tile_lon=-122.0
        )
        assert approaching is True
        # Level flight, altitude should be same
        assert alt == pytest.approx(10000.0, rel=0.01)

    def test_climbing_to_tile(self):
        """Aircraft climbing toward tile."""
        alt, approaching = predict_altitude_at_closest_approach(
            aircraft_lat=47.0,
            aircraft_lon=-122.0,
            aircraft_alt_ft=5000.0,
            aircraft_hdg=0.0,  # North
            aircraft_speed_mps=100.0,
            vertical_speed_fpm=1000.0,  # Climbing 1000 fpm
            tile_lat=47.1,  # ~11km north
            tile_lon=-122.0
        )
        assert approaching is True
        # Should predict higher altitude
        assert alt > 5000.0

    def test_descending_to_tile(self):
        """Aircraft descending toward tile."""
        alt, approaching = predict_altitude_at_closest_approach(
            aircraft_lat=47.0,
            aircraft_lon=-122.0,
            aircraft_alt_ft=10000.0,
            aircraft_hdg=0.0,  # North
            aircraft_speed_mps=100.0,
            vertical_speed_fpm=-1000.0,  # Descending 1000 fpm
            tile_lat=47.1,  # North
            tile_lon=-122.0
        )
        assert approaching is True
        # Should predict lower altitude
        assert alt < 10000.0

    def test_moving_away_from_tile(self):
        """Aircraft heading away from tile."""
        alt, approaching = predict_altitude_at_closest_approach(
            aircraft_lat=47.0,
            aircraft_lon=-122.0,
            aircraft_alt_ft=10000.0,
            aircraft_hdg=180.0,  # South (away from tile)
            aircraft_speed_mps=100.0,
            vertical_speed_fpm=-500.0,  # Descending
            tile_lat=47.1,  # North (behind us)
            tile_lon=-122.0
        )
        assert approaching is False
        # Should return current altitude when moving away
        assert alt == 10000.0


# =============================================================================
# Altitude Prediction Tests - Edge Cases
# =============================================================================


class TestAltitudePredictionEdgeCases:
    """Tests for edge cases in altitude prediction."""

    def test_stationary_aircraft(self):
        """Aircraft not moving (speed = 0)."""
        alt, approaching = predict_altitude_at_closest_approach(
            aircraft_lat=47.0,
            aircraft_lon=-122.0,
            aircraft_alt_ft=10000.0,
            aircraft_hdg=0.0,
            aircraft_speed_mps=0.0,  # Not moving
            vertical_speed_fpm=0.0,
            tile_lat=47.1,
            tile_lon=-122.0
        )
        # Not approaching (speed < MIN_APPROACH_SPEED_MPS)
        assert approaching is False
        assert alt == 10000.0

    def test_very_slow_aircraft(self):
        """Aircraft moving very slowly."""
        alt, approaching = predict_altitude_at_closest_approach(
            aircraft_lat=47.0,
            aircraft_lon=-122.0,
            aircraft_alt_ft=10000.0,
            aircraft_hdg=0.0,
            aircraft_speed_mps=0.5,  # Very slow
            vertical_speed_fpm=0.0,
            tile_lat=47.1,
            tile_lon=-122.0
        )
        # approach_speed would be 0.5 * cos(0) = 0.5 < MIN_APPROACH_SPEED_MPS
        assert approaching is False

    def test_perpendicular_to_tile(self):
        """Aircraft heading perpendicular to tile (90° angle)."""
        alt, approaching = predict_altitude_at_closest_approach(
            aircraft_lat=47.0,
            aircraft_lon=-122.0,
            aircraft_alt_ft=10000.0,
            aircraft_hdg=90.0,  # East
            aircraft_speed_mps=100.0,
            vertical_speed_fpm=0.0,
            tile_lat=47.1,  # North
            tile_lon=-122.0
        )
        # At exactly 90° angle difference:
        # - angle_diff = 90, which is NOT > 90, so we proceed
        # - approach_speed = 100 * cos(90°) = 0 m/s
        # - This is below MIN_APPROACH_SPEED_MPS (1.0 m/s)
        # Therefore: not approaching, returns current altitude
        assert approaching is False
        assert alt == 10000.0

    def test_aircraft_at_tile(self):
        """Aircraft already at the tile location."""
        alt, approaching = predict_altitude_at_closest_approach(
            aircraft_lat=47.0,
            aircraft_lon=-122.0,
            aircraft_alt_ft=10000.0,
            aircraft_hdg=0.0,
            aircraft_speed_mps=100.0,
            vertical_speed_fpm=0.0,
            tile_lat=47.0,  # Same location
            tile_lon=-122.0
        )
        # Distance < 100m, should return current altitude
        assert approaching is True
        assert alt == 10000.0

    def test_altitude_clamping_high(self):
        """Predicted altitude should be clamped to max."""
        # Climbing for a long time to a distant tile
        alt, approaching = predict_altitude_at_closest_approach(
            aircraft_lat=47.0,
            aircraft_lon=-122.0,
            aircraft_alt_ft=50000.0,
            aircraft_hdg=0.0,
            aircraft_speed_mps=50.0,  # Slow
            vertical_speed_fpm=5000.0,  # Very fast climb
            tile_lat=48.0,  # ~111km away
            tile_lon=-122.0
        )
        # Predicted altitude would be very high, should be clamped
        assert alt <= MAX_ALTITUDE_FT

    def test_altitude_clamping_low(self):
        """Predicted altitude should be clamped to min."""
        alt, approaching = predict_altitude_at_closest_approach(
            aircraft_lat=47.0,
            aircraft_lon=-122.0,
            aircraft_alt_ft=5000.0,
            aircraft_hdg=0.0,
            aircraft_speed_mps=50.0,
            vertical_speed_fpm=-10000.0,  # Extreme descent
            tile_lat=48.0,  # Far away
            tile_lon=-122.0
        )
        # Predicted altitude would be very negative, should be clamped
        assert alt >= MIN_ALTITUDE_FT


# =============================================================================
# Altitude Prediction Tests - Angular Scenarios
# =============================================================================


class TestAltitudePredictionAngles:
    """Tests for various heading/bearing angle scenarios."""

    def test_45_degree_approach(self):
        """Aircraft heading 45° toward tile."""
        alt, approaching = predict_altitude_at_closest_approach(
            aircraft_lat=47.0,
            aircraft_lon=-122.0,
            aircraft_alt_ft=10000.0,
            aircraft_hdg=45.0,  # Northeast
            aircraft_speed_mps=100.0,
            vertical_speed_fpm=0.0,
            tile_lat=47.1,  # North (bearing ~0°)
            tile_lon=-122.0
        )
        # Angle diff is 45°, cos(45) ≈ 0.707
        # Still approaching
        assert approaching is True

    def test_89_degree_approach(self):
        """Aircraft heading almost perpendicular (89°)."""
        # This is still technically approaching (< 90°)
        alt, approaching = predict_altitude_at_closest_approach(
            aircraft_lat=47.0,
            aircraft_lon=-122.0,
            aircraft_alt_ft=10000.0,
            aircraft_hdg=89.0,  # Almost east
            aircraft_speed_mps=100.0,
            vertical_speed_fpm=0.0,
            tile_lat=47.1,  # North (bearing ~0°)
            tile_lon=-122.0
        )
        # Angle diff is 89°, cos(89) ≈ 0.017
        # approach_speed = 100 * 0.017 = 1.7 m/s
        # This is just above MIN_APPROACH_SPEED_MPS (1.0)
        # So should still be approaching
        assert approaching is True

    def test_91_degree_receding(self):
        """Aircraft heading just past perpendicular (91°)."""
        alt, approaching = predict_altitude_at_closest_approach(
            aircraft_lat=47.0,
            aircraft_lon=-122.0,
            aircraft_alt_ft=10000.0,
            aircraft_hdg=91.0,  # Just past east
            aircraft_speed_mps=100.0,
            vertical_speed_fpm=-500.0,
            tile_lat=47.1,  # North (bearing ~0°)
            tile_lon=-122.0
        )
        # Angle diff is 91° > 90°, moving away
        assert approaching is False
        assert alt == 10000.0


# =============================================================================
# Tile Coordinate Tests
# =============================================================================


class TestTileCoordinates:
    """Tests for tile coordinate conversion."""

    def test_zoom_0(self):
        """Test zoom level 0 (single tile covering Earth)."""
        lat, lon = get_tile_center_coords(0, 0, 0)
        # Center of zoom 0 is approximately 0,0
        assert lat == pytest.approx(0.0, abs=1.0)
        assert lon == pytest.approx(0.0, abs=1.0)

    def test_known_location(self):
        """Test converting to known location."""
        # At higher zoom levels, we can verify approximate locations
        # Zoom 10 has 1024 tiles per axis
        # Seattle is approximately (47.6, -122.3)
        # At zoom 10: col ~164, row ~355 (approximately)
        lat, lon = get_tile_center_coords(355, 164, 10)
        # Should be in Pacific Northwest region
        assert 45 < lat < 50
        assert -125 < lon < -120

    def test_coordinate_ranges(self):
        """Test that coordinates are in valid ranges."""
        for zoom in [5, 10, 15]:
            max_coord = 2 ** zoom - 1
            for row, col in [(0, 0), (max_coord, max_coord), (max_coord // 2, max_coord // 2)]:
                lat, lon = get_tile_center_coords(row, col, zoom)
                assert -90 <= lat <= 90, f"Invalid lat {lat} for row={row}, col={col}, zoom={zoom}"
                assert -180 <= lon <= 180, f"Invalid lon {lon} for row={row}, col={col}, zoom={zoom}"


# =============================================================================
# Realistic Flight Scenarios
# =============================================================================


class TestRealisticFlightScenarios:
    """Tests simulating realistic flight scenarios."""

    def test_departure_climb(self):
        """Simulate departure climb from airport."""
        # Aircraft just took off, climbing to cruise
        alt, approaching = predict_altitude_at_closest_approach(
            aircraft_lat=47.45,  # Near Seattle
            aircraft_lon=-122.30,
            aircraft_alt_ft=3000.0,  # Low altitude
            aircraft_hdg=340.0,  # Heading northwest
            aircraft_speed_mps=80.0,  # ~160 knots, climbing
            vertical_speed_fpm=2000.0,  # Good climb rate
            tile_lat=47.60,  # Ahead and north
            tile_lon=-122.40
        )
        # Should be approaching and predicting higher altitude
        assert approaching is True
        assert alt > 3000.0

    def test_cruise_level(self):
        """Simulate stable cruise at FL350."""
        alt, approaching = predict_altitude_at_closest_approach(
            aircraft_lat=47.00,
            aircraft_lon=-122.00,
            aircraft_alt_ft=35000.0,
            aircraft_hdg=270.0,  # West
            aircraft_speed_mps=250.0,  # ~485 knots
            vertical_speed_fpm=0.0,  # Level
            tile_lat=47.00,  # Directly ahead
            tile_lon=-123.00
        )
        assert approaching is True
        assert alt == pytest.approx(35000.0, rel=0.01)

    def test_approach_descent(self):
        """Simulate approach descent."""
        alt, approaching = predict_altitude_at_closest_approach(
            aircraft_lat=47.30,
            aircraft_lon=-122.00,
            aircraft_alt_ft=8000.0,
            aircraft_hdg=220.0,  # Southwest
            aircraft_speed_mps=75.0,  # ~150 knots
            vertical_speed_fpm=-1500.0,  # Descending
            tile_lat=47.10,  # Ahead and south
            tile_lon=-122.30
        )
        assert approaching is True
        assert alt < 8000.0  # Should predict lower

    def test_holding_pattern(self):
        """Simulate holding pattern - tile behind aircraft."""
        alt, approaching = predict_altitude_at_closest_approach(
            aircraft_lat=47.50,
            aircraft_lon=-122.00,
            aircraft_alt_ft=10000.0,
            aircraft_hdg=90.0,  # East
            aircraft_speed_mps=70.0,
            vertical_speed_fpm=0.0,
            tile_lat=47.50,  # West of aircraft (behind)
            tile_lon=-122.50
        )
        # Tile is behind (bearing ~270°, heading 90°, diff = 180°)
        assert approaching is False

    def test_go_around(self):
        """Simulate go-around with climb."""
        alt, approaching = predict_altitude_at_closest_approach(
            aircraft_lat=47.45,
            aircraft_lon=-122.31,
            aircraft_alt_ft=500.0,  # Very low
            aircraft_hdg=340.0,  # Runway heading
            aircraft_speed_mps=70.0,
            vertical_speed_fpm=3000.0,  # Aggressive climb
            tile_lat=47.50,  # Ahead
            tile_lon=-122.35
        )
        assert approaching is True
        # Should predict significantly higher
        assert alt > 500.0 + 1000  # At least 1000ft higher


# =============================================================================
# Numerical Precision Tests
# =============================================================================


class TestNumericalPrecision:
    """Tests for numerical precision and edge cases."""

    def test_small_distances(self):
        """Test with very small distances (meters)."""
        alt, approaching = predict_altitude_at_closest_approach(
            aircraft_lat=47.000000,
            aircraft_lon=-122.000000,
            aircraft_alt_ft=10000.0,
            aircraft_hdg=0.0,
            aircraft_speed_mps=100.0,
            vertical_speed_fpm=0.0,
            tile_lat=47.000001,  # ~0.1m north
            tile_lon=-122.000000
        )
        # Very close, should return current altitude
        assert alt == pytest.approx(10000.0, rel=0.01)

    def test_large_distances(self):
        """Test with large distances (100+ km)."""
        alt, approaching = predict_altitude_at_closest_approach(
            aircraft_lat=47.00,
            aircraft_lon=-122.00,
            aircraft_alt_ft=35000.0,
            aircraft_hdg=0.0,
            aircraft_speed_mps=250.0,  # Fast jet
            vertical_speed_fpm=-500.0,  # Slight descent
            tile_lat=49.00,  # ~222km north
            tile_lon=-122.00
        )
        # Time to reach: 222000m / 250 m/s = 888 seconds = 14.8 minutes
        # Altitude change: -500 * 14.8 = -7400 ft
        # Predicted: 35000 - 7400 = 27600 ft
        assert approaching is True
        assert 25000 < alt < 30000

    def test_antimeridian(self):
        """Test near the antimeridian (180° longitude)."""
        alt, approaching = predict_altitude_at_closest_approach(
            aircraft_lat=47.00,
            aircraft_lon=179.90,
            aircraft_alt_ft=35000.0,
            aircraft_hdg=90.0,  # East
            aircraft_speed_mps=250.0,
            vertical_speed_fpm=0.0,
            tile_lat=47.00,
            tile_lon=-179.90  # Just past antimeridian
        )
        # Should handle wraparound correctly
        assert approaching is True

    def test_polar_regions(self):
        """Test in polar regions (high latitude)."""
        alt, approaching = predict_altitude_at_closest_approach(
            aircraft_lat=85.00,  # Near north pole
            aircraft_lon=0.00,
            aircraft_alt_ft=35000.0,
            aircraft_hdg=0.0,
            aircraft_speed_mps=250.0,
            vertical_speed_fpm=0.0,
            tile_lat=86.00,
            tile_lon=0.00
        )
        # Should still work at high latitudes
        assert isinstance(alt, float)
        assert isinstance(approaching, bool)

