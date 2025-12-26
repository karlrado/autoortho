#!/usr/bin/env python3
"""
Predicts altitude at closest approach to tiles.

Uses aircraft position, heading, speed, and vertical speed to
calculate what altitude the aircraft will be at when it reaches
the closest point to a given tile.

This module is used by the dynamic zoom system to determine
appropriate zoom levels based on predicted altitude, not just
current altitude.

Assumptions:
    - Aircraft travels in a straight line (great circle approximation)
    - Vertical speed remains constant over the prediction window
    - Tiles are approximated as points at their center

Usage:
    from utils.altitude_predictor import predict_altitude_at_closest_approach

    predicted_alt, will_approach = predict_altitude_at_closest_approach(
        aircraft_lat=47.5, aircraft_lon=-122.3,
        aircraft_alt_ft=10000, aircraft_hdg=270.0,
        aircraft_speed_mps=100.0, vertical_speed_fpm=-500.0,
        tile_lat=47.6, tile_lon=-122.5
    )
"""

import math
from typing import Tuple
import logging

log = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

EARTH_RADIUS_M = 6371000    # Earth's mean radius in meters
FT_TO_M = 0.3048            # Feet to meters conversion
M_TO_FT = 3.28084           # Meters to feet conversion

# Altitude bounds for clamping predictions
MIN_ALTITUDE_FT = -1000     # Below sea level (Death Valley is ~-282 ft)
MAX_ALTITUDE_FT = 60000     # Above FL600

# Minimum approach speed to be considered "approaching"
MIN_APPROACH_SPEED_MPS = 1.0

# Maximum prediction time (prevents unrealistic predictions for distant tiles)
MAX_PREDICTION_TIME_SEC = 3600  # 1 hour


# =============================================================================
# Geographic Calculations
# =============================================================================

def _haversine_distance(lat1: float, lon1: float,
                        lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two points in meters.

    Uses the Haversine formula which gives great-circle distances
    between two points on a sphere from their longitudes and latitudes.

    Args:
        lat1, lon1: First point coordinates in degrees
        lat2, lon2: Second point coordinates in degrees

    Returns:
        Distance in meters
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
    c = 2 * math.asin(math.sqrt(a))

    return EARTH_RADIUS_M * c


def _bearing_to_point(lat1: float, lon1: float,
                      lat2: float, lon2: float) -> float:
    """
    Calculate initial bearing from point 1 to point 2.

    Uses the forward azimuth formula for great circle navigation.
    The bearing is the direction to travel to reach the destination.

    Args:
        lat1, lon1: Starting point coordinates in degrees
        lat2, lon2: Destination point coordinates in degrees

    Returns:
        Bearing in degrees (0-360, where 0=North, 90=East, etc.)
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon_rad = math.radians(lon2 - lon1)

    x = math.sin(dlon_rad) * math.cos(lat2_rad)
    y = (math.cos(lat1_rad) * math.sin(lat2_rad) -
         math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad))

    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


def _angular_difference(heading1: float, heading2: float) -> float:
    """
    Calculate the smallest angular difference between two headings.

    Args:
        heading1, heading2: Headings in degrees (0-360)

    Returns:
        Absolute angular difference in degrees (0-180)
    """
    diff = abs(heading1 - heading2)
    if diff > 180:
        diff = 360 - diff
    return diff


# =============================================================================
# Altitude Prediction
# =============================================================================

def predict_altitude_at_closest_approach(
    aircraft_lat: float,
    aircraft_lon: float,
    aircraft_alt_ft: float,
    aircraft_hdg: float,
    aircraft_speed_mps: float,
    vertical_speed_fpm: float,
    tile_lat: float,
    tile_lon: float
) -> Tuple[float, bool]:
    """
    Predict altitude when aircraft reaches closest point to tile.

    The algorithm:
    1. Calculate bearing from aircraft to tile center
    2. If angle between heading and bearing > 90°, aircraft is moving away
       → Return current altitude (closest approach is now)
    3. Otherwise, calculate approach speed (velocity component toward tile)
    4. Calculate time to closest approach
    5. Apply vertical speed to predict altitude at that time

    Args:
        aircraft_lat: Aircraft latitude in degrees
        aircraft_lon: Aircraft longitude in degrees
        aircraft_alt_ft: Aircraft pressure altitude in feet
        aircraft_hdg: Aircraft heading in degrees (0-360)
        aircraft_speed_mps: Aircraft ground speed in meters/second
        vertical_speed_fpm: Aircraft vertical speed in feet/minute
                           (positive = climbing, negative = descending)
        tile_lat: Tile center latitude in degrees
        tile_lon: Tile center longitude in degrees

    Returns:
        Tuple of (predicted_altitude_ft, will_approach):
        - predicted_altitude_ft: Altitude in feet at closest approach
        - will_approach: True if aircraft is approaching tile,
                        False if moving away or stationary
    """
    # Calculate bearing from aircraft to tile
    bearing_to_tile = _bearing_to_point(
        aircraft_lat, aircraft_lon, tile_lat, tile_lon
    )

    # Calculate angular difference between heading and bearing to tile
    angle_diff = _angular_difference(aircraft_hdg, bearing_to_tile)

    # If angle > 90°, we're moving away from the tile
    # The closest approach has already happened (or is happening now)
    # Use current altitude
    if angle_diff > 90:
        return (aircraft_alt_ft, False)

    # Calculate distance to tile
    distance_m = _haversine_distance(
        aircraft_lat, aircraft_lon, tile_lat, tile_lon
    )

    # Edge case: already at the tile
    if distance_m < 100:  # Within 100 meters
        return (aircraft_alt_ft, True)

    # Project speed along the direction to tile
    # cos(angle_diff) gives the component of velocity toward tile
    approach_speed_mps = aircraft_speed_mps * math.cos(math.radians(angle_diff))

    # If not effectively approaching
    if approach_speed_mps < MIN_APPROACH_SPEED_MPS:
        return (aircraft_alt_ft, False)

    # Time to closest approach
    # This is simplified - assumes straight line flight
    # For tiles directly ahead, this is time to reach the tile
    # For tiles to the side, this is time to closest point on our track
    time_to_closest_sec = distance_m / approach_speed_mps

    # Clamp to reasonable prediction window
    time_to_closest_sec = min(time_to_closest_sec, MAX_PREDICTION_TIME_SEC)

    # Predict altitude change
    # vertical_speed_fpm is feet per minute
    # Convert time to minutes for calculation
    time_to_closest_min = time_to_closest_sec / 60.0
    alt_change_ft = vertical_speed_fpm * time_to_closest_min

    # Calculate predicted altitude
    predicted_alt_ft = aircraft_alt_ft + alt_change_ft

    # Clamp to reasonable bounds
    predicted_alt_ft = max(MIN_ALTITUDE_FT, min(MAX_ALTITUDE_FT, predicted_alt_ft))

    return (predicted_alt_ft, True)


def get_tile_center_coords(row: int, col: int, zoom: int) -> Tuple[float, float]:
    """
    Convert tile row/col/zoom to center lat/lon coordinates.

    Uses the Web Mercator (OSM) tile coordinate system.
    This is a convenience function that mirrors _chunk_to_latlon
    in getortho.py for use in testing and standalone calculations.

    Args:
        row: Tile row (Y coordinate, 0 at top)
        col: Tile column (X coordinate, 0 at left)
        zoom: Tile zoom level

    Returns:
        Tuple of (latitude, longitude) in degrees
    """
    n = 2.0 ** zoom
    # Use tile center by adding 0.5 to both coordinates
    lon = (col + 0.5) / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * (row + 0.5) / n)))
    lat = math.degrees(lat_rad)
    return (lat, lon)

