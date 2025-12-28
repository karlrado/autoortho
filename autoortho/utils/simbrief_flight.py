#!/usr/bin/env python3
"""
SimBrief Flight Plan Manager.

Manages SimBrief flight plan data for integration with dynamic zoom levels
and spatial prefetching. Provides methods to:
- Store and access flight plan waypoints with altitudes (both MSL and AGL)
- Find closest waypoints to a given position
- Determine if aircraft is on-route or deviated
- Get upcoming waypoints for prefetching

Altitude Calculations:
    This module uses Above Ground Level (AGL) altitude by default for zoom
    level calculations. AGL is calculated as:
    
        AGL = MSL altitude - terrain elevation (ground_height)
    
    AGL is more appropriate for imagery quality decisions because it represents
    the actual height above the terrain being viewed:
    
    - Flying at 10,000 ft MSL over 5,000 ft mountains = 5,000 ft AGL (needs higher zoom)
    - Flying at 10,000 ft MSL over the ocean = 10,000 ft AGL (can use lower zoom)
    
    When multiple waypoints are within the consideration radius of a tile,
    the module uses conservative values to ensure adequate detail:
    
    - Lowest flight altitude (MSL) - accounts for descent through the area
    - Highest ground elevation - accounts for mountains in the area
    - Conservative AGL = lowest_MSL - highest_ground
    
    This ensures maximum detail when flying over areas with varied terrain
    (e.g., descending over mountains).
    
    SimBrief provides ground_height (terrain elevation) for each waypoint,
    which is used to calculate AGL.

Thread Safety:
    All public methods are thread-safe using internal locking.
"""

import math
import threading
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

log = logging.getLogger(__name__)


# Earth radius in nautical miles
EARTH_RADIUS_NM = 3440.065


@dataclass
class PathPoint:
    """
    A point along the flight path with time-to-encounter information.
    
    Used for prefetching prioritization - points the aircraft will reach
    sooner should be prefetched first.
    
    Attributes:
        lat: Latitude in degrees
        lon: Longitude in degrees
        time_to_reach_sec: Estimated time to reach this point in seconds
        altitude_ft: Planned altitude at this point in feet (MSL)
        ground_height_ft: Terrain elevation at this point in feet
        distance_from_start_nm: Distance from current position along the path
    """
    lat: float
    lon: float
    time_to_reach_sec: float
    altitude_ft: int
    ground_height_ft: int
    distance_from_start_nm: float
    
    @property
    def altitude_agl_ft(self) -> int:
        """Calculate altitude Above Ground Level (AGL)."""
        return max(0, self.altitude_ft - self.ground_height_ft)


@dataclass
class FlightFix:
    """
    A waypoint/fix from the SimBrief flight plan.
    
    Attributes:
        ident: Fix identifier (e.g., "BOPTA", "TOC", "VHHH")
        name: Full name of the fix
        fix_type: Type of fix (wpt, apt, vor, ltlg)
        lat: Latitude in degrees
        lon: Longitude in degrees
        altitude_ft: Planned altitude at this fix in feet (MSL)
        ground_height_ft: Terrain elevation at this fix in feet (MSL)
        index: Position in the route (0 = departure, -1 = arrival)
        time_total_sec: Cumulative time from departure to this fix in seconds
        time_leg_sec: Time for this leg only in seconds
        groundspeed_kts: Planned ground speed at this fix in knots
        is_toc: True if this is Top of Climb
        is_tod: True if this is Top of Descent
        
    Properties:
        altitude_agl_ft: Altitude Above Ground Level (altitude_ft - ground_height_ft)
    """
    ident: str
    name: str
    fix_type: str
    lat: float
    lon: float
    altitude_ft: int
    ground_height_ft: int
    index: int
    time_total_sec: int = 0
    time_leg_sec: int = 0
    groundspeed_kts: int = 0
    is_toc: bool = False
    is_tod: bool = False
    
    @property
    def altitude_agl_ft(self) -> int:
        """
        Calculate altitude Above Ground Level (AGL).
        
        AGL = MSL altitude - terrain elevation
        
        This is more relevant for imagery quality decisions since
        it represents actual height above the terrain being viewed.
        """
        return max(0, self.altitude_ft - self.ground_height_ft)
    
    def distance_to(self, lat: float, lon: float) -> float:
        """
        Calculate great-circle distance to a point in nautical miles.
        
        Uses Haversine formula for accuracy.
        """
        lat1_rad = math.radians(self.lat)
        lat2_rad = math.radians(lat)
        delta_lat = math.radians(lat - self.lat)
        delta_lon = math.radians(lon - self.lon)
        
        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return EARTH_RADIUS_NM * c


class SimBriefFlightManager:
    """
    Manages SimBrief flight plan data for AutoOrtho integration.
    
    This class provides the bridge between SimBrief flight plans and
    AutoOrtho's dynamic zoom and prefetching systems.
    
    Usage:
        manager = SimBriefFlightManager()
        manager.load_flight_data(simbrief_json)
        
        # Get altitude for a tile position
        altitude = manager.get_altitude_at_position(tile_lat, tile_lon, radius_nm=50)
        
        # Check if on route
        if manager.is_on_route(aircraft_lat, aircraft_lon, threshold_nm=40):
            # Use flight plan data
        else:
            # Fall back to DataRef-based calculations
    """
    
    def __init__(self):
        """Initialize the flight manager with empty state."""
        self._lock = threading.RLock()
        self._fixes: List[FlightFix] = []
        self._origin_icao: str = ""
        self._dest_icao: str = ""
        self._cruise_altitude_ft: int = 0
        self._flight_loaded: bool = False
        self._current_fix_index: int = 0  # Track progress along route
        
    def load_flight_data(self, simbrief_data: dict) -> bool:
        """
        Load flight data from SimBrief JSON response.
        
        Args:
            simbrief_data: Full SimBrief API response as dict
            
        Returns:
            True if flight data was loaded successfully
        """
        with self._lock:
            self._fixes = []
            self._flight_loaded = False
            self._current_fix_index = 0
            
            try:
                # Extract origin/destination
                origin = simbrief_data.get('origin', {})
                destination = simbrief_data.get('destination', {})
                general = simbrief_data.get('general', {})
                navlog = simbrief_data.get('navlog', {})
                
                self._origin_icao = origin.get('icao_code', '')
                self._dest_icao = destination.get('icao_code', '')
                
                # Get cruise altitude
                try:
                    self._cruise_altitude_ft = int(general.get('initial_altitude', 0))
                except (ValueError, TypeError):
                    self._cruise_altitude_ft = 0
                
                # Parse fixes from navlog
                fixes_data = navlog.get('fix', [])
                if not isinstance(fixes_data, list):
                    log.warning("SimBrief navlog fix data is not a list")
                    return False
                
                for i, fix_data in enumerate(fixes_data):
                    try:
                        fix = self._parse_fix(fix_data, i)
                        if fix:
                            self._fixes.append(fix)
                    except Exception as e:
                        log.debug(f"Error parsing fix {i}: {e}")
                        continue
                
                if len(self._fixes) < 2:
                    log.warning("SimBrief flight plan has fewer than 2 fixes")
                    return False
                
                self._flight_loaded = True
                log.info(f"Loaded SimBrief flight: {self._origin_icao} -> {self._dest_icao} "
                        f"with {len(self._fixes)} fixes")
                return True
                
            except Exception as e:
                log.error(f"Error loading SimBrief flight data: {e}")
                return False
    
    def _parse_fix(self, fix_data: dict, index: int) -> Optional[FlightFix]:
        """Parse a single fix from SimBrief data."""
        try:
            ident = fix_data.get('ident', '')
            name = fix_data.get('name', ident)
            fix_type = fix_data.get('type', 'wpt')
            
            lat = float(fix_data.get('pos_lat', 0))
            lon = float(fix_data.get('pos_long', 0))
            
            # Validate coordinates
            if lat < -90 or lat > 90 or lon < -180 or lon > 180:
                return None
            
            altitude_ft = int(fix_data.get('altitude_feet', 0))
            
            # Get ground height (terrain elevation) - defaults to 0 if not available
            ground_height_ft = int(fix_data.get('ground_height', 0))
            
            # Get timing information from SimBrief
            # time_total: cumulative seconds from departure
            # time_leg: seconds for this leg only
            # groundspeed: planned ground speed at this fix in knots
            time_total_sec = int(fix_data.get('time_total', 0))
            time_leg_sec = int(fix_data.get('time_leg', 0))
            groundspeed_kts = int(fix_data.get('groundspeed', 0))
            
            # Check for TOC/TOD
            is_toc = ident.upper() == 'TOC' or 'TOP OF CLIMB' in name.upper()
            is_tod = ident.upper() == 'TOD' or 'TOP OF DESCENT' in name.upper()
            
            return FlightFix(
                ident=ident,
                name=name,
                fix_type=fix_type,
                lat=lat,
                lon=lon,
                altitude_ft=altitude_ft,
                ground_height_ft=ground_height_ft,
                index=index,
                time_total_sec=time_total_sec,
                time_leg_sec=time_leg_sec,
                groundspeed_kts=groundspeed_kts,
                is_toc=is_toc,
                is_tod=is_tod
            )
        except Exception as e:
            log.debug(f"Error parsing fix: {e}")
            return None
    
    def clear(self) -> None:
        """Clear all flight data."""
        with self._lock:
            self._fixes = []
            self._origin_icao = ""
            self._dest_icao = ""
            self._cruise_altitude_ft = 0
            self._flight_loaded = False
            self._current_fix_index = 0
    
    @property
    def is_loaded(self) -> bool:
        """Check if a flight plan is loaded."""
        with self._lock:
            return self._flight_loaded
    
    @property
    def fix_count(self) -> int:
        """Get the number of fixes in the flight plan."""
        with self._lock:
            return len(self._fixes)
    
    @property
    def origin(self) -> str:
        """Get origin ICAO code."""
        with self._lock:
            return self._origin_icao
    
    @property
    def destination(self) -> str:
        """Get destination ICAO code."""
        with self._lock:
            return self._dest_icao
    
    @property
    def cruise_altitude_ft(self) -> int:
        """Get planned cruise altitude in feet."""
        with self._lock:
            return self._cruise_altitude_ft
    
    def get_closest_fix(self, lat: float, lon: float) -> Optional[Tuple[FlightFix, float]]:
        """
        Find the closest fix to a given position.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            
        Returns:
            Tuple of (FlightFix, distance_nm) or None if no fixes
        """
        with self._lock:
            if not self._fixes:
                return None
            
            closest_fix = None
            closest_distance = float('inf')
            
            for fix in self._fixes:
                distance = fix.distance_to(lat, lon)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_fix = fix
            
            return (closest_fix, closest_distance) if closest_fix else None
    
    def get_fixes_within_radius(self, lat: float, lon: float, 
                                 radius_nm: float) -> List[Tuple[FlightFix, float]]:
        """
        Get all fixes within a given radius of a position.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            radius_nm: Radius in nautical miles
            
        Returns:
            List of (FlightFix, distance_nm) tuples, sorted by distance
        """
        with self._lock:
            if not self._fixes:
                return []
            
            nearby_fixes = []
            for fix in self._fixes:
                distance = fix.distance_to(lat, lon)
                if distance <= radius_nm:
                    nearby_fixes.append((fix, distance))
            
            # Sort by distance
            nearby_fixes.sort(key=lambda x: x[1])
            return nearby_fixes
    
    def get_altitude_at_position(self, lat: float, lon: float, 
                                  radius_nm: float = 50,
                                  use_agl: bool = True) -> Optional[int]:
        """
        Get the planned altitude for a position based on nearby fixes.
        
        When multiple fixes are within the radius, uses conservative values:
        - Lowest flight altitude (MSL) - accounts for descent through the area
        - Highest ground elevation - accounts for mountains in the area
        - AGL = lowest_MSL - highest_ground = most conservative effective altitude
        
        This ensures maximum detail when flying over areas with varied terrain
        (e.g., descending over mountains).
        
        When use_agl is True (default), returns Above Ground Level (AGL) altitude
        which is more appropriate for imagery quality decisions since it represents
        actual height above the terrain being viewed.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            radius_nm: Radius in nautical miles to consider fixes
            use_agl: If True, returns conservative AGL; if False, returns lowest MSL
            
        Returns:
            Conservative altitude (ft AGL or MSL) based on nearby fixes, or None if no fixes nearby
        """
        with self._lock:
            if not self._fixes:
                return None
            
            # Get fixes within radius
            nearby = self.get_fixes_within_radius(lat, lon, radius_nm)
            
            if not nearby:
                # No fixes within radius - find closest fix
                closest = self.get_closest_fix(lat, lon)
                if closest:
                    fix = closest[0]
                    return fix.altitude_agl_ft if use_agl else fix.altitude_ft
                return None
            
            if use_agl:
                # Conservative AGL calculation:
                # - Use lowest flight altitude (MSL) - we might be descending
                # - Use highest ground elevation - there might be mountains
                # - AGL = lowest_MSL - highest_ground = most conservative result
                lowest_msl = min(fix.altitude_ft for fix, _ in nearby)
                highest_ground = max(fix.ground_height_ft for fix, _ in nearby)
                conservative_agl = max(0, lowest_msl - highest_ground)
                return conservative_agl
            else:
                # Return lowest MSL altitude
                return min(fix.altitude_ft for fix, _ in nearby)
    
    def is_on_route(self, lat: float, lon: float, threshold_nm: float = 40) -> bool:
        """
        Check if a position is within threshold distance of the route.
        
        Uses distance to the closest route segment, not just waypoints.
        
        Args:
            lat: Aircraft latitude
            lon: Aircraft longitude
            threshold_nm: Maximum distance from route to be considered "on route"
            
        Returns:
            True if position is within threshold of the route
        """
        with self._lock:
            if not self._fixes or len(self._fixes) < 2:
                return False
            
            # Check distance to each route segment
            for i in range(len(self._fixes) - 1):
                fix1 = self._fixes[i]
                fix2 = self._fixes[i + 1]
                
                distance = self._distance_to_segment(
                    lat, lon,
                    fix1.lat, fix1.lon,
                    fix2.lat, fix2.lon
                )
                
                if distance <= threshold_nm:
                    return True
            
            return False
    
    def get_distance_from_origin(self, lat: float, lon: float) -> Optional[float]:
        """
        Calculate distance from a position to the origin airport.
        
        Uses the first fix in the flight plan as the origin reference point.
        This allows prefetching to start while parked at the departure airport.
        
        Args:
            lat: Aircraft latitude
            lon: Aircraft longitude
            
        Returns:
            Distance in nautical miles, or None if no flight plan is loaded
        """
        with self._lock:
            if not self._fixes:
                return None
            
            origin_fix = self._fixes[0]
            return self._haversine_distance(lat, lon, origin_fix.lat, origin_fix.lon)
    
    def _distance_to_segment(self, lat: float, lon: float,
                              lat1: float, lon1: float,
                              lat2: float, lon2: float) -> float:
        """
        Calculate minimum distance from a point to a line segment.
        
        Uses simplified spherical geometry for efficiency.
        Returns distance in nautical miles.
        """
        # Convert to radians for calculation
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Calculate distances to endpoints
        d1 = self._haversine_distance(lat, lon, lat1, lon1)
        d2 = self._haversine_distance(lat, lon, lat2, lon2)
        segment_length = self._haversine_distance(lat1, lon1, lat2, lon2)
        
        # If segment is very short, return distance to closest endpoint
        if segment_length < 0.1:  # Less than 0.1 nm
            return min(d1, d2)
        
        # Project point onto segment using simple approach
        # This uses bearing-based projection
        
        # Calculate bearings
        bearing_1_to_point = self._initial_bearing(lat1, lon1, lat, lon)
        bearing_1_to_2 = self._initial_bearing(lat1, lon1, lat2, lon2)
        bearing_2_to_point = self._initial_bearing(lat2, lon2, lat, lon)
        bearing_2_to_1 = self._initial_bearing(lat2, lon2, lat1, lon1)
        
        # Check if point projects onto the segment
        angle1 = abs(self._normalize_angle(bearing_1_to_point - bearing_1_to_2))
        angle2 = abs(self._normalize_angle(bearing_2_to_point - bearing_2_to_1))
        
        if angle1 > 90 or angle2 > 90:
            # Point is past one of the endpoints
            return min(d1, d2)
        
        # Calculate cross-track distance
        # Using cross-track distance formula
        cross_track = self._cross_track_distance(lat, lon, lat1, lon1, lat2, lon2)
        return abs(cross_track)
    
    def _haversine_distance(self, lat1: float, lon1: float, 
                            lat2: float, lon2: float) -> float:
        """Calculate haversine distance in nautical miles."""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return EARTH_RADIUS_NM * c
    
    def _initial_bearing(self, lat1: float, lon1: float, 
                         lat2: float, lon2: float) -> float:
        """Calculate initial bearing from point 1 to point 2 in degrees."""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lon_rad = math.radians(lon2 - lon1)
        
        x = math.sin(delta_lon_rad) * math.cos(lat2_rad)
        y = (math.cos(lat1_rad) * math.sin(lat2_rad) -
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon_rad))
        
        return math.degrees(math.atan2(x, y)) % 360
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to -180 to 180 range."""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
    
    def _cross_track_distance(self, lat: float, lon: float,
                               lat1: float, lon1: float,
                               lat2: float, lon2: float) -> float:
        """Calculate cross-track distance in nautical miles."""
        # Distance from point 1 to current point
        d13 = self._haversine_distance(lat1, lon1, lat, lon) / EARTH_RADIUS_NM
        
        # Bearing from point 1 to current point
        theta13 = math.radians(self._initial_bearing(lat1, lon1, lat, lon))
        
        # Bearing from point 1 to point 2
        theta12 = math.radians(self._initial_bearing(lat1, lon1, lat2, lon2))
        
        # Cross-track distance
        dxt = math.asin(math.sin(d13) * math.sin(theta13 - theta12))
        
        return dxt * EARTH_RADIUS_NM
    
    def get_upcoming_fixes(self, lat: float, lon: float, 
                           count: int = 10) -> List[FlightFix]:
        """
        Get the next N fixes from the current position along the route.
        
        Updates internal progress tracking to efficiently find position.
        
        Args:
            lat: Current aircraft latitude
            lon: Current aircraft longitude
            count: Maximum number of fixes to return
            
        Returns:
            List of upcoming FlightFix objects
        """
        with self._lock:
            if not self._fixes:
                return []
            
            # Find the closest fix to determine position on route
            closest = self.get_closest_fix(lat, lon)
            if not closest:
                return []
            
            closest_fix, _ = closest
            start_index = closest_fix.index
            
            # Look at fixes ahead of current position
            # Include current fix and subsequent fixes
            upcoming = []
            for i in range(start_index, min(start_index + count, len(self._fixes))):
                upcoming.append(self._fixes[i])
            
            return upcoming
    
    def get_all_fixes(self) -> List[FlightFix]:
        """Get all fixes in the flight plan."""
        with self._lock:
            return list(self._fixes)
    
    def get_path_points_with_time(self, aircraft_lat: float, aircraft_lon: float,
                                   lookahead_sec: float,
                                   spacing_nm: float = 20.0) -> List[PathPoint]:
        """
        Get interpolated points along the flight path with time-to-encounter.
        
        Uses SimBrief's pre-calculated time_total for each fix to determine
        time-to-encounter. This is more accurate than calculating from current
        ground speed because SimBrief accounts for winds, climb/descent speeds, etc.
        
        The method:
        1. Finds the aircraft's position on the route
        2. Calculates the aircraft's "current time" based on position
        3. Walks forward through fixes, interpolating points at regular intervals
        4. Calculates time_to_reach = point_time - aircraft_current_time
        5. Stops when lookahead time is exceeded
        
        Args:
            aircraft_lat: Current aircraft latitude
            aircraft_lon: Current aircraft longitude
            lookahead_sec: Maximum time ahead to generate points (seconds)
            spacing_nm: Distance between interpolated points (nautical miles)
            
        Returns:
            List of PathPoint objects sorted by time_to_reach_sec (closest first)
        """
        with self._lock:
            if not self._fixes or len(self._fixes) < 2:
                return []
            
            # Find the closest segment and projection point
            closest_segment_idx, projection_lat, projection_lon, dist_along_segment = \
                self._find_closest_segment_projection(aircraft_lat, aircraft_lon)
            
            if closest_segment_idx < 0:
                return []
            
            # Calculate the aircraft's "current time" on the route
            # by interpolating between the fix times based on position
            fix1 = self._fixes[closest_segment_idx]
            fix2 = self._fixes[closest_segment_idx + 1]
            segment_length = self._haversine_distance(fix1.lat, fix1.lon, fix2.lat, fix2.lon)
            
            if segment_length > 0:
                segment_fraction = dist_along_segment / segment_length
            else:
                segment_fraction = 0.0
            
            # Interpolate current time based on position between fixes
            # fix1.time_total_sec is time to reach fix1, fix2.time_total_sec is time to reach fix2
            aircraft_current_time_sec = fix1.time_total_sec + segment_fraction * (fix2.time_total_sec - fix1.time_total_sec)
            
            path_points = []
            cumulative_distance_nm = 0.0
            
            # Start from the projection point and work forward along the route
            current_lat = projection_lat
            current_lon = projection_lon
            current_segment_idx = closest_segment_idx
            remaining_in_segment = self._haversine_distance(
                current_lat, current_lon,
                self._fixes[current_segment_idx + 1].lat,
                self._fixes[current_segment_idx + 1].lon
            )
            
            while current_segment_idx < len(self._fixes) - 1:
                fix1 = self._fixes[current_segment_idx]
                fix2 = self._fixes[current_segment_idx + 1]
                segment_length = self._haversine_distance(fix1.lat, fix1.lon, fix2.lat, fix2.lon)
                
                # Generate points along this segment at spacing intervals
                distance_to_next_point = spacing_nm
                
                while remaining_in_segment >= distance_to_next_point:
                    # Calculate how far we've moved into this segment
                    distance_into_segment = segment_length - remaining_in_segment + distance_to_next_point
                    seg_fraction = distance_into_segment / segment_length if segment_length > 0 else 0
                    seg_fraction = max(0.0, min(1.0, seg_fraction))
                    
                    # Calculate interpolated position
                    point_lat, point_lon = self._interpolate_position(
                        fix1.lat, fix1.lon,
                        fix2.lat, fix2.lon,
                        seg_fraction
                    )
                    
                    # Interpolate time using SimBrief's fix times
                    point_time_sec = fix1.time_total_sec + seg_fraction * (fix2.time_total_sec - fix1.time_total_sec)
                    time_to_reach = point_time_sec - aircraft_current_time_sec
                    
                    # Check if we've exceeded lookahead
                    if time_to_reach > lookahead_sec:
                        return path_points
                    
                    # Skip points behind us (negative time)
                    if time_to_reach < 0:
                        remaining_in_segment -= distance_to_next_point
                        cumulative_distance_nm += distance_to_next_point
                        continue
                    
                    # Interpolate altitude and ground height
                    altitude_ft = int(fix1.altitude_ft + seg_fraction * (fix2.altitude_ft - fix1.altitude_ft))
                    ground_height_ft = int(fix1.ground_height_ft + seg_fraction * (fix2.ground_height_ft - fix1.ground_height_ft))
                    
                    cumulative_distance_nm += distance_to_next_point
                    
                    path_points.append(PathPoint(
                        lat=point_lat,
                        lon=point_lon,
                        time_to_reach_sec=time_to_reach,
                        altitude_ft=altitude_ft,
                        ground_height_ft=ground_height_ft,
                        distance_from_start_nm=cumulative_distance_nm
                    ))
                    
                    # Update remaining distance in segment
                    remaining_in_segment -= distance_to_next_point
                
                # Move to next segment
                if current_segment_idx < len(self._fixes) - 2:
                    cumulative_distance_nm += remaining_in_segment
                    current_segment_idx += 1
                    remaining_in_segment = self._haversine_distance(
                        self._fixes[current_segment_idx].lat,
                        self._fixes[current_segment_idx].lon,
                        self._fixes[current_segment_idx + 1].lat,
                        self._fixes[current_segment_idx + 1].lon
                    )
                else:
                    break
            
            return path_points
    
    def _find_closest_segment_projection(self, lat: float, lon: float) -> Tuple[int, float, float, float]:
        """
        Find the route segment closest to the given position and the projection point.
        
        Returns:
            Tuple of (segment_index, projection_lat, projection_lon, distance_along_segment)
            Returns (-1, 0, 0, 0) if no valid segment found.
        """
        if len(self._fixes) < 2:
            return (-1, 0.0, 0.0, 0.0)
        
        best_segment_idx = -1
        best_distance = float('inf')
        best_projection = (0.0, 0.0)
        best_along_track = 0.0
        
        for i in range(len(self._fixes) - 1):
            fix1 = self._fixes[i]
            fix2 = self._fixes[i + 1]
            
            # Calculate projection onto this segment
            proj_lat, proj_lon, along_track = self._project_onto_segment(
                lat, lon, fix1.lat, fix1.lon, fix2.lat, fix2.lon
            )
            
            # Distance from aircraft to projection
            distance = self._haversine_distance(lat, lon, proj_lat, proj_lon)
            
            if distance < best_distance:
                best_distance = distance
                best_segment_idx = i
                best_projection = (proj_lat, proj_lon)
                best_along_track = along_track
        
        return (best_segment_idx, best_projection[0], best_projection[1], best_along_track)
    
    def _project_onto_segment(self, lat: float, lon: float,
                               lat1: float, lon1: float,
                               lat2: float, lon2: float) -> Tuple[float, float, float]:
        """
        Project a point onto a line segment.
        
        Returns:
            Tuple of (projection_lat, projection_lon, distance_along_segment_nm)
        """
        # Calculate distances
        d1 = self._haversine_distance(lat1, lon1, lat, lon)
        d2 = self._haversine_distance(lat2, lon2, lat, lon)
        segment_length = self._haversine_distance(lat1, lon1, lat2, lon2)
        
        if segment_length < 0.1:  # Very short segment
            return (lat1, lon1, 0.0)
        
        # Use bearing-based projection
        bearing_1_to_point = self._initial_bearing(lat1, lon1, lat, lon)
        bearing_1_to_2 = self._initial_bearing(lat1, lon1, lat2, lon2)
        
        angle_diff = abs(self._normalize_angle(bearing_1_to_point - bearing_1_to_2))
        
        if angle_diff > 90:
            # Point is behind segment start
            return (lat1, lon1, 0.0)
        
        bearing_2_to_point = self._initial_bearing(lat2, lon2, lat, lon)
        bearing_2_to_1 = self._initial_bearing(lat2, lon2, lat1, lon1)
        
        angle_diff_2 = abs(self._normalize_angle(bearing_2_to_point - bearing_2_to_1))
        
        if angle_diff_2 > 90:
            # Point is past segment end
            return (lat2, lon2, segment_length)
        
        # Calculate along-track distance
        along_track = d1 * math.cos(math.radians(angle_diff))
        
        # Clamp to segment
        along_track = max(0, min(segment_length, along_track))
        
        # Calculate projection point
        fraction = along_track / segment_length if segment_length > 0 else 0
        proj_lat = lat1 + fraction * (lat2 - lat1)
        proj_lon = lon1 + fraction * (lon2 - lon1)
        
        return (proj_lat, proj_lon, along_track)
    
    def _interpolate_position(self, lat1: float, lon1: float,
                               lat2: float, lon2: float,
                               fraction: float) -> Tuple[float, float]:
        """
        Interpolate between two positions.
        
        Uses simple linear interpolation (acceptable for short distances).
        
        Args:
            lat1, lon1: Start position
            lat2, lon2: End position
            fraction: Interpolation fraction (0 = start, 1 = end)
            
        Returns:
            Tuple of (interpolated_lat, interpolated_lon)
        """
        fraction = max(0.0, min(1.0, fraction))
        lat = lat1 + fraction * (lat2 - lat1)
        lon = lon1 + fraction * (lon2 - lon1)
        return (lat, lon)
    
    def _get_segment_fraction(self, lat: float, lon: float,
                               fix1: FlightFix, fix2: FlightFix) -> float:
        """
        Calculate the fraction along a segment where a point lies.
        
        Returns a value between 0 (at fix1) and 1 (at fix2).
        """
        d_to_fix1 = self._haversine_distance(lat, lon, fix1.lat, fix1.lon)
        segment_length = self._haversine_distance(fix1.lat, fix1.lon, fix2.lat, fix2.lon)
        
        if segment_length < 0.1:
            return 0.0
        
        return min(1.0, d_to_fix1 / segment_length)
    
    def get_route_info(self) -> Dict:
        """
        Get summary information about the loaded route.
        
        Returns:
            Dictionary with route summary info
        """
        with self._lock:
            if not self._flight_loaded:
                return {'loaded': False}
            
            return {
                'loaded': True,
                'origin': self._origin_icao,
                'destination': self._dest_icao,
                'cruise_altitude_ft': self._cruise_altitude_ft,
                'fix_count': len(self._fixes),
                'first_fix': self._fixes[0].ident if self._fixes else None,
                'last_fix': self._fixes[-1].ident if self._fixes else None,
            }


# Global instance for use across the application
simbrief_flight_manager = SimBriefFlightManager()

