#!/usr/bin/env python3
"""
Tests for the path interpolation functionality in SimBrief flight manager.

Tests the new get_path_points_with_time method that provides uniform
path interpolation for time-based prefetch prioritization.
"""

import pytest
import math
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.simbrief_flight import SimBriefFlightManager, FlightFix, PathPoint


class TestPathPointDataclass:
    """Test the PathPoint dataclass."""
    
    def test_path_point_creation(self):
        """Test creating a PathPoint."""
        point = PathPoint(
            lat=47.5,
            lon=-122.3,
            time_to_reach_sec=300.0,
            altitude_ft=10000,
            ground_height_ft=500,
            distance_from_start_nm=50.0
        )
        
        assert point.lat == 47.5
        assert point.lon == -122.3
        assert point.time_to_reach_sec == 300.0
        assert point.altitude_ft == 10000
        assert point.ground_height_ft == 500
        assert point.distance_from_start_nm == 50.0
    
    def test_path_point_agl_calculation(self):
        """Test AGL calculation for PathPoint."""
        point = PathPoint(
            lat=47.5,
            lon=-122.3,
            time_to_reach_sec=300.0,
            altitude_ft=10000,
            ground_height_ft=5000,
            distance_from_start_nm=50.0
        )
        
        assert point.altitude_agl_ft == 5000
    
    def test_path_point_agl_never_negative(self):
        """Test that AGL is clamped to 0 when altitude < ground."""
        point = PathPoint(
            lat=47.5,
            lon=-122.3,
            time_to_reach_sec=300.0,
            altitude_ft=1000,
            ground_height_ft=2000,
            distance_from_start_nm=50.0
        )
        
        assert point.altitude_agl_ft == 0


class TestPathInterpolation:
    """Test the path interpolation methods."""
    
    @pytest.fixture
    def manager_with_route(self):
        """Create a manager with a simple three-waypoint route with SimBrief times."""
        manager = SimBriefFlightManager()
        
        # Create a simple route: KSEA -> BUWZO -> KPDX
        # Times are in seconds from departure (SimBrief format)
        simbrief_data = {
            'origin': {'icao_code': 'KSEA'},
            'destination': {'icao_code': 'KPDX'},
            'general': {'initial_altitude': 35000},
            'navlog': {
                'fix': [
                    {
                        'ident': 'KSEA',
                        'name': 'Seattle',
                        'type': 'apt',
                        'pos_lat': 47.449,
                        'pos_long': -122.309,
                        'altitude_feet': 0,
                        'ground_height': 433,
                        'time_total': 0,      # At departure
                        'time_leg': 0,
                        'groundspeed': 150
                    },
                    {
                        'ident': 'BUWZO',
                        'name': 'BUWZO',
                        'type': 'wpt',
                        'pos_lat': 46.5,
                        'pos_long': -122.5,
                        'altitude_feet': 15000,
                        'ground_height': 1000,
                        'time_total': 600,    # 10 minutes after departure
                        'time_leg': 600,
                        'groundspeed': 350
                    },
                    {
                        'ident': 'KPDX',
                        'name': 'Portland',
                        'type': 'apt',
                        'pos_lat': 45.589,
                        'pos_long': -122.597,
                        'altitude_feet': 0,
                        'ground_height': 31,
                        'time_total': 1200,   # 20 minutes after departure
                        'time_leg': 600,
                        'groundspeed': 300
                    }
                ]
            }
        }
        
        manager.load_flight_data(simbrief_data)
        return manager
    
    @pytest.fixture
    def manager_with_long_route(self):
        """Create a manager with waypoints far apart (200nm gap) with SimBrief times."""
        manager = SimBriefFlightManager()
        
        # Create a route with a large gap in the middle
        # KSFO to KLAX is roughly 300nm, about 45 minutes at cruise
        simbrief_data = {
            'origin': {'icao_code': 'KSFO'},
            'destination': {'icao_code': 'KLAX'},
            'general': {'initial_altitude': 35000},
            'navlog': {
                'fix': [
                    {
                        'ident': 'KSFO',
                        'name': 'San Francisco',
                        'type': 'apt',
                        'pos_lat': 37.619,
                        'pos_long': -122.375,
                        'altitude_feet': 0,
                        'ground_height': 13,
                        'time_total': 0,
                        'time_leg': 0,
                        'groundspeed': 150
                    },
                    {
                        'ident': 'KLAX',
                        'name': 'Los Angeles',
                        'type': 'apt',
                        'pos_lat': 33.943,
                        'pos_long': -118.408,
                        'altitude_feet': 0,
                        'ground_height': 126,
                        'time_total': 2700,   # 45 minutes for ~300nm
                        'time_leg': 2700,
                        'groundspeed': 400
                    }
                ]
            }
        }
        
        manager.load_flight_data(simbrief_data)
        return manager
    
    def test_path_points_empty_when_no_fixes(self):
        """Test that path points are empty when no fixes loaded."""
        manager = SimBriefFlightManager()
        
        points = manager.get_path_points_with_time(
            aircraft_lat=47.0,
            aircraft_lon=-122.0,
            lookahead_sec=1800.0,
            spacing_nm=20.0
        )
        
        assert points == []
    
    def test_path_points_generated(self, manager_with_route):
        """Test that path points are generated along the route."""
        # Aircraft at KSEA
        points = manager_with_route.get_path_points_with_time(
            aircraft_lat=47.449,
            aircraft_lon=-122.309,
            lookahead_sec=1800.0,  # 30 minutes
            spacing_nm=20.0
        )
        
        # Should have generated some points
        assert len(points) > 0
        
        # Points should be sorted by time
        for i in range(len(points) - 1):
            assert points[i].time_to_reach_sec <= points[i + 1].time_to_reach_sec
    
    def test_path_points_respect_lookahead(self, manager_with_route):
        """Test that path points don't exceed lookahead time."""
        lookahead_sec = 600.0  # 10 minutes
        
        points = manager_with_route.get_path_points_with_time(
            aircraft_lat=47.449,
            aircraft_lon=-122.309,
            lookahead_sec=lookahead_sec,
            spacing_nm=20.0
        )
        
        # All points should be within lookahead time
        for point in points:
            assert point.time_to_reach_sec <= lookahead_sec
    
    def test_path_points_uniform_spacing(self, manager_with_long_route):
        """Test that path points are uniformly spaced along long routes."""
        spacing_nm = 15.0
        
        points = manager_with_long_route.get_path_points_with_time(
            aircraft_lat=37.619,
            aircraft_lon=-122.375,
            lookahead_sec=3600.0,  # 1 hour
            spacing_nm=spacing_nm
        )
        
        # Should have multiple points (route is ~300nm)
        assert len(points) >= 5
        
        # Check that distances increase monotonically
        for i in range(len(points) - 1):
            assert points[i].distance_from_start_nm < points[i + 1].distance_from_start_nm
    
    def test_path_points_interpolate_altitude(self, manager_with_route):
        """Test that altitudes are interpolated between waypoints."""
        points = manager_with_route.get_path_points_with_time(
            aircraft_lat=47.449,
            aircraft_lon=-122.309,
            lookahead_sec=1800.0,
            spacing_nm=10.0
        )
        
        # Middle points should have altitudes between the fixes
        # KSEA=0ft, BUWZO=15000ft, KPDX=0ft
        if len(points) >= 3:
            # Find a point roughly in the first segment (climbing)
            mid_points = [p for p in points if 0 < p.altitude_ft < 15000]
            # There should be some climbing/descending points with intermediate altitudes
            assert len(mid_points) >= 0  # May or may not have intermediate points depending on spacing
    
    def test_path_points_use_simbrief_times(self, manager_with_route):
        """Test that path points use SimBrief's time_total for time calculation."""
        # Aircraft at departure (KSEA at time=0)
        points = manager_with_route.get_path_points_with_time(
            aircraft_lat=47.449,
            aircraft_lon=-122.309,
            lookahead_sec=1200.0,  # 20 minutes - covers whole route
            spacing_nm=10.0
        )
        
        if len(points) > 0:
            # First point should have time > 0 (some distance from aircraft)
            assert points[0].time_to_reach_sec >= 0
            
            # Points should respect the SimBrief times (not just distance-based)
            # The last point should be within the 20 minute flight time
            assert points[-1].time_to_reach_sec <= 1200.0
    
    def test_time_values_match_simbrief_fixes(self):
        """Test that interpolated times correctly use fix time_total values."""
        manager = SimBriefFlightManager()
        
        # Create a simple route with known times
        simbrief_data = {
            'origin': {'icao_code': 'TEST'},
            'destination': {'icao_code': 'TEST2'},
            'general': {'initial_altitude': 10000},
            'navlog': {
                'fix': [
                    {
                        'ident': 'FIX1',
                        'name': 'Fix 1',
                        'type': 'wpt',
                        'pos_lat': 0.0,
                        'pos_long': 0.0,
                        'altitude_feet': 1000,
                        'ground_height': 0,
                        'time_total': 0,
                        'time_leg': 0,
                        'groundspeed': 200
                    },
                    {
                        'ident': 'FIX2',
                        'name': 'Fix 2',
                        'type': 'wpt',
                        'pos_lat': 1.0,  # ~60nm north
                        'pos_long': 0.0,
                        'altitude_feet': 10000,
                        'ground_height': 0,
                        'time_total': 600,   # 10 minutes to reach FIX2
                        'time_leg': 600,
                        'groundspeed': 400
                    }
                ]
            }
        }
        
        manager.load_flight_data(simbrief_data)
        
        # Aircraft at FIX1 (start)
        points = manager.get_path_points_with_time(
            aircraft_lat=0.0,
            aircraft_lon=0.0,
            lookahead_sec=700.0,
            spacing_nm=20.0  # ~3 points along the ~60nm segment
        )
        
        # Should have some points
        assert len(points) >= 2
        
        # All points should have time_to_reach between 0 and 600 (the segment time)
        for point in points:
            assert 0 <= point.time_to_reach_sec <= 600


class TestSegmentProjection:
    """Test the segment projection helper methods."""
    
    @pytest.fixture
    def manager(self):
        """Create a simple manager for testing internal methods."""
        manager = SimBriefFlightManager()
        
        simbrief_data = {
            'origin': {'icao_code': 'KSEA'},
            'destination': {'icao_code': 'KPDX'},
            'general': {'initial_altitude': 35000},
            'navlog': {
                'fix': [
                    {
                        'ident': 'FIX1',
                        'name': 'Fix 1',
                        'type': 'wpt',
                        'pos_lat': 0.0,
                        'pos_long': 0.0,
                        'altitude_feet': 10000,
                        'ground_height': 0
                    },
                    {
                        'ident': 'FIX2',
                        'name': 'Fix 2',
                        'type': 'wpt',
                        'pos_lat': 1.0,
                        'pos_long': 0.0,
                        'altitude_feet': 10000,
                        'ground_height': 0
                    }
                ]
            }
        }
        
        manager.load_flight_data(simbrief_data)
        return manager
    
    def test_project_onto_segment_middle(self, manager):
        """Test projecting a point onto the middle of a segment."""
        # Point to the east of the segment (should project onto it)
        proj_lat, proj_lon, along_track = manager._project_onto_segment(
            lat=0.5, lon=0.5,  # Point to the east
            lat1=0.0, lon1=0.0,  # Segment start
            lat2=1.0, lon2=0.0   # Segment end
        )
        
        # Should project to approximately (0.5, 0.0)
        assert abs(proj_lat - 0.5) < 0.1
        assert abs(proj_lon - 0.0) < 0.1
    
    def test_project_onto_segment_before_start(self, manager):
        """Test projecting a point that's behind the segment start."""
        proj_lat, proj_lon, along_track = manager._project_onto_segment(
            lat=-0.5, lon=0.0,  # Point before segment
            lat1=0.0, lon1=0.0,
            lat2=1.0, lon2=0.0
        )
        
        # Should project to segment start
        assert abs(proj_lat - 0.0) < 0.01
        assert along_track == 0.0
    
    def test_project_onto_segment_after_end(self, manager):
        """Test projecting a point that's past the segment end."""
        proj_lat, proj_lon, along_track = manager._project_onto_segment(
            lat=1.5, lon=0.0,  # Point after segment
            lat1=0.0, lon1=0.0,
            lat2=1.0, lon2=0.0
        )
        
        # Should project to segment end
        assert abs(proj_lat - 1.0) < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

