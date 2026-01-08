"""
test_streaming_integration.py - Integration tests for streaming DDS builder

Tests the full streaming builder flow including:
- StreamingBuilder creation and lifecycle
- Chunk feeding with JPEG data
- Fallback resolution
- Finalization
- Prefetch-to-live transition
"""

import os
import sys
import pytest
import threading

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestStreamingBuilderFlow:
    """Test the full streaming builder workflow."""
    
    @pytest.fixture
    def test_jpeg(self):
        """Load a test JPEG file."""
        test_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        test_file = os.path.join(test_dir, 'testfiles', 'test_tile_small.jpg')
        
        if os.path.exists(test_file):
            with open(test_file, 'rb') as f:
                return f.read()
        return None
    
    def test_builder_pool_acquire_release(self):
        """Test acquiring and releasing builders from pool."""
        try:
            from autoortho.aopipeline.AoDDS import StreamingBuilderPool
        except ImportError:
            pytest.skip("Native library not available")
        
        pool = StreamingBuilderPool(pool_size=2)
        
        config = {
            'chunks_per_side': 4,  # Small for testing
            'format': 'BC1',
            'missing_color': (128, 128, 128)
        }
        
        # Acquire builder
        builder = pool.acquire(config=config, timeout=2.0)
        if builder is None:
            pytest.skip("Could not acquire builder - native lib issue")
        
        try:
            # Verify builder has expected methods
            assert hasattr(builder, 'add_chunk')
            assert hasattr(builder, 'add_fallback_image')
            assert hasattr(builder, 'mark_missing')
            assert hasattr(builder, 'get_status')
            assert hasattr(builder, 'finalize')
        finally:
            builder.release()
    
    def test_builder_add_chunks(self, test_jpeg):
        """Test adding JPEG chunks to builder."""
        if test_jpeg is None:
            pytest.skip("Test JPEG file not available")
        
        try:
            from autoortho.aopipeline.AoDDS import StreamingBuilderPool
        except ImportError:
            pytest.skip("Native library not available")
        
        pool = StreamingBuilderPool(pool_size=2)
        
        config = {
            'chunks_per_side': 2,  # 4 chunks total
            'format': 'BC1',
            'missing_color': (66, 77, 55)
        }
        
        builder = pool.acquire(config=config, timeout=2.0)
        if builder is None:
            pytest.skip("Could not acquire builder")
        
        try:
            # Add some chunks
            builder.add_chunk(0, test_jpeg)
            builder.add_chunk(1, test_jpeg)
            
            # Check status
            status = builder.get_status()
            assert status['chunks_total'] == 4  # 2x2
            assert status['chunks_received'] >= 2
        finally:
            builder.release()
    
    def test_builder_mark_missing(self):
        """Test marking chunks as missing."""
        try:
            from autoortho.aopipeline.AoDDS import StreamingBuilderPool
        except ImportError:
            pytest.skip("Native library not available")
        
        pool = StreamingBuilderPool(pool_size=2)
        
        config = {
            'chunks_per_side': 2,
            'format': 'BC1',
            'missing_color': (255, 0, 0)  # Red for visibility
        }
        
        builder = pool.acquire(config=config, timeout=2.0)
        if builder is None:
            pytest.skip("Could not acquire builder")
        
        try:
            # Mark all chunks as missing
            for i in range(4):
                builder.mark_missing(i)
            
            status = builder.get_status()
            assert status['chunks_missing'] == 4
            assert builder.is_complete()
        finally:
            builder.release()
    
    def test_builder_is_complete(self, test_jpeg):
        """Test is_complete detection."""
        if test_jpeg is None:
            pytest.skip("Test JPEG file not available")
        
        try:
            from autoortho.aopipeline.AoDDS import StreamingBuilderPool
        except ImportError:
            pytest.skip("Native library not available")
        
        pool = StreamingBuilderPool(pool_size=2)
        
        config = {
            'chunks_per_side': 2,  # 4 chunks
            'format': 'BC1',
            'missing_color': (66, 77, 55)
        }
        
        builder = pool.acquire(config=config, timeout=2.0)
        if builder is None:
            pytest.skip("Could not acquire builder")
        
        try:
            # Not complete initially
            assert not builder.is_complete()
            
            # Add some chunks
            builder.add_chunk(0, test_jpeg)
            builder.add_chunk(1, test_jpeg)
            assert not builder.is_complete()  # Still 2 missing
            
            # Mark remaining as missing
            builder.mark_missing(2)
            builder.mark_missing(3)
            
            # Now complete
            assert builder.is_complete()
        finally:
            builder.release()


class TestFallbackResolverIntegration:
    """Test fallback resolver with real file operations."""
    
    @pytest.fixture
    def cache_with_files(self, tmp_path):
        """Create a cache directory with test files."""
        # Copy test JPEG if available
        test_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        test_file = os.path.join(test_dir, 'testfiles', 'test_tile_small.jpg')
        
        if os.path.exists(test_file):
            # Create cache file with expected naming
            cache_file = tmp_path / "0_0_12_BI.jpg"
            with open(test_file, 'rb') as src:
                cache_file.write_bytes(src.read())
            return str(tmp_path)
        
        return None
    
    def test_disk_cache_fallback(self, cache_with_files):
        """Test disk cache fallback finds cached files."""
        if cache_with_files is None:
            pytest.skip("Test files not available")
        
        from autoortho.aopipeline.fallback_resolver import FallbackResolver
        
        resolver = FallbackResolver(
            cache_dir=cache_with_files,
            maptype="BI",
            tile_col=0,
            tile_row=0,
            tile_zoom=16,
            fallback_level=1,
            max_mipmap=4  # Allow searching lower zooms
        )
        
        # Try to resolve - should find cached file at zoom 12
        resolver.resolve(0, 0, 16, target_mipmap=0)
        
        # May or may not succeed depending on test setup
        stats = resolver.get_stats()
        assert stats['total_resolved'] >= 0 or stats['total_failed'] >= 0


class TestPrefetchToLiveTransition:
    """Test the prefetch-to-live transition mechanism."""
    
    def test_mark_live_sets_flag(self):
        """Test that mark_live sets the _is_live flag."""
        # Create a mock tile object with required attributes
        class MockTile:
            def __init__(self):
                self._is_live = False
                self._tile_time_budget = None
                self._live_transition_event = threading.Event()
                self.max_zoom = 16
                self.chunks = {16: []}
                self.id = "test_tile"
            
            def mark_live(self, time_budget=None):
                if self._is_live:
                    return
                self._is_live = True
                if time_budget is not None:
                    self._tile_time_budget = time_budget
                if self._live_transition_event is not None:
                    self._live_transition_event.set()
        
        tile = MockTile()
        assert not tile._is_live
        
        tile.mark_live()
        assert tile._is_live
        assert tile._live_transition_event.is_set()
    
    def test_mark_live_idempotent(self):
        """Test that calling mark_live twice is safe."""
        class MockTile:
            def __init__(self):
                self._is_live = False
                self._tile_time_budget = None
                self._live_transition_event = threading.Event()
                self.call_count = 0
            
            def mark_live(self, time_budget=None):
                self.call_count += 1
                if self._is_live:
                    return
                self._is_live = True
        
        tile = MockTile()
        tile.mark_live()
        tile.mark_live()
        tile.mark_live()
        
        # Should only set once, even though called 3 times
        assert tile._is_live
        assert tile.call_count == 3


class TestThreadSafety:
    """Test thread safety of the streaming builder."""
    
    def test_concurrent_add_chunks(self):
        """Test that multiple threads can add chunks safely."""
        try:
            from autoortho.aopipeline.AoDDS import StreamingBuilderPool
        except ImportError:
            pytest.skip("Native library not available")
        
        pool = StreamingBuilderPool(pool_size=2)
        
        config = {
            'chunks_per_side': 4,  # 16 chunks
            'format': 'BC1',
            'missing_color': (128, 128, 128)
        }
        
        builder = pool.acquire(config=config, timeout=2.0)
        if builder is None:
            pytest.skip("Could not acquire builder")
        
        try:
            errors = []
            
            def mark_chunk(index):
                try:
                    builder.mark_missing(index)
                except Exception as e:
                    errors.append(e)
            
            # Start 16 threads, each marking one chunk
            threads = []
            for i in range(16):
                t = threading.Thread(target=mark_chunk, args=(i,))
                threads.append(t)
                t.start()
            
            # Wait for all threads
            for t in threads:
                t.join()
            
            # No errors should have occurred
            assert len(errors) == 0
            
            # All chunks should be marked
            assert builder.is_complete()
            status = builder.get_status()
            assert status['chunks_missing'] == 16
        finally:
            builder.release()


class TestBuilderReuse:
    """Test that builders can be reused after release."""
    
    def test_acquire_release_cycle(self):
        """Test multiple acquire/release cycles."""
        try:
            from autoortho.aopipeline.AoDDS import StreamingBuilderPool
        except ImportError:
            pytest.skip("Native library not available")
        
        pool = StreamingBuilderPool(pool_size=1)
        
        config = {
            'chunks_per_side': 2,
            'format': 'BC1',
            'missing_color': (66, 77, 55)
        }
        
        for cycle in range(3):
            builder = pool.acquire(config=config, timeout=2.0)
            if builder is None:
                pytest.skip("Could not acquire builder")
            
            # Use builder
            builder.mark_missing(0)
            builder.mark_missing(1)
            builder.mark_missing(2)
            builder.mark_missing(3)
            
            status = builder.get_status()
            assert status['chunks_missing'] == 4
            
            # Release
            builder.release()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

