"""
test_streaming_builder.py - Unit tests for the streaming DDS builder

Tests the C API wrapper (StreamingBuilder, BuilderPool) and the
FallbackResolver component.
"""

import os
import sys
import pytest
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestStreamingBuilderImport:
    """Test that the streaming builder modules can be imported."""
    
    def test_import_aodds(self):
        """Test importing AoDDS module."""
        from autoortho.aopipeline import AoDDS
        assert hasattr(AoDDS, 'StreamingBuilder')
        assert hasattr(AoDDS, 'StreamingBuilderPool')
        assert hasattr(AoDDS, 'get_default_builder_pool')
    
    def test_import_fallback_resolver(self):
        """Test importing FallbackResolver module."""
        from autoortho.aopipeline.fallback_resolver import FallbackResolver, TimeBudget
        assert FallbackResolver is not None
        assert TimeBudget is not None
    
    def test_builder_config_struct(self):
        """Test BuilderConfig structure exists."""
        from autoortho.aopipeline.AoDDS import BuilderConfig
        config = BuilderConfig(
            chunks_per_side=16,
            format=0,  # BC1
            missing_r=66,
            missing_g=77,
            missing_b=55
        )
        assert config.chunks_per_side == 16
        assert config.format == 0
    
    def test_builder_status_struct(self):
        """Test BuilderStatus structure exists."""
        from autoortho.aopipeline.AoDDS import BuilderStatus
        status = BuilderStatus()
        assert hasattr(status, 'chunks_total')
        assert hasattr(status, 'chunks_received')
        assert hasattr(status, 'chunks_decoded')


class TestTimeBudget:
    """Test the TimeBudget class."""
    
    def test_time_budget_creation(self):
        """Test creating a time budget."""
        from autoortho.aopipeline.fallback_resolver import TimeBudget
        
        budget = TimeBudget(5.0)
        assert budget.timeout == 5.0
        assert budget.remaining > 0
        assert not budget.exhausted
    
    def test_time_budget_exhaustion(self):
        """Test that budget exhausts after timeout."""
        from autoortho.aopipeline.fallback_resolver import TimeBudget
        
        budget = TimeBudget(0.1)  # 100ms budget
        time.sleep(0.15)  # Wait longer than budget
        
        assert budget.exhausted
        assert budget.remaining == 0
    
    def test_time_budget_elapsed(self):
        """Test elapsed time tracking."""
        from autoortho.aopipeline.fallback_resolver import TimeBudget
        
        budget = TimeBudget(10.0)
        time.sleep(0.1)
        
        assert budget.elapsed >= 0.1
        assert budget.elapsed < 10.0


class TestFallbackResolver:
    """Test the FallbackResolver class."""
    
    def test_resolver_creation(self, tmpdir):
        """Test creating a fallback resolver."""
        from autoortho.aopipeline.fallback_resolver import FallbackResolver
        
        resolver = FallbackResolver(
            cache_dir=str(tmpdir),
            maptype="BI",
            tile_col=1234,
            tile_row=5678,
            tile_zoom=16,
            fallback_level=2
        )
        
        assert resolver.cache_dir == str(tmpdir)
        assert resolver.maptype == "BI"
        assert resolver.tile_zoom == 16
        assert resolver.fallback_level == 2
    
    def test_resolver_no_fallbacks(self, tmpdir):
        """Test resolver with fallback_level=0 returns None."""
        from autoortho.aopipeline.fallback_resolver import FallbackResolver
        
        resolver = FallbackResolver(
            cache_dir=str(tmpdir),
            maptype="BI",
            tile_col=1234,
            tile_row=5678,
            tile_zoom=16,
            fallback_level=0
        )
        
        result = resolver.resolve(1234, 5678, 16)
        assert result is None
        assert resolver.stats['total_failed'] == 1
    
    def test_resolver_disk_cache_miss(self, tmpdir):
        """Test resolver when disk cache has no matching files."""
        from autoortho.aopipeline.fallback_resolver import FallbackResolver
        
        resolver = FallbackResolver(
            cache_dir=str(tmpdir),
            maptype="BI",
            tile_col=1234,
            tile_row=5678,
            tile_zoom=16,
            fallback_level=1  # Only cache + mipmap, no network
        )
        
        # Empty cache, should fail
        result = resolver.resolve(1234, 5678, 16)
        assert result is None
        assert resolver.stats['disk_cache_misses'] >= 1
    
    def test_resolver_stats(self, tmpdir):
        """Test resolver statistics tracking."""
        from autoortho.aopipeline.fallback_resolver import FallbackResolver
        
        resolver = FallbackResolver(
            cache_dir=str(tmpdir),
            maptype="BI",
            tile_col=1234,
            tile_row=5678,
            tile_zoom=16,
            fallback_level=1
        )
        
        # Call resolve multiple times
        resolver.resolve(1234, 5678, 16)
        resolver.resolve(1235, 5678, 16)
        resolver.resolve(1236, 5678, 16)
        
        stats = resolver.get_stats()
        assert 'total_failed' in stats
        assert 'disk_cache_misses' in stats
        assert stats['total_failed'] == 3
    
    def test_resolver_reset_stats(self, tmpdir):
        """Test resetting resolver statistics."""
        from autoortho.aopipeline.fallback_resolver import FallbackResolver
        
        resolver = FallbackResolver(
            cache_dir=str(tmpdir),
            maptype="BI",
            tile_col=1234,
            tile_row=5678,
            tile_zoom=16,
            fallback_level=1
        )
        
        resolver.resolve(1234, 5678, 16)
        assert resolver.stats['total_failed'] == 1
        
        resolver.reset_stats()
        assert resolver.stats['total_failed'] == 0


class TestStreamingBuilderPool:
    """Test the StreamingBuilderPool class."""
    
    def test_pool_creation(self):
        """Test creating a builder pool."""
        from autoortho.aopipeline.AoDDS import StreamingBuilderPool
        
        pool = StreamingBuilderPool(pool_size=2)
        assert pool._pool_size == 2
        assert pool.available_count == 0  # Not initialized yet
    
    def test_get_default_pool(self):
        """Test getting the default global pool."""
        from autoortho.aopipeline.AoDDS import get_default_builder_pool
        
        pool1 = get_default_builder_pool()
        pool2 = get_default_builder_pool()
        
        # Should return same instance
        assert pool1 is pool2


class TestStreamingBuilderResult:
    """Test the StreamingBuilderResult namedtuple."""
    
    def test_result_success(self):
        """Test creating a success result."""
        from autoortho.aopipeline.AoDDS import StreamingBuilderResult
        
        result = StreamingBuilderResult(success=True, bytes_written=12345)
        assert result.success is True
        assert result.bytes_written == 12345
        assert result.error == ""
    
    def test_result_failure(self):
        """Test creating a failure result."""
        from autoortho.aopipeline.AoDDS import StreamingBuilderResult
        
        result = StreamingBuilderResult(
            success=False,
            bytes_written=0,
            error="Test error"
        )
        assert result.success is False
        assert result.bytes_written == 0
        assert result.error == "Test error"


class TestBuilderPoolAcquisition:
    """Test builder pool acquisition behavior."""
    
    def test_acquire_with_config(self):
        """Test acquiring a builder with configuration."""
        from autoortho.aopipeline.AoDDS import StreamingBuilderPool
        
        pool = StreamingBuilderPool(pool_size=2)
        
        config = {
            'chunks_per_side': 16,
            'format': 'BC1',
            'missing_color': (66, 77, 55)
        }
        
        try:
            builder = pool.acquire(config=config, timeout=1.0)
            if builder is not None:
                # Builder acquired successfully
                assert hasattr(builder, 'add_chunk')
                assert hasattr(builder, 'finalize')
                builder.release()
        except ImportError:
            # Native library not available - skip
            pytest.skip("Native library not available")


class TestFallbackChainOrder:
    """Test that fallbacks are tried in the correct order."""
    
    def test_fallback_chain_components_exist(self):
        """Verify fallback resolver has all fallback methods."""
        from autoortho.aopipeline.fallback_resolver import FallbackResolver
        
        assert hasattr(FallbackResolver, '_try_disk_cache_fallback')
        assert hasattr(FallbackResolver, '_try_mipmap_scale_fallback')
        assert hasattr(FallbackResolver, '_try_network_fallback')
    
    def test_fallback_level_controls_chain(self, tmpdir):
        """Test that fallback_level controls which fallbacks are used."""
        from autoortho.aopipeline.fallback_resolver import FallbackResolver
        
        # Level 0: No fallbacks
        resolver0 = FallbackResolver(
            cache_dir=str(tmpdir), maptype="BI",
            tile_col=0, tile_row=0, tile_zoom=16,
            fallback_level=0
        )
        
        # Level 1: Cache + mipmap
        resolver1 = FallbackResolver(
            cache_dir=str(tmpdir), maptype="BI",
            tile_col=0, tile_row=0, tile_zoom=16,
            fallback_level=1
        )
        
        # Level 2: All fallbacks
        resolver2 = FallbackResolver(
            cache_dir=str(tmpdir), maptype="BI",
            tile_col=0, tile_row=0, tile_zoom=16,
            fallback_level=2
        )
        
        # All should fail with empty cache, but stats should differ
        resolver0.resolve(0, 0, 16)
        resolver1.resolve(0, 0, 16)
        resolver2.resolve(0, 0, 16)
        
        # Level 0 should not attempt disk cache
        assert resolver0.stats['disk_cache_misses'] == 0
        
        # Level 1 should attempt disk cache
        assert resolver1.stats['disk_cache_misses'] >= 1


class TestConfigOptions:
    """Test that new config options are properly defined."""
    
    def test_streaming_builder_config_exists(self):
        """Test that streaming builder config options are in defaults."""
        from autoortho.aoconfig import AOConfig
        
        cfg = AOConfig.__new__(AOConfig)
        cfg._defaults = AOConfig._defaults
        
        # Check that new options appear in defaults string
        assert 'streaming_builder_enabled' in cfg._defaults
        assert 'streaming_builder_pool_size' in cfg._defaults


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

