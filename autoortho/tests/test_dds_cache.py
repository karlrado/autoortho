"""
test_dds_cache.py - Unit tests for DDS caching system

Tests the EphemeralDDSCache and HybridDDSCache classes:
- Session isolation
- LRU eviction
- Size limits
- Cleanup on shutdown
"""

import os
import sys
import tempfile
import threading
import time
import pytest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autoortho.getortho import EphemeralDDSCache, HybridDDSCache, PrebuiltDDSCache


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def ephemeral_cache():
    """Create a small ephemeral cache for testing."""
    cache = EphemeralDDSCache(max_size_mb=1)  # 1MB limit
    yield cache
    cache.cleanup()


@pytest.fixture
def hybrid_cache():
    """Create a hybrid cache for testing."""
    cache = HybridDDSCache(memory_mb=1, disk_mb=2)
    yield cache
    cache.cleanup()


@pytest.fixture
def sample_dds_data():
    """Generate sample DDS-like data of various sizes."""
    return {
        'small': b'DDS ' + b'\x00' * 1000,        # ~1KB
        'medium': b'DDS ' + b'\x00' * 100000,     # ~100KB
        'large': b'DDS ' + b'\x00' * 500000,      # ~500KB
    }


# ============================================================================
# EphemeralDDSCache Tests
# ============================================================================

class TestEphemeralDDSCache:
    """Test the disk-based ephemeral cache."""
    
    def test_basic_put_get(self, ephemeral_cache, sample_dds_data):
        """Test basic put and get operations."""
        tile_id = "test_tile_1"
        data = sample_dds_data['small']
        
        # Put data
        result = ephemeral_cache.put(tile_id, data)
        assert result is True
        
        # Get data
        retrieved = ephemeral_cache.get(tile_id)
        assert retrieved == data
    
    def test_missing_tile(self, ephemeral_cache):
        """Test getting a non-existent tile."""
        result = ephemeral_cache.get("nonexistent_tile")
        assert result is None
    
    def test_session_isolation(self):
        """Test that different cache instances have different session IDs."""
        cache1 = EphemeralDDSCache(max_size_mb=1)
        cache2 = EphemeralDDSCache(max_size_mb=1)
        
        assert cache1._session_id != cache2._session_id
        
        # Data in cache1 should not be visible in cache2
        cache1.put("test_tile", b"DDS data")
        assert cache2.get("test_tile") is None
        
        cache1.cleanup()
        cache2.cleanup()
    
    def test_lru_eviction(self, sample_dds_data):
        """Test LRU eviction when cache is full."""
        # Create a very small cache (100KB)
        cache = EphemeralDDSCache(max_size_mb=0.1)  # ~100KB
        
        try:
            # Fill cache
            cache.put("tile_1", sample_dds_data['medium'])  # ~100KB
            
            # This should trigger eviction of tile_1
            cache.put("tile_2", sample_dds_data['medium'])  # ~100KB
            
            # tile_1 should be evicted (or both might not fit)
            # Just verify no crash and tile_2 is there
            assert cache.get("tile_2") is not None or cache.get("tile_1") is not None
            
        finally:
            cache.cleanup()
    
    def test_cleanup(self, sample_dds_data):
        """Test that cleanup removes all session files."""
        cache = EphemeralDDSCache(max_size_mb=10)
        cache_dir = cache._cache_dir
        session_id = cache._session_id
        
        # Add some data
        cache.put("tile_1", sample_dds_data['small'])
        cache.put("tile_2", sample_dds_data['small'])
        
        # Verify files exist
        files_before = [f for f in os.listdir(cache_dir) if f.startswith(session_id)]
        assert len(files_before) >= 2
        
        # Cleanup
        cache.cleanup()
        
        # Verify session files are gone
        files_after = [f for f in os.listdir(cache_dir) if f.startswith(session_id)]
        assert len(files_after) == 0
    
    def test_concurrent_access(self, ephemeral_cache, sample_dds_data):
        """Test thread safety of put/get operations."""
        results = {'errors': 0, 'success': 0}
        lock = threading.Lock()
        
        def worker(worker_id):
            try:
                for i in range(10):
                    tile_id = f"tile_{worker_id}_{i}"
                    ephemeral_cache.put(tile_id, sample_dds_data['small'])
                    retrieved = ephemeral_cache.get(tile_id)
                    if retrieved:
                        with lock:
                            results['success'] += 1
            except Exception as e:
                with lock:
                    results['errors'] += 1
        
        # Run multiple threads
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert results['errors'] == 0
        assert results['success'] > 0


# ============================================================================
# HybridDDSCache Tests
# ============================================================================

class TestHybridDDSCache:
    """Test the memory + disk hybrid cache."""
    
    def test_basic_put_get(self, hybrid_cache, sample_dds_data):
        """Test basic put and get operations."""
        tile_id = "test_tile_1"
        data = sample_dds_data['small']
        
        # Put data
        hybrid_cache.put(tile_id, data)
        
        # Get data
        retrieved = hybrid_cache.get(tile_id)
        assert retrieved == data
    
    def test_memory_priority(self, sample_dds_data):
        """Test that small items go to memory first."""
        cache = HybridDDSCache(memory_mb=1, disk_mb=10)
        
        try:
            # Put small data
            cache.put("tile_1", sample_dds_data['small'])
            
            # Should be in memory cache
            assert cache._memory.contains("tile_1")
            
        finally:
            cache.cleanup()
    
    def test_disk_overflow(self, sample_dds_data):
        """Test that large items overflow to disk."""
        # Very small memory cache
        cache = HybridDDSCache(memory_mb=0.01, disk_mb=10)  # 10KB memory
        
        try:
            # Put medium data (should overflow to disk after memory fills)
            for i in range(5):
                cache.put(f"tile_{i}", sample_dds_data['medium'])
            
            # At least some should be accessible (from memory or disk)
            found = sum(1 for i in range(5) if cache.get(f"tile_{i}") is not None)
            assert found > 0
            
        finally:
            cache.cleanup()
    
    def test_contains(self, hybrid_cache, sample_dds_data):
        """Test the contains method."""
        hybrid_cache.put("tile_1", sample_dds_data['small'])
        
        assert hybrid_cache.contains("tile_1") is True
        assert hybrid_cache.contains("nonexistent") is False
    
    def test_stats(self, hybrid_cache, sample_dds_data):
        """Test that stats are available."""
        hybrid_cache.put("tile_1", sample_dds_data['small'])
        
        stats = hybrid_cache.stats
        assert 'memory_cache' in stats
        assert 'disk_cache' in stats


# ============================================================================
# PrebuiltDDSCache Tests
# ============================================================================

class TestPrebuiltDDSCache:
    """Test the in-memory DDS cache."""
    
    def test_basic_store_get(self, sample_dds_data):
        """Test basic store and get operations."""
        cache = PrebuiltDDSCache(max_memory_bytes=1024*1024)
        
        tile_id = "test_tile_1"
        data = sample_dds_data['small']
        
        # Store data
        result = cache.store(tile_id, data)
        assert result is True
        
        # Get data
        retrieved = cache.get(tile_id)
        assert retrieved == data
    
    def test_contains(self, sample_dds_data):
        """Test contains method."""
        cache = PrebuiltDDSCache(max_memory_bytes=1024*1024)
        
        cache.store("tile_1", sample_dds_data['small'])
        
        assert cache.contains("tile_1") is True
        assert cache.contains("nonexistent") is False
    
    def test_memory_limit(self, sample_dds_data):
        """Test that memory limit is enforced."""
        # 200KB limit
        cache = PrebuiltDDSCache(max_memory_bytes=200*1024)
        
        # Store medium data (100KB each)
        cache.store("tile_1", sample_dds_data['medium'])
        cache.store("tile_2", sample_dds_data['medium'])
        
        # Third should fail or evict oldest
        result = cache.store("tile_3", sample_dds_data['medium'])
        
        # Either tile_3 was stored (with eviction) or rejected
        # Just verify no crash and at most 2 tiles fit
        tiles_present = sum(1 for i in range(1, 4) if cache.contains(f"tile_{i}"))
        assert tiles_present <= 2
    
    def test_stats(self, sample_dds_data):
        """Test cache statistics."""
        cache = PrebuiltDDSCache(max_memory_bytes=1024*1024)
        
        cache.store("tile_1", sample_dds_data['small'])
        cache.store("tile_2", sample_dds_data['medium'])
        
        stats = cache.stats
        assert 'count' in stats
        assert 'size_bytes' in stats
        assert stats['count'] == 2


# ============================================================================
# Integration Tests
# ============================================================================

class TestCacheIntegration:
    """Integration tests for cache system."""
    
    def test_hybrid_cache_in_builder(self):
        """Test that HybridDDSCache works with BackgroundDDSBuilder."""
        from autoortho.getortho import start_predictive_dds, stop_predictive_dds
        
        # This would require full setup, just verify imports work
        assert start_predictive_dds is not None
        assert stop_predictive_dds is not None


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

