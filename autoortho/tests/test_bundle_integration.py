"""
Integration tests for AOB2 bundle system.

Tests the interaction between bundle consolidation, reading, and DDS building.
"""

import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Sample JPEG data (minimal valid JPEG header + data)
SAMPLE_JPEG = bytes([
    0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
    0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
]) + bytes(100)


class TestBundleConsolidationFlow:
    """Tests for the complete consolidation workflow."""
    
    @pytest.fixture
    def cache_dir(self, tmp_path):
        """Create a temporary cache directory with sample JPEG files."""
        cache = tmp_path / "cache"
        cache.mkdir()
        
        # Create sample JPEG files for a 4x4 tile
        tile_row = 100
        tile_col = 200
        zoom = 16
        maptype = "BI"
        chunks_per_side = 4
        
        for i in range(chunks_per_side * chunks_per_side):
            chunk_row = i // chunks_per_side
            chunk_col = i % chunks_per_side
            abs_col = tile_col * chunks_per_side + chunk_col
            abs_row = tile_row * chunks_per_side + chunk_row
            
            # Create JPEG file
            jpeg_path = cache / f"{abs_col}_{abs_row}_{zoom}_{maptype}.jpg"
            with open(jpeg_path, 'wb') as f:
                f.write(SAMPLE_JPEG + bytes([i] * 50))  # Unique data per chunk
        
        return str(cache), tile_row, tile_col, zoom, maptype, chunks_per_side
    
    def test_consolidate_jpegs_to_bundle(self, cache_dir):
        """Test consolidating JPEG files into a bundle."""
        from autoortho.aopipeline.AoBundle2 import create_bundle_python, Bundle2Python
        from autoortho.utils.bundle_paths import get_bundle2_path
        
        cache, tile_row, tile_col, zoom, maptype, chunks_per_side = cache_dir
        
        # Create bundle
        output_path = get_bundle2_path(cache, tile_row, tile_col, maptype, zoom)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        result_path = create_bundle_python(
            cache_dir=cache,
            tile_row=tile_row,
            tile_col=tile_col,
            maptype=maptype,
            zoom=zoom,
            chunks_per_side=chunks_per_side,
            output_path=output_path
        )
        
        assert os.path.exists(result_path)
        
        # Verify bundle contents
        bundle = Bundle2Python(result_path)
        assert bundle.has_zoom(zoom)
        assert bundle.get_chunk_count(zoom) == chunks_per_side * chunks_per_side
        
        # Verify all chunks are present
        all_chunks = bundle.get_all_chunks(zoom)
        for i, chunk in enumerate(all_chunks):
            assert chunk is not None, f"Chunk {i} should not be None"
            assert len(chunk) > len(SAMPLE_JPEG), f"Chunk {i} should have extra data"
    
    def test_bundle_read_matches_individual_files(self, cache_dir):
        """Test that bundle read returns same data as individual files."""
        from autoortho.aopipeline.AoBundle2 import create_bundle_python, Bundle2Python
        from autoortho.utils.bundle_paths import get_bundle2_path
        
        cache, tile_row, tile_col, zoom, maptype, chunks_per_side = cache_dir
        
        # Read individual files first
        individual_data = []
        for i in range(chunks_per_side * chunks_per_side):
            chunk_row = i // chunks_per_side
            chunk_col = i % chunks_per_side
            abs_col = tile_col * chunks_per_side + chunk_col
            abs_row = tile_row * chunks_per_side + chunk_row
            
            jpeg_path = os.path.join(cache, f"{abs_col}_{abs_row}_{zoom}_{maptype}.jpg")
            with open(jpeg_path, 'rb') as f:
                individual_data.append(f.read())
        
        # Create bundle
        output_path = get_bundle2_path(cache, tile_row, tile_col, maptype, zoom)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        create_bundle_python(
            cache_dir=cache,
            tile_row=tile_row,
            tile_col=tile_col,
            maptype=maptype,
            zoom=zoom,
            chunks_per_side=chunks_per_side,
            output_path=output_path
        )
        
        # Read from bundle
        bundle = Bundle2Python(output_path)
        bundle_data = bundle.get_all_chunks(zoom)
        
        # Compare
        for i, (ind, bnd) in enumerate(zip(individual_data, bundle_data)):
            assert ind == bnd, f"Chunk {i} data mismatch"


class TestBundleFallback:
    """Tests for fallback behavior when bundles are unavailable."""
    
    def test_fallback_to_individual_files(self, tmp_path):
        """Test that missing bundle falls back to individual files."""
        from autoortho.utils.bundle_paths import get_bundle2_path, bundle_exists
        
        cache_dir = str(tmp_path / "cache")
        os.makedirs(cache_dir)
        
        # No bundle exists
        assert not bundle_exists(cache_dir, row=100, col=200, maptype="BI", zoom=16)
    
    def test_bundle_exists_check(self, tmp_path):
        """Test bundle existence check."""
        from autoortho.aopipeline.AoBundle2 import create_bundle_from_data_python
        from autoortho.utils.bundle_paths import get_bundle2_path, bundle_exists
        
        cache_dir = str(tmp_path / "cache")
        
        # No bundle initially
        assert not bundle_exists(cache_dir, 100, 200, "BI", 16)
        
        # Create bundle
        output_path = get_bundle2_path(cache_dir, 100, 200, "BI", 16)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        create_bundle_from_data_python(
            tile_row=100, tile_col=200, maptype="BI", zoom=16,
            jpeg_datas=[SAMPLE_JPEG] * 16,
            output_path=output_path
        )
        
        # Now bundle exists
        assert bundle_exists(cache_dir, 100, 200, "BI", 16)


class TestDSFOrganization:
    """Tests for DSF-based bundle organization."""
    
    def test_bundles_organized_by_dsf(self, tmp_path):
        """Test that bundles are created in DSF-based directory structure."""
        from autoortho.aopipeline.AoBundle2 import create_bundle_from_data_python
        from autoortho.utils.bundle_paths import get_bundle2_path, get_bundle2_dir
        
        cache_dir = str(tmp_path / "cache")
        
        # Create bundle
        output_path = get_bundle2_path(cache_dir, 26000, 10880, "BI", 16)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        create_bundle_from_data_python(
            tile_row=26000, tile_col=10880, maptype="BI", zoom=16,
            jpeg_datas=[SAMPLE_JPEG] * 16,
            output_path=output_path
        )
        
        # Check path structure
        assert "bundles" in output_path
        
        # Bundle directory should contain DSF-like structure with maptype folder
        bundle_dir = get_bundle2_dir(cache_dir, 26000, 10880, 16, "BI")
        path_parts = bundle_dir.split(os.sep)
        
        # Should have band, tile, and maptype directories
        assert any('+' in p or '-' in p for p in path_parts)
        assert "BI" in path_parts  # maptype is now a folder
    
    def test_enumerate_bundles(self, tmp_path):
        """Test bundle enumeration."""
        from autoortho.aopipeline.AoBundle2 import create_bundle_from_data_python
        from autoortho.utils.bundle_paths import (
            get_bundle2_path, enumerate_bundles
        )
        
        cache_dir = str(tmp_path / "cache")
        
        # Create multiple bundles
        for row, col in [(100, 200), (101, 200), (100, 201)]:
            output_path = get_bundle2_path(cache_dir, row, col, "BI", 16)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            create_bundle_from_data_python(
                tile_row=row, tile_col=col, maptype="BI", zoom=16,
                jpeg_datas=[SAMPLE_JPEG] * 16,
                output_path=output_path
            )
        
        # Enumerate
        bundles = enumerate_bundles(cache_dir)
        
        assert len(bundles) == 3
        
        rows = {b['row'] for b in bundles}
        cols = {b['col'] for b in bundles}
        
        assert rows == {100, 101}
        assert cols == {200, 201}


class TestConsolidatorIntegration:
    """Tests for consolidator integration with tile system."""
    
    def test_consolidator_creates_bundle_structure(self, tmp_path):
        """Test that consolidator creates proper directory structure."""
        from autoortho.aopipeline.bundle_consolidator import BundleConsolidator
        from autoortho.utils.bundle_paths import get_bundle2_dir
        
        cache_dir = str(tmp_path / "cache")
        os.makedirs(cache_dir)
        
        consolidator = BundleConsolidator(
            cache_dir=cache_dir,
            delete_jpegs=False,
            enabled=True
        )
        
        # Schedule consolidation (won't actually run without JPEGs)
        consolidator.schedule(row=100, col=200, maptype="BI", zoom=16)
        
        # Shutdown
        consolidator.shutdown()
    
    def test_consolidator_stats(self, tmp_path):
        """Test consolidator statistics tracking."""
        from autoortho.aopipeline.bundle_consolidator import BundleConsolidator
        
        cache_dir = str(tmp_path / "cache")
        os.makedirs(cache_dir)
        
        consolidator = BundleConsolidator(
            cache_dir=cache_dir,
            enabled=True
        )
        
        stats = consolidator.get_stats()
        
        assert 'bundles_created' in stats
        assert 'jpegs_consolidated' in stats
        assert 'errors' in stats
        
        consolidator.shutdown()


class TestMigrationFromJPEGs:
    """Tests for migrating from individual JPEGs to bundles."""
    
    def test_mixed_cache_state(self, tmp_path):
        """Test system works with mixed bundle and JPEG state."""
        from autoortho.aopipeline.AoBundle2 import (
            create_bundle_from_data_python, Bundle2Python
        )
        from autoortho.utils.bundle_paths import get_bundle2_path, bundle_exists
        
        cache_dir = str(tmp_path / "cache")
        os.makedirs(cache_dir)
        
        # Create some individual JPEGs
        for i in range(4):
            jpeg_path = os.path.join(cache_dir, f"{i}_0_16_BI.jpg")
            with open(jpeg_path, 'wb') as f:
                f.write(SAMPLE_JPEG)
        
        # Create a bundle for different tile
        bundle_path = get_bundle2_path(cache_dir, 100, 200, "BI", 16)
        os.makedirs(os.path.dirname(bundle_path), exist_ok=True)
        
        create_bundle_from_data_python(
            tile_row=100, tile_col=200, maptype="BI", zoom=16,
            jpeg_datas=[SAMPLE_JPEG] * 16,
            output_path=bundle_path
        )
        
        # Both should be accessible
        assert bundle_exists(cache_dir, 100, 200, "BI", 16)
        assert os.path.exists(os.path.join(cache_dir, "0_0_16_BI.jpg"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
