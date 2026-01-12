"""
Unit tests for AOB2 multi-zoom mutable bundle format.

Tests both native (C) and pure Python implementations.
"""

import os
import struct
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pytest


# Sample JPEG data (minimal valid JPEG header)
SAMPLE_JPEG = bytes([
    0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
    0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
]) + bytes(100)  # Add some padding


def create_sample_jpegs(count: int, base_size: int = 128) -> List[Optional[bytes]]:
    """Create sample JPEG data arrays for testing."""
    import random
    jpegs = []
    for i in range(count):
        # 90% chance of valid data, 10% missing
        if random.random() < 0.9:
            size = base_size + random.randint(0, 100)
            jpegs.append(SAMPLE_JPEG + bytes(random.randint(0, 255) for _ in range(size)))
        else:
            jpegs.append(None)
    return jpegs


class TestBundle2Python:
    """Tests for pure Python bundle reader."""
    
    def test_create_and_read_bundle(self, tmp_path):
        """Test creating and reading a bundle with pure Python."""
        from autoortho.aopipeline.AoBundle2 import (
            create_bundle_from_data_python, Bundle2Python,
            BUNDLE2_MAGIC, BUNDLE2_VERSION
        )
        
        bundle_path = str(tmp_path / "test.aob2")
        
        # Create test data
        chunk_count = 16  # 4x4 grid
        jpegs = create_sample_jpegs(chunk_count)
        
        # Create bundle
        result_path = create_bundle_from_data_python(
            tile_row=100,
            tile_col=200,
            maptype="BI",
            zoom=16,
            jpeg_datas=jpegs,
            output_path=bundle_path
        )
        
        assert os.path.exists(result_path)
        
        # Read bundle
        bundle = Bundle2Python(result_path)
        
        # Verify header
        assert bundle.header['magic'] == BUNDLE2_MAGIC
        assert bundle.header['version'] == BUNDLE2_VERSION
        assert bundle.header['tile_row'] == 100
        assert bundle.header['tile_col'] == 200
        assert bundle.maptype == "BI"
        
        # Verify zoom levels
        assert 16 in bundle.zoom_levels
        assert bundle.has_zoom(16)
        assert not bundle.has_zoom(15)
        
        # Verify chunk count
        assert bundle.get_chunk_count(16) == chunk_count
        
        # Verify chunk data
        for i, expected in enumerate(jpegs):
            actual = bundle.get_chunk(16, i)
            if expected is not None:
                assert actual is not None
                assert actual == expected
            else:
                assert actual is None
    
    def test_get_all_chunks(self, tmp_path):
        """Test retrieving all chunks at once."""
        from autoortho.aopipeline.AoBundle2 import (
            create_bundle_from_data_python, Bundle2Python
        )
        
        bundle_path = str(tmp_path / "test.aob2")
        chunk_count = 256  # 16x16 grid
        jpegs = create_sample_jpegs(chunk_count)
        
        create_bundle_from_data_python(
            tile_row=50, tile_col=60, maptype="EOX",
            zoom=17, jpeg_datas=jpegs, output_path=bundle_path
        )
        
        bundle = Bundle2Python(bundle_path)
        all_chunks = bundle.get_all_chunks(17)
        
        assert len(all_chunks) == chunk_count
        
        for i, (expected, actual) in enumerate(zip(jpegs, all_chunks)):
            if expected is not None:
                assert actual == expected
            else:
                assert actual is None
    
    def test_chunk_info(self, tmp_path):
        """Test chunk metadata retrieval."""
        from autoortho.aopipeline.AoBundle2 import (
            create_bundle_from_data_python, Bundle2Python, ChunkFlags
        )
        
        bundle_path = str(tmp_path / "test.aob2")
        jpegs = [SAMPLE_JPEG, None, SAMPLE_JPEG + b'extra']
        
        create_bundle_from_data_python(
            tile_row=1, tile_col=1, maptype="BI",
            zoom=14, jpeg_datas=jpegs, output_path=bundle_path
        )
        
        bundle = Bundle2Python(bundle_path)
        
        # Valid chunk
        info0 = bundle.get_chunk_info(14, 0)
        assert info0 is not None
        assert info0['size'] == len(SAMPLE_JPEG)
        assert info0['flags'] & ChunkFlags.VALID
        
        # Missing chunk
        info1 = bundle.get_chunk_info(14, 1)
        assert info1 is not None
        assert info1['size'] == 0
        assert info1['flags'] == ChunkFlags.MISSING
        
        # Another valid chunk
        info2 = bundle.get_chunk_info(14, 2)
        assert info2 is not None
        assert info2['size'] == len(SAMPLE_JPEG) + 5
    
    def test_invalid_bundle_magic(self, tmp_path):
        """Test that invalid magic number raises error."""
        from autoortho.aopipeline.AoBundle2 import Bundle2Python
        
        # Create file with invalid magic
        bundle_path = str(tmp_path / "invalid.aob2")
        with open(bundle_path, 'wb') as f:
            f.write(b'XXXX' + bytes(60))  # Invalid magic
        
        with pytest.raises(ValueError, match="Invalid magic"):
            Bundle2Python(bundle_path)
    
    def test_file_too_small(self, tmp_path):
        """Test that too-small file raises error."""
        from autoortho.aopipeline.AoBundle2 import Bundle2Python
        
        bundle_path = str(tmp_path / "small.aob2")
        with open(bundle_path, 'wb') as f:
            f.write(b'AOB2' + bytes(10))  # Too small
        
        with pytest.raises(ValueError, match="too small"):
            Bundle2Python(bundle_path)


class TestBundlePaths:
    """Tests for bundle path utilities."""
    
    def test_tile_to_lat_lon(self):
        """Test slippy tile to lat/lon conversion."""
        from autoortho.utils.bundle_paths import tile_to_lat_lon
        
        # Known test case: tile at ZL16
        lat, lon = tile_to_lat_lon(row=26000, col=10880, zoom=16)
        
        # Should be somewhere in the northern hemisphere, western hemisphere
        assert lat > 0  # Northern
        assert lon < 0  # Western
    
    def test_tile_to_dsf_coords(self):
        """Test conversion to DSF 1-degree coordinates."""
        from autoortho.utils.bundle_paths import tile_to_dsf_coords
        
        lat, lon = tile_to_dsf_coords(row=26000, col=10880, zoom=16)
        
        # Should be integers
        assert isinstance(lat, int)
        assert isinstance(lon, int)
    
    def test_get_bundle2_path(self):
        """Test bundle path generation."""
        from autoortho.utils.bundle_paths import get_bundle2_path
        
        path = get_bundle2_path(
            cache_dir="/cache",
            row=100, col=200,
            maptype="BI", zoom=16
        )
        
        # Should contain bundle directory structure with maptype as folder
        assert "bundles" in path
        assert path.endswith(".aob2")
        assert "100_200.aob2" in path
        assert "/BI/" in path or "\\BI\\" in path  # maptype is now a folder
    
    def test_get_bundle2_filename(self):
        """Test bundle filename generation."""
        from autoortho.utils.bundle_paths import get_bundle2_filename
        
        # Maptype is no longer in filename - it's in the parent directory
        filename = get_bundle2_filename(row=123, col=456)
        assert filename == "123_456.aob2"
    
    def test_parse_bundle_filename(self):
        """Test bundle filename parsing."""
        from autoortho.utils.bundle_paths import parse_bundle_filename
        
        # Maptype is no longer in filename - it's in the parent directory
        result = parse_bundle_filename("123_456.aob2")
        assert result == (123, 456)
        
        result = parse_bundle_filename("invalid")
        assert result is None
    
    def test_tile_row_col_to_dsf_key(self):
        """Test DSF key generation."""
        from autoortho.utils.bundle_paths import tile_row_col_to_dsf_key
        
        key = tile_row_col_to_dsf_key(row=26000, col=10880, zoom=16)
        
        # Should be formatted like "+37-122"
        assert key[0] in ('+', '-')
        assert len(key) == 8  # +XX-YYY


class TestBundle2:
    """Tests for Bundle2 wrapper class."""
    
    def test_bundle2_open(self, tmp_path):
        """Test Bundle2 wrapper opening."""
        from autoortho.aopipeline.AoBundle2 import (
            create_bundle_from_data_python, Bundle2
        )
        
        bundle_path = str(tmp_path / "test.aob2")
        jpegs = create_sample_jpegs(16)
        
        create_bundle_from_data_python(
            tile_row=1, tile_col=1, maptype="BI",
            zoom=16, jpeg_datas=jpegs, output_path=bundle_path
        )
        
        # Open with wrapper (will use Python fallback since native may not be built)
        bundle = Bundle2(bundle_path, use_native=False)
        
        all_chunks = bundle.get_all_chunks(16)
        assert len(all_chunks) == 16
        
        bundle.close()


class TestConsolidator:
    """Tests for bundle consolidator."""
    
    def test_consolidator_init(self, tmp_path):
        """Test consolidator initialization."""
        from autoortho.aopipeline.bundle_consolidator import BundleConsolidator
        
        consolidator = BundleConsolidator(
            cache_dir=str(tmp_path),
            delete_jpegs=False,
            max_workers=1,
            enabled=True
        )
        
        assert consolidator.enabled
        assert consolidator.get_pending_count() == 0
        
        consolidator.shutdown()
    
    def test_schedule_consolidation(self, tmp_path):
        """Test scheduling consolidation tasks."""
        from autoortho.aopipeline.bundle_consolidator import BundleConsolidator
        
        consolidator = BundleConsolidator(
            cache_dir=str(tmp_path),
            delete_jpegs=False,
            max_workers=1,
            enabled=True
        )
        
        # Schedule a task
        result = consolidator.schedule(row=100, col=200, maptype="BI", zoom=16)
        assert result is True
        
        # Same tile shouldn't be scheduled again
        result = consolidator.schedule(row=100, col=200, maptype="BI", zoom=16)
        assert result is False
        
        # Different tile should be scheduled
        result = consolidator.schedule(row=101, col=200, maptype="BI", zoom=16)
        assert result is True
        
        consolidator.shutdown(wait=False)
    
    def test_is_pending(self, tmp_path):
        """Test pending check."""
        from autoortho.aopipeline.bundle_consolidator import BundleConsolidator
        
        consolidator = BundleConsolidator(
            cache_dir=str(tmp_path),
            enabled=True
        )
        
        assert not consolidator.is_pending(100, 200, "BI")
        
        consolidator.schedule(row=100, col=200, maptype="BI", zoom=16)
        
        assert consolidator.is_pending(100, 200, "BI")
        
        consolidator.shutdown(wait=False)
    
    def test_disabled_consolidator(self, tmp_path):
        """Test that disabled consolidator doesn't schedule."""
        from autoortho.aopipeline.bundle_consolidator import BundleConsolidator
        
        consolidator = BundleConsolidator(
            cache_dir=str(tmp_path),
            enabled=False
        )
        
        result = consolidator.schedule(row=100, col=200, maptype="BI", zoom=16)
        assert result is False
        
        consolidator.shutdown()


class TestAoBundle2Module:
    """Tests for AoBundle2 module-level functions."""
    
    def test_is_available(self):
        """Test availability check."""
        from autoortho.aopipeline.AoBundle2 import is_available
        
        # Should return bool without crashing
        result = is_available()
        assert isinstance(result, bool)
    
    def test_bundle2_extension(self):
        """Test extension constant."""
        from autoortho.aopipeline.AoBundle2 import BUNDLE2_EXTENSION
        
        assert BUNDLE2_EXTENSION == ".aob2"
    
    def test_flags(self):
        """Test flag constants."""
        from autoortho.aopipeline.AoBundle2 import BundleFlags, ChunkFlags
        
        assert BundleFlags.MUTABLE == 0x0001
        assert BundleFlags.MULTI_ZOOM == 0x0002
        
        assert ChunkFlags.MISSING == 0x0000
        assert ChunkFlags.VALID == 0x0001
        assert ChunkFlags.GARBAGE == 0x0080


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
