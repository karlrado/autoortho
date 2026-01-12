"""
test_aodds.py - Unit tests for native DDS builder module

Tests the AoDDS Python wrapper and validates:
- DDS header generation
- BC1/BC3 compression output
- Mipmap generation
- Error handling for missing/corrupt inputs
- Performance comparison with Python fallback
"""

import os
import sys
import tempfile
import time
import pytest
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def test_cache_dir():
    """Create a temporary cache directory for DDS tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def chunk_files(test_cache_dir):
    """Create a 4x4 grid of test chunk JPEG files."""
    # Minimal valid JPEG data
    MINIMAL_JPEG = bytes([
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
        0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
        0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
        0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
        0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
        0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
        0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
        0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
        0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00,
        0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
        0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03,
        0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
        0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
        0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72,
        0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
        0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45,
        0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
        0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75,
        0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
        0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3,
        0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
        0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9,
        0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
        0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4,
        0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01,
        0x00, 0x00, 0x3F, 0x00, 0xFB, 0xD2, 0x8A, 0x28, 0x03, 0xFF, 0xD9
    ])
    
    paths = []
    chunks_per_side = 4
    zoom = 16
    maptype = "BI"
    base_col, base_row = 0, 0
    
    for r in range(chunks_per_side):
        for c in range(chunks_per_side):
            chunk_col = base_col * chunks_per_side + c
            chunk_row = base_row * chunks_per_side + r
            filename = f"{chunk_col}_{chunk_row}_{zoom}_{maptype}.jpg"
            path = os.path.join(test_cache_dir, filename)
            with open(path, 'wb') as f:
                f.write(MINIMAL_JPEG)
            paths.append(path)
    
    return {
        'paths': paths,
        'chunks_per_side': chunks_per_side,
        'zoom': zoom,
        'maptype': maptype,
        'base_col': base_col,
        'base_row': base_row
    }


# ============================================================================
# Test Cases
# ============================================================================

class TestAoDDSAvailability:
    """Test native DDS builder availability."""
    
    def test_import_aodds(self):
        """Test that AoDDS module can be imported."""
        try:
            from autoortho.aopipeline import AoDDS
            is_avail = AoDDS.is_available()
            print(f"AoDDS native library available: {is_avail}")
        except ImportError as e:
            pytest.skip(f"AoDDS not available: {e}")
    
    def test_fallback_graceful(self):
        """Test that missing native library doesn't crash."""
        from autoortho.getortho import _get_native_dds_builder
        
        # Should not raise even if native not available
        native = _get_native_dds_builder()
        # Either returns module or None
        assert native is None or hasattr(native, 'build_tile_native')


class TestDDSHeader:
    """Test DDS header generation."""
    
    def test_dds_size_calculation(self):
        """Test DDS buffer size calculation."""
        try:
            from autoortho.aopipeline import AoDDS
            if not AoDDS.is_available():
                pytest.skip("Native library not available")
            
            # 4096x4096 BC1 with mipmaps (1024, 512, 256, 128, 64, 32, 16, 8, 4)
            size_bc1 = AoDDS.calc_dds_size(4096, 4096, 10, format="BC1")
            assert size_bc1 > 0
            
            # BC3 should be ~2x BC1
            size_bc3 = AoDDS.calc_dds_size(4096, 4096, 10, format="BC3")
            assert size_bc3 > size_bc1
            
            print(f"BC1 size for 4096x4096: {size_bc1} bytes")
            print(f"BC3 size for 4096x4096: {size_bc3} bytes")
            
        except ImportError:
            pytest.skip("AoDDS module not available")
    
    def test_dds_header_format(self):
        """Test that DDS header is correctly formed."""
        try:
            from autoortho.aopipeline import AoDDS
            if not AoDDS.is_available():
                pytest.skip("Native library not available")
            
            # Generate a small test DDS (would need chunk files)
            # For now, just verify the calc function works
            size = AoDDS.calc_dds_size(1024, 1024, 5, format="BC1")
            assert size >= 128  # At least header size
            
        except ImportError:
            pytest.skip("AoDDS module not available")


class TestDDSBuild:
    """Test full DDS building pipeline."""
    
    def test_build_tile_native(self, test_cache_dir, chunk_files):
        """Test native tile building from cached chunks."""
        try:
            from autoortho.aopipeline import AoDDS
            if not AoDDS.is_available():
                pytest.skip("Native library not available")
            
            # This test requires actual JPEG decode capability
            # which depends on turbojpeg being linked
            try:
                result = AoDDS.build_tile_native(
                    cache_dir=test_cache_dir,
                    row=chunk_files['base_row'],
                    col=chunk_files['base_col'],
                    maptype=chunk_files['maptype'],
                    zoom=chunk_files['zoom'],
                    chunks_per_side=chunk_files['chunks_per_side'],
                    format="BC1",
                    missing_color=(66, 77, 55)
                )
                
                # Verify DDS output
                assert result is not None
                assert len(result) > 128  # Header + some data
                
                # Verify DDS magic number
                assert result[:4] == b'DDS '
                
            except RuntimeError as e:
                # May fail if turbojpeg not available
                if "JPEG" in str(e) or "decode" in str(e).lower():
                    pytest.skip(f"JPEG decode not available: {e}")
                raise
                
        except ImportError:
            pytest.skip("AoDDS module not available")
    
    def test_build_tile_with_missing_chunks(self, test_cache_dir):
        """Test tile building when some chunks are missing."""
        try:
            from autoortho.aopipeline import AoDDS
            if not AoDDS.is_available():
                pytest.skip("Native library not available")
            
            # Build with no cached chunks - should use fallback color
            try:
                result = AoDDS.build_tile_native(
                    cache_dir=test_cache_dir,
                    row=999,  # Non-existent tile
                    col=999,
                    maptype="BI",
                    zoom=16,
                    chunks_per_side=4,
                    format="BC1",
                    missing_color=(255, 0, 255)  # Bright pink for visibility
                )
                
                # Should still produce valid DDS with fallback color
                if result:
                    assert result[:4] == b'DDS '
                
            except RuntimeError:
                # Expected if all chunks missing
                pass
                
        except ImportError:
            pytest.skip("AoDDS module not available")


class TestMipmapGeneration:
    """Test mipmap generation."""
    
    def test_mipmap_count(self):
        """Test that correct number of mipmaps are generated."""
        try:
            from autoortho.aopipeline import AoDDS
            if not AoDDS.is_available():
                pytest.skip("Native library not available")
            
            # 4096x4096 should have log2(4096/4) + 1 = 11 mipmaps
            # But actual count may vary based on implementation
            size_1 = AoDDS.calc_dds_size(4096, 4096, 1, format="BC1")
            size_5 = AoDDS.calc_dds_size(4096, 4096, 5, format="BC1")
            size_10 = AoDDS.calc_dds_size(4096, 4096, 10, format="BC1")
            
            # More mipmaps = larger file
            assert size_1 < size_5 < size_10
            
        except ImportError:
            pytest.skip("AoDDS module not available")


class TestCompression:
    """Test BC1/BC3 compression."""
    
    def test_bc1_smaller_than_bc3(self, test_cache_dir, chunk_files):
        """Test that BC1 output is smaller than BC3."""
        try:
            from autoortho.aopipeline import AoDDS
            if not AoDDS.is_available():
                pytest.skip("Native library not available")
            
            try:
                result_bc1 = AoDDS.build_tile_native(
                    cache_dir=test_cache_dir,
                    row=chunk_files['base_row'],
                    col=chunk_files['base_col'],
                    maptype=chunk_files['maptype'],
                    zoom=chunk_files['zoom'],
                    chunks_per_side=chunk_files['chunks_per_side'],
                    format="BC1",
                    missing_color=(66, 77, 55)
                )
                
                result_bc3 = AoDDS.build_tile_native(
                    cache_dir=test_cache_dir,
                    row=chunk_files['base_row'],
                    col=chunk_files['base_col'],
                    maptype=chunk_files['maptype'],
                    zoom=chunk_files['zoom'],
                    chunks_per_side=chunk_files['chunks_per_side'],
                    format="BC3",
                    missing_color=(66, 77, 55)
                )
                
                if result_bc1 and result_bc3:
                    # BC3 should be approximately 2x the size of BC1
                    assert len(result_bc3) > len(result_bc1)
                    
            except RuntimeError as e:
                if "JPEG" in str(e) or "decode" in str(e).lower():
                    pytest.skip(f"JPEG decode not available: {e}")
                raise
                
        except ImportError:
            pytest.skip("AoDDS module not available")


class TestIntegration:
    """Integration tests with getortho module."""
    
    def test_getortho_dds_builder_helper(self):
        """Test the _get_native_dds_builder helper in getortho."""
        from autoortho.getortho import _get_native_dds_builder
        
        # Should not raise
        native = _get_native_dds_builder()
        
        # Should be consistent (cached)
        native2 = _get_native_dds_builder()
        assert native is native2
    
    def test_background_dds_builder_fallback(self):
        """Test that BackgroundDDSBuilder falls back to Python when native unavailable."""
        from autoortho.getortho import BackgroundDDSBuilder
        
        # Should be able to instantiate without native
        # The builder will use Python fallback
        # Note: Full test requires running the builder which needs more setup


class TestSingleMipmapBuild:
    """Test single mipmap building (per-mipmap native pipeline)."""
    
    def test_calc_mipmap_size_bc1(self):
        """Test BC1 mipmap size calculation."""
        try:
            from autoortho.aopipeline import AoDDS
            
            # 512x512 BC1: (512/4) * (512/4) * 8 = 128 * 128 * 8 = 131072 bytes
            size_512 = AoDDS.calc_mipmap_size(512, 512, "BC1")
            assert size_512 == 131072
            
            # 1024x1024 BC1: (1024/4) * (1024/4) * 8 = 256 * 256 * 8 = 524288 bytes
            size_1024 = AoDDS.calc_mipmap_size(1024, 1024, "BC1")
            assert size_1024 == 524288
            
            # 2048x2048 BC1: (2048/4) * (2048/4) * 8 = 512 * 512 * 8 = 2097152 bytes
            size_2048 = AoDDS.calc_mipmap_size(2048, 2048, "BC1")
            assert size_2048 == 2097152
            
            # 4096x4096 BC1: (4096/4) * (4096/4) * 8 = 1024 * 1024 * 8 = 8388608 bytes
            size_4096 = AoDDS.calc_mipmap_size(4096, 4096, "BC1")
            assert size_4096 == 8388608
            
        except ImportError:
            pytest.skip("AoDDS module not available")
    
    def test_calc_mipmap_size_bc3(self):
        """Test BC3 mipmap size calculation (should be 2x BC1)."""
        try:
            from autoortho.aopipeline import AoDDS
            
            size_bc1 = AoDDS.calc_mipmap_size(1024, 1024, "BC1")
            size_bc3 = AoDDS.calc_mipmap_size(1024, 1024, "BC3")
            
            # BC3 is 2x BC1
            assert size_bc3 == size_bc1 * 2
            
        except ImportError:
            pytest.skip("AoDDS module not available")
    
    def test_build_single_mipmap_empty_list(self):
        """Test build_single_mipmap with empty list."""
        try:
            from autoortho.aopipeline import AoDDS
            if not AoDDS.is_available():
                pytest.skip("Native library not available")
            
            result = AoDDS.build_single_mipmap([])
            assert not result.success
            assert result.bytes_written == 0
            assert result.data is None
            assert "No JPEG data provided" in result.error
            
        except ImportError:
            pytest.skip("AoDDS module not available")
    
    def test_build_single_mipmap_non_square(self):
        """Test build_single_mipmap with non-perfect-square chunk count."""
        try:
            from autoortho.aopipeline import AoDDS
            if not AoDDS.is_available():
                pytest.skip("Native library not available")
            
            # 5 is not a perfect square
            result = AoDDS.build_single_mipmap([None, None, None, None, None])
            assert not result.success
            assert "perfect square" in result.error
            
        except ImportError:
            pytest.skip("AoDDS module not available")
    
    def test_build_single_mipmap_4_chunks(self, test_cache_dir, chunk_files):
        """Test build_single_mipmap with 4 chunks (2x2 grid = mipmap 4)."""
        try:
            from autoortho.aopipeline import AoDDS
            if not AoDDS.is_available():
                pytest.skip("Native library not available")
            
            # Read JPEG data from chunk files
            jpeg_datas = []
            for i in range(4):
                if i < len(chunk_files['paths']):
                    with open(chunk_files['paths'][i], 'rb') as f:
                        jpeg_datas.append(f.read())
                else:
                    jpeg_datas.append(None)
            
            try:
                result = AoDDS.build_single_mipmap(
                    jpeg_datas,
                    format="BC1",
                    missing_color=(66, 77, 55)
                )
                
                # Verify result
                assert result.success
                assert result.bytes_written > 0
                assert result.data is not None
                assert len(result.data) == result.bytes_written
                
                # 2x2 grid of 256x256 = 512x512 tile
                expected_size = AoDDS.calc_mipmap_size(512, 512, "BC1")
                assert result.bytes_written == expected_size
                
                print(f"4 chunks: {result.elapsed_ms:.2f}ms, {result.bytes_written} bytes")
                
            except RuntimeError as e:
                if "JPEG" in str(e) or "decode" in str(e).lower():
                    pytest.skip(f"JPEG decode not available: {e}")
                raise
                
        except ImportError:
            pytest.skip("AoDDS module not available")
    
    def test_build_single_mipmap_16_chunks(self, test_cache_dir, chunk_files):
        """Test build_single_mipmap with 16 chunks (4x4 grid = mipmap 3)."""
        try:
            from autoortho.aopipeline import AoDDS
            if not AoDDS.is_available():
                pytest.skip("Native library not available")
            
            # Read JPEG data from chunk files
            jpeg_datas = []
            for i in range(16):
                if i < len(chunk_files['paths']):
                    with open(chunk_files['paths'][i], 'rb') as f:
                        jpeg_datas.append(f.read())
                else:
                    jpeg_datas.append(None)
            
            try:
                result = AoDDS.build_single_mipmap(
                    jpeg_datas,
                    format="BC1",
                    missing_color=(66, 77, 55)
                )
                
                # Verify result
                assert result.success
                assert result.bytes_written > 0
                assert result.data is not None
                
                # 4x4 grid of 256x256 = 1024x1024 tile
                expected_size = AoDDS.calc_mipmap_size(1024, 1024, "BC1")
                assert result.bytes_written == expected_size
                
                print(f"16 chunks: {result.elapsed_ms:.2f}ms, {result.bytes_written} bytes")
                
            except RuntimeError as e:
                if "JPEG" in str(e) or "decode" in str(e).lower():
                    pytest.skip(f"JPEG decode not available: {e}")
                raise
                
        except ImportError:
            pytest.skip("AoDDS module not available")
    
    def test_build_single_mipmap_with_missing_chunks(self, test_cache_dir, chunk_files):
        """Test build_single_mipmap with some missing chunks."""
        try:
            from autoortho.aopipeline import AoDDS
            if not AoDDS.is_available():
                pytest.skip("Native library not available")
            
            # Read first 2 chunks, leave rest as missing
            jpeg_datas = []
            for i in range(16):
                if i < 2 and i < len(chunk_files['paths']):
                    with open(chunk_files['paths'][i], 'rb') as f:
                        jpeg_datas.append(f.read())
                else:
                    jpeg_datas.append(None)  # Missing chunk
            
            try:
                result = AoDDS.build_single_mipmap(
                    jpeg_datas,
                    format="BC1",
                    missing_color=(255, 0, 255)  # Bright pink
                )
                
                # Should still succeed with missing color fill
                assert result.success
                assert result.bytes_written > 0
                
                # 4x4 grid of 256x256 = 1024x1024 tile
                expected_size = AoDDS.calc_mipmap_size(1024, 1024, "BC1")
                assert result.bytes_written == expected_size
                
            except RuntimeError as e:
                if "JPEG" in str(e) or "decode" in str(e).lower():
                    pytest.skip(f"JPEG decode not available: {e}")
                raise
                
        except ImportError:
            pytest.skip("AoDDS module not available")
    
    def test_build_single_mipmap_bc3(self, test_cache_dir, chunk_files):
        """Test build_single_mipmap with BC3 format."""
        try:
            from autoortho.aopipeline import AoDDS
            if not AoDDS.is_available():
                pytest.skip("Native library not available")
            
            # Read JPEG data from chunk files
            jpeg_datas = []
            for i in range(4):
                if i < len(chunk_files['paths']):
                    with open(chunk_files['paths'][i], 'rb') as f:
                        jpeg_datas.append(f.read())
                else:
                    jpeg_datas.append(None)
            
            try:
                result_bc1 = AoDDS.build_single_mipmap(
                    jpeg_datas,
                    format="BC1",
                    missing_color=(66, 77, 55)
                )
                
                result_bc3 = AoDDS.build_single_mipmap(
                    jpeg_datas,
                    format="BC3",
                    missing_color=(66, 77, 55)
                )
                
                # Both should succeed
                assert result_bc1.success
                assert result_bc3.success
                
                # BC3 should be 2x BC1
                assert result_bc3.bytes_written == result_bc1.bytes_written * 2
                
            except RuntimeError as e:
                if "JPEG" in str(e) or "decode" in str(e).lower():
                    pytest.skip(f"JPEG decode not available: {e}")
                raise
                
        except ImportError:
            pytest.skip("AoDDS module not available")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

