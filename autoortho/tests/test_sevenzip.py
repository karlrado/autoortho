"""
Unit tests for the 7-Zip wrapper module (autoortho.utils.sevenzip).

This module replaces py7zr to avoid _zstd dependency issues on Linux with Python 3.14+.
"""

import os
import sys
import io
import tempfile
import subprocess
import struct
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.sevenzip import (
    get_7zip_binary,
    SevenZipFile,
    SevenZipFileInfo,
    SevenZipError,
    BytesIOFactory,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sevenzip_binary():
    """Get the 7-Zip binary path, skip if not found."""
    try:
        binary = get_7zip_binary()
        if not os.path.isfile(binary):
            pytest.skip(f"7-Zip binary not found at: {binary}")
        return binary
    except SevenZipError as e:
        pytest.skip(str(e))


@pytest.fixture
def temp_archive(sevenzip_binary, tmp_path):
    """Create a temporary 7z archive with test files."""
    # Create test files
    file1 = tmp_path / "test_file1.txt"
    file1.write_text("Hello, World!")
    
    file2 = tmp_path / "test_file2.txt"
    file2.write_text("This is a test file with some content.\n" * 10)
    
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    file3 = subdir / "nested_file.txt"
    file3.write_text("Nested file content")
    
    # Create 7z archive
    archive_path = tmp_path / "test_archive.7z"
    
    result = subprocess.run(
        [sevenzip_binary, "a", "-t7z", str(archive_path), 
         str(file1), str(file2), str(file3)],
        capture_output=True,
        cwd=str(tmp_path),
    )
    
    if result.returncode != 0:
        pytest.skip(f"Failed to create test archive: {result.stderr.decode()}")
    
    # Clean up source files so we're only testing extraction
    file1.unlink()
    file2.unlink()
    file3.unlink()
    subdir.rmdir()
    
    return archive_path, {
        "test_file1.txt": b"Hello, World!",
        "test_file2.txt": b"This is a test file with some content.\n" * 10,
        "subdir/nested_file.txt": b"Nested file content",
    }


# =============================================================================
# Tests for get_7zip_binary()
# =============================================================================

class TestGet7ZipBinary:
    """Tests for the get_7zip_binary() function."""
    
    def test_returns_string(self):
        """Should return a string path."""
        try:
            binary = get_7zip_binary()
            assert isinstance(binary, str)
        except SevenZipError:
            pytest.skip("7-Zip binary not available")
    
    def test_binary_exists(self, sevenzip_binary):
        """Binary path should point to an existing file."""
        assert os.path.isfile(sevenzip_binary)
    
    def test_binary_is_executable(self, sevenzip_binary):
        """Should be able to execute the binary."""
        result = subprocess.run(
            [sevenzip_binary],
            capture_output=True,
        )
        # 7-Zip returns non-zero when run without args, but should not crash
        assert result.returncode is not None


# =============================================================================
# Tests for SevenZipFileInfo
# =============================================================================

class TestSevenZipFileInfo:
    """Tests for the SevenZipFileInfo class."""
    
    def test_creation(self):
        """Should create info object with correct attributes."""
        info = SevenZipFileInfo("test.txt", size=1024, compressed=512)
        
        assert info.name == "test.txt"
        assert info.size == 1024
        assert info.uncompressed == 1024
        assert info.uncompressed_size == 1024
        assert info.compressed == 512
    
    def test_defaults(self):
        """Should have sensible defaults."""
        info = SevenZipFileInfo("test.txt")
        
        assert info.size == 0
        assert info.compressed == 0


# =============================================================================
# Tests for BytesIOFactory
# =============================================================================

class TestBytesIOFactory:
    """Tests for the BytesIOFactory class."""
    
    def test_creation(self):
        """Should create factory with products dict."""
        factory = BytesIOFactory()
        
        assert hasattr(factory, 'products')
        assert isinstance(factory.products, dict)
        assert len(factory.products) == 0
    
    def test_limit(self):
        """Should store limit value."""
        factory = BytesIOFactory(limit=1024 * 1024)
        
        assert factory.limit == 1024 * 1024
    
    def test_default_limit(self):
        """Should have sensible default limit."""
        factory = BytesIOFactory()
        
        assert factory.limit == 128 * 1024 * 1024  # 128 MiB


# =============================================================================
# Tests for SevenZipFile
# =============================================================================

class TestSevenZipFile:
    """Tests for the SevenZipFile class."""
    
    def test_context_manager(self, temp_archive):
        """Should work as a context manager."""
        archive_path, _ = temp_archive
        
        with SevenZipFile(str(archive_path), mode='r') as archive:
            assert archive is not None
    
    def test_getnames(self, temp_archive):
        """Should list all files in the archive."""
        archive_path, expected_files = temp_archive
        
        with SevenZipFile(str(archive_path), mode='r') as archive:
            names = archive.getnames()
        
        assert isinstance(names, list)
        assert len(names) == len(expected_files)
        
        # Check all expected files are present (with possible path variations)
        for expected_name in expected_files.keys():
            found = any(
                n == expected_name or n.endswith(expected_name) or expected_name.endswith(n)
                for n in names
            )
            assert found, f"Expected file '{expected_name}' not found in {names}"
    
    def test_getinfo(self, temp_archive):
        """Should return file info with size."""
        archive_path, expected_files = temp_archive
        
        with SevenZipFile(str(archive_path), mode='r') as archive:
            names = archive.getnames()
            
            for name in names:
                info = archive.getinfo(name)
                
                assert info is not None
                assert isinstance(info, SevenZipFileInfo)
                assert info.size >= 0
    
    def test_getinfo_nonexistent(self, temp_archive):
        """Should return None for nonexistent file."""
        archive_path, _ = temp_archive
        
        with SevenZipFile(str(archive_path), mode='r') as archive:
            info = archive.getinfo("nonexistent_file.txt")
        
        assert info is None
    
    def test_extract_to_bytes(self, temp_archive):
        """Should extract file content as bytes."""
        archive_path, expected_files = temp_archive
        
        with SevenZipFile(str(archive_path), mode='r') as archive:
            names = archive.getnames()
            
            for name in names:
                data = archive.extract_to_bytes(name)
                
                assert isinstance(data, bytes)
                assert len(data) > 0
    
    def test_extract_to_bytes_content(self, temp_archive):
        """Should extract correct file content."""
        archive_path, expected_files = temp_archive
        
        with SevenZipFile(str(archive_path), mode='r') as archive:
            names = archive.getnames()
            
            # Find and extract test_file1.txt
            for name in names:
                if "test_file1" in name:
                    data = archive.extract_to_bytes(name)
                    assert data == b"Hello, World!"
                    break
            else:
                pytest.fail("test_file1.txt not found in archive")
    
    def test_extract_with_factory(self, temp_archive):
        """Should extract to factory.products dict."""
        archive_path, expected_files = temp_archive
        
        with SevenZipFile(str(archive_path), mode='r') as archive:
            names = archive.getnames()
            
            factory = BytesIOFactory()
            archive.extract(targets=[names[0]], factory=factory)
            
            assert len(factory.products) == 1
            assert names[0] in factory.products
            
            # Product should be BytesIO-like
            bio = factory.products[names[0]]
            assert hasattr(bio, 'read')
            assert hasattr(bio, 'seek')
            assert hasattr(bio, 'size')
            
            # Should be able to read content
            bio.seek(0)
            data = bio.read()
            assert len(data) > 0
    
    def test_extract_factory_size_method(self, temp_archive):
        """Extracted BytesIO should have size() method for py7zr compatibility."""
        archive_path, _ = temp_archive
        
        with SevenZipFile(str(archive_path), mode='r') as archive:
            names = archive.getnames()
            
            factory = BytesIOFactory()
            archive.extract(targets=[names[0]], factory=factory)
            
            bio = factory.products[names[0]]
            
            # size() should return the total size
            size = bio.size()
            assert isinstance(size, int)
            assert size > 0
            
            # size() should not affect current position
            bio.seek(5)
            _ = bio.size()
            assert bio.tell() == 5
    
    def test_nonexistent_archive(self):
        """Should raise FileNotFoundError for nonexistent archive."""
        with pytest.raises(FileNotFoundError):
            SevenZipFile("/nonexistent/path/to/archive.7z", mode='r')
    
    def test_write_mode_not_supported(self, temp_archive):
        """Should raise error for write mode."""
        archive_path, _ = temp_archive
        
        with pytest.raises(SevenZipError):
            SevenZipFile(str(archive_path), mode='w')


# =============================================================================
# Integration test with real-world usage pattern
# =============================================================================

class TestPy7zrCompatibility:
    """Tests ensuring behavioral equivalence with py7zr."""
    
    def test_fileinfo_has_all_required_attributes(self, temp_archive):
        """SevenZipFileInfo must have all attributes that aoseasons.py checks."""
        archive_path, _ = temp_archive
        
        with SevenZipFile(str(archive_path), mode='r') as archive:
            names = archive.getnames()
            info = archive.getinfo(names[0])
            
            # aoseasons.py checks these attributes in order
            assert hasattr(info, 'uncompressed'), "Missing 'uncompressed' attribute"
            assert hasattr(info, 'uncompressed_size'), "Missing 'uncompressed_size' attribute"
            assert hasattr(info, 'size'), "Missing 'size' attribute"
            
            # All should return the same value
            assert info.uncompressed == info.uncompressed_size == info.size
    
    def test_nested_file_extraction(self, temp_archive):
        """Should correctly extract files in subdirectories."""
        archive_path, expected_files = temp_archive
        
        with SevenZipFile(str(archive_path), mode='r') as archive:
            names = archive.getnames()
            
            # Find nested file using aoseasons.py pattern
            search_name = "nested_file.txt"
            target_name = None
            for n in names:
                if n == search_name or n.endswith("/" + search_name) or n.endswith("\\" + search_name):
                    target_name = n
                    break
            
            assert target_name is not None, f"Could not find {search_name}"
            
            # Extract and verify content
            factory = BytesIOFactory()
            archive.extract(targets=[target_name], factory=factory)
            
            fobj = factory.products.get(target_name)
            assert fobj is not None
            
            fobj.seek(0)
            content = fobj.read()
            assert content == b"Nested file content"
    
    def test_getnames_returns_new_list(self, temp_archive):
        """getnames() should return a copy, not the internal list."""
        archive_path, _ = temp_archive
        
        with SevenZipFile(str(archive_path), mode='r') as archive:
            names1 = archive.getnames()
            names2 = archive.getnames()
            
            # Should be equal but not the same object
            assert names1 == names2
            assert names1 is not names2
            
            # Modifying one shouldn't affect the other
            names1.append("fake_file.txt")
            assert "fake_file.txt" not in names2
    
    def test_extract_preserves_factory_limit(self, temp_archive):
        """BytesIOFactory limit should be preserved."""
        archive_path, _ = temp_archive
        
        custom_limit = 256 * 1024 * 1024
        factory = BytesIOFactory(limit=custom_limit)
        
        assert factory.limit == custom_limit
        
        with SevenZipFile(str(archive_path), mode='r') as archive:
            names = archive.getnames()
            archive.extract(targets=[names[0]], factory=factory)
        
        # Limit should still be intact
        assert factory.limit == custom_limit


class TestBinaryDataIntegrity:
    """Tests to verify extracted data matches original byte-for-byte."""
    
    @pytest.fixture
    def binary_archive(self, sevenzip_binary, tmp_path):
        """Create archive with various binary content patterns."""
        # Create files with different binary patterns
        test_files = {
            # Random-looking binary data (non-ASCII, non-compressible)
            "binary_random.bin": bytes(range(256)) * 100,  # 25.6 KB
            # Structured binary (like DSF header)
            "structured.bin": b"XPLNEDSF" + struct.pack("<I", 1) + b"\x00" * 1000,
            # All zeros (compresses well)
            "zeros.bin": b"\x00" * 10000,
            # All 0xFF (edge case)
            "ones.bin": b"\xff" * 10000,
            # Mixed with newlines (could cause text-mode issues)
            "mixed_newlines.bin": b"data\r\n\x00\xff\r\nmore\n\x00data",
        }
        
        # Write test files
        for name, content in test_files.items():
            (tmp_path / name).write_bytes(content)
        
        # Create 7z archive
        archive_path = tmp_path / "binary_test.7z"
        file_list = list(test_files.keys())
        
        subprocess.run(
            [sevenzip_binary, 'a', str(archive_path)] + file_list,
            cwd=str(tmp_path),
            capture_output=True,
            check=True
        )
        
        return archive_path, test_files
    
    def test_binary_extraction_exact_match(self, binary_archive):
        """Extracted binary data must match original byte-for-byte."""
        archive_path, expected_files = binary_archive
        
        with SevenZipFile(str(archive_path), mode='r') as archive:
            for name, expected_content in expected_files.items():
                extracted = archive.extract_to_bytes(name)
                
                assert extracted == expected_content, (
                    f"Mismatch for {name}: "
                    f"expected {len(expected_content)} bytes, got {len(extracted)} bytes. "
                    f"First diff at byte {next((i for i, (a, b) in enumerate(zip(expected_content, extracted)) if a != b), 'N/A')}"
                )
    
    def test_binary_extraction_via_factory(self, binary_archive):
        """Factory extraction should also preserve binary data exactly."""
        archive_path, expected_files = binary_archive
        
        with SevenZipFile(str(archive_path), mode='r') as archive:
            for name, expected_content in expected_files.items():
                factory = BytesIOFactory()
                archive.extract(targets=[name], factory=factory)
                
                bio = factory.products[name]
                bio.seek(0)
                extracted = bio.read()
                
                assert extracted == expected_content, (
                    f"Factory mismatch for {name}: "
                    f"expected {len(expected_content)} bytes, got {len(extracted)} bytes"
                )
    
    def test_dsf_like_header_preserved(self, binary_archive):
        """Verify structured binary header is preserved (like DSF files)."""
        archive_path, expected_files = binary_archive
        
        with SevenZipFile(str(archive_path), mode='r') as archive:
            extracted = archive.extract_to_bytes("structured.bin")
            
            # Verify DSF-like header
            assert extracted[:8] == b"XPLNEDSF", "DSF magic header corrupted"
            version = struct.unpack("<I", extracted[8:12])[0]
            assert version == 1, f"Version field corrupted: expected 1, got {version}"


class TestIntegration:
    """Integration tests mimicking aoseasons.py usage pattern."""
    
    def test_aoseasons_pattern(self, temp_archive):
        """Test the extraction pattern used in aoseasons.py."""
        archive_path, expected_files = temp_archive
        
        with SevenZipFile(str(archive_path), mode='r') as archive:
            # 1. Get list of files
            arc_names = archive.getnames()
            assert len(arc_names) > 0
            
            # 2. Find target file (suffix matching like in aoseasons.py)
            target_name = None
            search_name = "test_file1.txt"
            for n in arc_names:
                if n == search_name or n.endswith("/" + search_name) or n.endswith(search_name):
                    target_name = n
                    break
            
            assert target_name is not None, f"Could not find {search_name}"
            
            # 3. Get file info for size
            info = archive.getinfo(target_name)
            
            uncompressed = None
            if info is not None:
                for attr in ("uncompressed", "uncompressed_size", "size"):
                    val = getattr(info, attr, None)
                    if isinstance(val, int) and val > 0:
                        uncompressed = val
                        break
            
            if not uncompressed:
                uncompressed = 128 * 1024 * 1024  # Fallback
            
            # 4. Extract to BytesIO via factory
            limit = int(uncompressed) + (1 * 1024 * 1024)
            factory = BytesIOFactory(limit=limit)
            archive.extract(targets=[target_name], factory=factory)
            
            fobj = factory.products.get(target_name)
            assert fobj is not None
            
            # 5. Read and verify content
            fobj.seek(0)
            content = fobj.read()
            assert content == b"Hello, World!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
