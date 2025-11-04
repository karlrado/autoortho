#!/usr/bin/env python3

import gc
import os
import time
import pytest
import psutil
import shutil

import logging
logging.basicConfig(level=logging.DEBUG)

import requests

import getortho

#getortho.ISPC = False
maptypes = ['BI', 'GO2', 'NAIP', 'EOX', 'USGS', 'ARC', 'Firefly']


@pytest.fixture
def chunk(tmpdir):
    return getortho.Chunk(2176, 3232, 'EOX', 13, cache_dir=tmpdir)

def test_chunk_get(chunk):
    ret = chunk.get()
    assert ret == True

def test_chunk_getter(tmpdir):
    c = getortho.Chunk(2176, 3232, 'EOX', 13, cache_dir=tmpdir)
    getortho.chunk_getter.submit(c)
    ready = c.ready.wait(5)
    assert ready == True


@pytest.mark.parametrize("maptype", maptypes)
def test_maptype_chunk(maptype, tmpdir):
    c = getortho.Chunk(2176, 3232, maptype, 13, cache_dir=tmpdir)
    ret = c.get()
    assert ret
    assert getortho._is_jpeg(c.data[:3])
   
    session = requests.Session()
    c = getortho.Chunk(2176, 3264, maptype, 13, cache_dir=tmpdir)
    ret = c.get(session=session)
    assert ret
    assert getortho._is_jpeg(c.data[:3])


@pytest.fixture
def tile(tmpdir):
    t = getortho.Tile(2176, 3232, 'EOX', 13, cache_dir=tmpdir)
    return t

def test_get_bytes(tmpdir):
    tile = getortho.Tile(2176, 3232, 'BI', 13, cache_dir=tmpdir)
    # Requesting just more than a 4x4 even row of blocks worth
    ret = tile.get_bytes(0, 131208)
    assert ret
    
    testfile = tile.write()
    with open(testfile, 'rb') as h:
        h.seek(128)
        data = h.read(8)
        # Verify that we still get data for the read on this odd row
        h.seek(131200)
        mmdata = h.read(8)
    assert data != b'\x00'*8
    assert mmdata != b'\x00'*8
    #assert True == False


def test_get_bytes_mip1(tmpdir):
    tile = getortho.Tile(2176, 3232, 'BI', 13, cache_dir=tmpdir)
    #ret = tile.get_bytes(8388672, 4194304)
    mmstart = tile.dds.mipmap_list[1].startpos
    ret = tile.get_bytes(mmstart, 1024)
    assert ret
    
    testfile = tile.write()
    with open(testfile, 'rb') as h:
        h.seek(mmstart)
        data = h.read(8)

    assert data != b'\x00'*8


def test_get_bytes_mip_end(tmpdir):
    tile = getortho.Tile(2176, 3232, 'BI', 13, cache_dir=tmpdir)
    #ret = tile.get_bytes(8388672, 4194304)
    mmend = tile.dds.mipmap_list[0].endpos
    ret = tile.get_bytes(mmend-1024, 1024)
    assert ret
    
    testfile = tile.write()
    with open(testfile, 'rb') as h:
        #h.seek(20709504)
        h.seek(mmend-1024)
        data = h.read(8)

    assert data != b'\x00'*8


def test_get_bytes_mip_span(tmpdir):
    tile = getortho.Tile(2176, 3232, 'BI', 13, cache_dir=tmpdir)
    #ret = tile.get_bytes(8388672, 4194304)
    mm0end = tile.dds.mipmap_list[0].endpos
    mm1start = tile.dds.mipmap_list[1].startpos
    ret = tile.get_bytes(mm0end-16384, 32768)
    assert ret
    
    testfile = tile.write()
    with open(testfile, 'rb') as h:
        #h.seek(20709504)

        h.seek(mm0end-16384)
        data0 = h.read(8)
        h.seek(mm1start)
        data1 = h.read(8)

    assert data0 != b'\x00'*8
    assert data1 == b'\x00'*8


def test_get_bytes_row_span(tmpdir):
    tile = getortho.Tile(2176, 3232, 'BI', 13, cache_dir=tmpdir)
    #ret = tile.get_bytes(8388672, 4194304)
    mm1start = tile.dds.mipmap_list[1].startpos
    ret = tile.get_bytes(mm1start + 261144, 4096)
    assert ret
    
    testfile = tile.write()
    with open(testfile, 'rb') as h:
        h.seek(mm1start + 262144)
        data = h.read(8)

    assert data != b'\x00'*8


def test_find_mipmap_pos():
    tile = getortho.Tile(2176, 3232, 'BI', 13)

    mm0start = tile.dds.mipmap_list[0].startpos
    m = tile.find_mipmap_pos(mm0start + 1)
    assert m == 0

    mm1start = tile.dds.mipmap_list[1].startpos
    m = tile.find_mipmap_pos(mm1start + 262144)
    assert m == 1

    mm2start = tile.dds.mipmap_list[2].startpos
    m = tile.find_mipmap_pos(mm2start + 32)
    assert m == 2


def test_read_bytes(tmpdir):
    tile = getortho.Tile(2176, 3232, 'BI', 13, cache_dir=tmpdir)
    data0 = tile.read_dds_bytes(0, 131073)
    assert data0[128:136] != b'\x00'*8
    data1 = tile.read_dds_bytes(131073,100000)
    assert data1[0:7] != b'\x00*8'
   
    print(len(data0))
    with open(f"{tmpdir}/readtest.dds", 'wb') as h:
        h.write(data0)
        #h.write(data1)

    testfile = tile.write()
    with open(testfile, 'rb') as h:
        h.seek(131073)
        filedata = h.read(8)

    assert data1[0:8] == filedata    


def test_get_mipmap(tmpdir):
    tile = getortho.Tile(2176, 3232, 'BI', 13, cache_dir=tmpdir)
    tile.min_zoom = 5
    ret = tile.get_mipmap(6)
    testfile = tile.write()
    assert ret


def test_get_bytes_all(tmpdir):
    tile = getortho.Tile(2176, 3232, 'BI', 13, cache_dir=tmpdir)
    ret = tile.get_bytes(0, 131072)
    #ret = tile.get()
    testfile = tile.write()
    assert ret

def test_get_header(tmpdir):
    tile = getortho.Tile(2176, 3232, 'BI', 13, cache_dir=tmpdir)
    ret = tile.get_header()
    assert ret

def _test_get_BI_tile(tmpdir):
    tile = getortho.Tile(2176, 3232, 'BI', 13, cache_dir=tmpdir)
    ret = tile.get()
    assert ret

def test_tile_fetch(tmpdir):
    tile = getortho.Tile(2176, 3232, 'EOX', 13, cache_dir=tmpdir)
    ret = tile.fetch()
    assert ret == True
    assert len(tile.chunks[13]) == (tile.width * tile.height)
    #getortho.chunk_getter.stop() 
    #time.sleep(10)

def _test_tile_fetch_many(tmpdir):
    start_col = 2176
    start_row = 3232

    #for c in range(2176, 2432, 16):
    #    for r in range(3232, 3488, 16):
    for c in range(2176, 2200, 16):
        for r in range(3232, 3264, 16):
            t = getortho.Tile(c, r, 'BI', 13, cache_dir=tmpdir)
            t.get()
            #t.fetch()
            #print(len(t.chunks))

    #assert True == False


def _test_tile_quick_zoom(tmpdir):
    t = getortho.Tile(2176, 3232, 'EOX', 13, cache_dir=tmpdir)
    t.get(quick_zoom=10)
    t.get(quick_zoom=11)
    t.get(quick_zoom=12)
    t.get()
    #assert True == False

def _test_tile_get(tile):
    ret = tile.get()
    assert ret


def _test_tile_mem(tmpdir):
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss
    t = getortho.Tile(2176, 3232, 'EOX', 13, cache_dir=tmpdir)
    t.get_mipmap(0)
    time.sleep(2)
    mip0_mem = process.memory_info().rss
    print(f"{start_mem} {mip0_mem}  used:  {(mip0_mem - start_mem)/pow(2,20)} MB")
    assert True == False


def _test_tile_close(tmpdir):
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss
    t = getortho.Tile(2176, 3232, 'EOX', 13, cache_dir=tmpdir)
    t.get()
    get_mem = process.memory_info().rss
    t.close()
    del(t)
    gc.collect()
    time.sleep(5)
    close_mem = process.memory_info().rss
    print(f"S: {start_mem} G: {get_mem} C: {close_mem}.  Diff {close_mem-start_mem}")
    t = getortho.Tile(2176, 3232, 'EOX', 13, cache_dir=tmpdir)
    t.get()
    get_mem = process.memory_info().rss
    t.close()
    del(t)
    gc.collect()
    time.sleep(5)
    close_mem = process.memory_info().rss
    print(f"S: {start_mem} G: {get_mem} C: {close_mem}.  Diff {close_mem-start_mem}")

#def test_map(tmpdir):
#    m = getortho.Map(cache_dir=tmpdir)
#    ret = m.get_tiles(2176, 3232, 'EOX', 13)
#    assert ret

# def test_map_background(tmpdir):
#     m = getortho.Map(cache_dir=tmpdir)
#     start_c = 2176
#     start_r = 3232
#     num_c = 2
#     num_r = 1
#     for c in range(start_c, (start_c + num_c*16), 16):
#         for r in range(start_r, (start_r + num_r*16), 16):
#             ret = m.get_tiles(c, r, 'EOX', 13, background=True)
#     
#     for t in m.tiles:
#         print(f"Waiting on {t}")
#         ret = t.ready.wait(600)
#         assert ret == True
#         assert len(t.chunks[13]) == 256
# 
#     files = os.listdir(tmpdir)
#     assert len(m.tiles) == len(files)

def test_get_bytes_mm4_mm0(tmpdir):
    tile = getortho.Tile(17408, 25856, 'BI', 16, cache_dir=tmpdir)
    #tile = getortho.Tile(21760, 32320, 'BI', 16, cache_dir=tmpdir)
    #tile = getortho.Tile(2176, 3232, 'BI', 13, cache_dir=tmpdir)
    #ret = tile.get_bytes(8388672, 4194304)
    mmstart = tile.dds.mipmap_list[4].startpos
    ret = tile.read_dds_bytes(mmstart, 1024)
    assert ret
   
    tile.maxchunk_wait = 0.05
    mmstart = tile.dds.mipmap_list[0].startpos
    ret = tile.read_dds_bytes(mmstart, 8388608)
    assert ret

    tile.write()
    #assert True == False

def test_get_best_chunk(tmpdir):
    tile = getortho.Tile(17408, 25856, 'BI', 16, cache_dir=tmpdir)
    
    # Verify we get a match
    tile.get_img(2)
    ret = tile.get_best_chunk(17408, 25857, 0, 16)
    assert(ret)
    ret.write_jpg(os.path.join(tmpdir, "chunk.jpg"))

    # Test no matches
    tile2 = getortho.Tile(17408, 26856, 'BI', 16, cache_dir=tmpdir)
    ret = tile2.get_best_chunk(17408, 26857, 0, 16)
    assert not ret

    # image sources can return fake jpeg files, account for this
    tile3 = getortho.Tile(18408, 26856, 'BI', 16, cache_dir=tmpdir)
    shutil.copyfile(
        os.path.join('testfiles', 'test_tile_small.png'),
        os.path.join(tmpdir, '4602_6714_14_BI.jpg')
    )
    ret = tile3.get_best_chunk(18408, 26857, 0, 16)
    assert not ret


@pytest.mark.parametrize("mm", [4,3,2,1])
def test_get_best_chunks_all(mm, tmpdir):
    tile = getortho.Tile(17408, 25856, 'BI', 16, cache_dir=tmpdir)
    
    # Verify we get a match
    tile.get_img(mm)

    for x in range(16):
        for y in range(16):
            ret = tile.get_best_chunk(17408+x, 25856+y, 0, 16)
            assert(ret)
            ret.write_jpg(os.path.join(tmpdir, f"best_{mm}_{x}_{y}.jpg"))

    #assert True == False


# ============================================================================
# Upscaling System Tests
# ============================================================================

class TestCoordinateMapping:
    """Test the mathematical coordinate transformations used in upscaling."""
    
    def test_chunk_coordinate_downscaling(self):
        """Verify that chunk coordinates scale correctly across mipmap levels."""
        col = 12345
        row = 6789
        diff = 2
        
        col_p = col >> diff
        row_p = row >> diff
        
        assert col_p == 3086, f"Expected col_p=3086, got {col_p}"
        assert row_p == 1697, f"Expected row_p=1697, got {row_p}"
    
    def test_offset_calculation(self):
        """Verify that offsets within a lower-detail chunk are calculated correctly."""
        col = 12345
        row = 6789
        scalefactor = 4
        
        col_offset = col % scalefactor
        row_offset = row % scalefactor
        
        assert col_offset == 1, f"Expected col_offset=1, got {col_offset}"
        assert row_offset == 1, f"Expected row_offset=1, got {row_offset}"
    
    def test_coordinate_roundtrip(self):
        """Verify that coordinates can be mapped down and back up consistently."""
        col_hd = 12345
        row_hd = 6789
        diff = 2
        scalefactor = 1 << diff
        
        # Map to low-detail
        col_ld = col_hd >> diff
        row_ld = row_hd >> diff
        
        # Get offset within low-detail chunk
        col_offset = col_hd % scalefactor
        row_offset = row_hd % scalefactor
        
        # Reconstruct high-detail coordinates
        col_reconstructed_base = col_ld << diff
        row_reconstructed_base = row_ld << diff
        col_reconstructed = col_reconstructed_base + col_offset
        row_reconstructed = row_reconstructed_base + row_offset
        
        assert col_reconstructed == col_hd, f"Col roundtrip failed: {col_reconstructed} != {col_hd}"
        assert row_reconstructed == row_hd, f"Row roundtrip failed: {row_reconstructed} != {row_hd}"


class TestUpscalingVisualComparison:
    """Test upscaling quality by comparing upscaled vs native images."""
    
    def test_upscale_2x_vs_native(self, tmpdir):
        """Compare 2× upscaled image (from mipmap 1) to native mipmap 0."""
        # Use separate cache for this test
        cache_dir = os.path.join(tmpdir, "cache_2x")
        os.makedirs(cache_dir, exist_ok=True)
        
        # First, download and save the NATIVE mipmap 0 chunk
        chunk_native = getortho.Chunk(2176, 3232, 'EOX', 13, cache_dir=cache_dir)
        getortho.chunk_getter.submit(chunk_native)
        chunk_native.ready.wait(20)
        
        if chunk_native.data:
            from aoimage import AoImage
            native_img = AoImage.load_from_memory(chunk_native.data)
            native_path = os.path.join(tmpdir, "native_mm0_2176_3232.jpg")
            native_img.write_jpg(native_path, quality=95)
            print(f"Saved native image: {native_path}")
        
        # Now build ONLY mipmap 1 (lower detail)
        tile = getortho.Tile(2176, 3232, 'EOX', 13, cache_dir=cache_dir, max_zoom=13)
        tile.get_img(1, maxwait=20)
        
        # Get upscaled version from mipmap 1 (will ONLY find mm1, not mm0)
        upscaled_img = tile.get_best_chunk(2176, 3232, 0, 13)
        
        if upscaled_img and upscaled_img is not False:
            upscaled_path = os.path.join(tmpdir, "upscaled_2x_from_mm1_2176_3232.jpg")
            upscaled_img.write_jpg(upscaled_path, quality=95)
            print(f"Saved 2× upscaled image: {upscaled_path}")
            
            # Verify dimensions match
            assert upscaled_img.size == (256, 256), "Upscaled image should be 256×256"
            
            print(f"\nCompare images:")
            print(f"  Native:    {native_path}")
            print(f"  Upscaled:  {upscaled_path}")
    
    def test_upscale_4x_vs_native(self, tmpdir):
        """Compare 4× upscaled image (from mipmap 2) to native mipmap 0."""
        # Use separate cache for this test
        cache_dir = os.path.join(tmpdir, "cache_4x")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Download native mipmap 0
        chunk_native = getortho.Chunk(2176, 3232, 'EOX', 13, cache_dir=cache_dir)
        getortho.chunk_getter.submit(chunk_native)
        chunk_native.ready.wait(20)
        
        if chunk_native.data:
            from aoimage import AoImage
            native_img = AoImage.load_from_memory(chunk_native.data)
            native_path = os.path.join(tmpdir, "native_mm0_4x.jpg")
            native_img.write_jpg(native_path, quality=95)
        
        # Build ONLY mipmap 2
        tile = getortho.Tile(2176, 3232, 'EOX', 13, cache_dir=cache_dir, max_zoom=13)
        tile.get_img(2, maxwait=20)
        
        # Get upscaled version (4×)
        upscaled_img = tile.get_best_chunk(2176, 3232, 0, 13)
        
        if upscaled_img and upscaled_img is not False:
            upscaled_path = os.path.join(tmpdir, "upscaled_4x_from_mm2.jpg")
            upscaled_img.write_jpg(upscaled_path, quality=95)
            print(f"\nCompare 4× upscaling:")
            print(f"  Native:    {native_path}")
            print(f"  Upscaled:  {upscaled_path}")
    
    def test_upscale_8x_vs_native(self, tmpdir):
        """Compare 8× upscaled image (from mipmap 3) to native mipmap 0."""
        # Use separate cache for this test
        cache_dir = os.path.join(tmpdir, "cache_8x")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Download native
        chunk_native = getortho.Chunk(2176, 3232, 'EOX', 13, cache_dir=cache_dir)
        getortho.chunk_getter.submit(chunk_native)
        chunk_native.ready.wait(20)
        
        if chunk_native.data:
            from aoimage import AoImage
            native_img = AoImage.load_from_memory(chunk_native.data)
            native_path = os.path.join(tmpdir, "native_mm0_8x.jpg")
            native_img.write_jpg(native_path, quality=95)
        
        # Build ONLY mipmap 3
        tile = getortho.Tile(2176, 3232, 'EOX', 13, cache_dir=cache_dir, max_zoom=13)
        tile.get_img(3, maxwait=20)
        
        # Get upscaled version (8×)
        upscaled_img = tile.get_best_chunk(2176, 3232, 0, 13)
        
        if upscaled_img and upscaled_img is not False:
            upscaled_path = os.path.join(tmpdir, "upscaled_8x_from_mm3.jpg")
            upscaled_img.write_jpg(upscaled_path, quality=95)
            print(f"\nCompare 8× upscaling:")
            print(f"  Native:    {native_path}")
            print(f"  Upscaled:  {upscaled_path}")
    
    def test_upscale_16x_vs_native(self, tmpdir):
        """Compare 16× upscaled image (from mipmap 4) to native mipmap 0."""
        # Use separate cache for this test
        cache_dir = os.path.join(tmpdir, "cache_16x")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Download native
        chunk_native = getortho.Chunk(2176, 3232, 'EOX', 13, cache_dir=cache_dir)
        getortho.chunk_getter.submit(chunk_native)
        chunk_native.ready.wait(20)
        
        if chunk_native.data:
            from aoimage import AoImage
            native_img = AoImage.load_from_memory(chunk_native.data)
            native_path = os.path.join(tmpdir, "native_mm0_16x.jpg")
            native_img.write_jpg(native_path, quality=95)
        
        # Build ONLY mipmap 4 (lowest detail)
        tile = getortho.Tile(2176, 3232, 'EOX', 13, cache_dir=cache_dir, max_zoom=13)
        tile.get_img(4, maxwait=20)
        
        # Get upscaled version (16×)
        upscaled_img = tile.get_best_chunk(2176, 3232, 0, 13)
        
        if upscaled_img and upscaled_img is not False:
            upscaled_path = os.path.join(tmpdir, "upscaled_16x_from_mm4.jpg")
            upscaled_img.write_jpg(upscaled_path, quality=95)
            print(f"\nCompare 16× upscaling:")
            print(f"  Native:    {native_path}")
            print(f"  Upscaled:  {upscaled_path}")
            print(f"\nNote: 16× upscale will be very blocky but better than missing tile!")


class TestFallbackChainIntegration:
    """Test the complete fallback chain with visual output."""
    
    def test_fallback_chain_with_images(self, tmpdir):
        """Test complete fallback chain and save images at each stage."""
        # Create output directory for comparison
        output_dir = os.path.join(tmpdir, "fallback_comparison")
        os.makedirs(output_dir, exist_ok=True)
        
        # Download native mipmap 0 for reference
        print("\n=== Downloading native mipmap 0 for reference ===")
        chunk_native = getortho.Chunk(2176, 3232, 'EOX', 13, cache_dir=tmpdir)
        getortho.chunk_getter.submit(chunk_native)
        chunk_native.ready.wait(20)
        
        if chunk_native.data:
            from aoimage import AoImage
            native_img = AoImage.load_from_memory(chunk_native.data)
            native_path = os.path.join(output_dir, "0_native_mipmap0.jpg")
            native_img.write_jpg(native_path, quality=95)
            print(f"✓ Saved: {native_path}")
        
        # Test each mipmap upscaling SEPARATELY with fresh cache dirs
        # This ensures get_best_chunk() finds ONLY the specific mipmap we're testing
        for mm in [1, 2, 3, 4]:
            print(f"\n=== Testing fallback from mipmap {mm} (scale factor: {2**mm}×) ===")
            
            # Create a separate cache directory for this mipmap test
            mm_cache_dir = os.path.join(tmpdir, f"cache_mm{mm}")
            os.makedirs(mm_cache_dir, exist_ok=True)
            
            # Create new tile with isolated cache
            tile_mm = getortho.Tile(2176, 3232, 'EOX', 13, cache_dir=mm_cache_dir, max_zoom=13)
            
            # Build ONLY this specific mipmap
            tile_mm.get_img(mm, maxwait=20)
            
            # Get upscaled version (will find ONLY the mipmap we just built)
            upscaled_img = tile_mm.get_best_chunk(2176, 3232, 0, 13)
            
            if upscaled_img and upscaled_img is not False:
                upscaled_path = os.path.join(output_dir, f"{mm}_upscaled_{2**mm}x_from_mm{mm}.jpg")
                upscaled_img.write_jpg(upscaled_path, quality=95)
                print(f"✓ Saved: {upscaled_path}")
                print(f"  Cache dir: {mm_cache_dir}")
            else:
                print(f"✗ Failed to get upscaled image from mipmap {mm}")
            
            # Clean up tile
            tile_mm.close()
        
        print(f"\n=== All comparison images saved to: {output_dir} ===")
        print("View these images side-by-side to see upscaling quality at different scales")
        print("\nExpected quality:")
        print("  2× (mm1): Nearly identical to native")
        print("  4× (mm2): Slight blur")
        print("  8× (mm3): Noticeable blockiness")
        print("  16× (mm4): Very blocky but recognizable")
    
    def test_cascading_fallback_with_images(self, tmpdir):
        """Test cascading fallback and save the result."""
        output_dir = os.path.join(tmpdir, "cascade_test")
        os.makedirs(output_dir, exist_ok=True)
        
        tile = getortho.Tile(2176, 3232, 'EOX', 13, cache_dir=tmpdir, max_zoom=13)
        
        # Test cascading fallback (simulates mipmap 0 failure)
        print("\n=== Testing cascading fallback (on-demand download) ===")
        cascaded_img = tile.get_or_build_lower_mipmap_chunk(0, 2176, 3232, 13)
        
        if cascaded_img:
            cascade_path = os.path.join(output_dir, "cascaded_fallback.jpg")
            cascaded_img.write_jpg(cascade_path, quality=95)
            print(f"✓ Cascading fallback succeeded: {cascade_path}")
            
            # Also save the native for comparison
            chunk_native = getortho.Chunk(2176, 3232, 'EOX', 13, cache_dir=tmpdir)
            getortho.chunk_getter.submit(chunk_native)
            if chunk_native.ready.wait(20) and chunk_native.data:
                from aoimage import AoImage
                native_img = AoImage.load_from_memory(chunk_native.data)
                native_path = os.path.join(output_dir, "native_for_comparison.jpg")
                native_img.write_jpg(native_path, quality=95)
                print(f"✓ Native saved: {native_path}")
        else:
            print("✗ Cascading fallback failed")


class TestBuiltMipmapUpscaling:
    """Test upscaling from built mipmaps stored in self.imgs."""
    
    def test_upscale_from_built_mipmap_with_images(self, tmpdir):
        """Test upscaling from built mipmap and save visual comparison."""
        output_dir = os.path.join(tmpdir, "built_mipmap_test")
        os.makedirs(output_dir, exist_ok=True)
        
        tile = getortho.Tile(2176, 3232, 'EOX', 13, cache_dir=tmpdir, max_zoom=13)
        
        print("\n=== Testing upscaling from built mipmaps (self.imgs) ===")
        
        # Build mipmap 2
        img_mm2 = tile.get_img(2, maxwait=20)
        assert img_mm2 is not None, "Failed to build mipmap 2"
        
        # Save the full mipmap 2 for reference
        mm2_path = os.path.join(output_dir, "full_mipmap2.jpg")
        img_mm2.write_jpg(mm2_path, quality=95)
        print(f"✓ Full mipmap 2 saved: {mm2_path}")
        
        # Now upscale a chunk from built mipmap
        chunk_img = tile.get_downscaled_from_higher_mipmap(0, 2176, 3232, 13)
        
        if chunk_img:
            upscaled_path = os.path.join(output_dir, "upscaled_from_built_mm2.jpg")
            chunk_img.write_jpg(upscaled_path, quality=95)
            print(f"✓ Upscaled from built mipmap: {upscaled_path}")
            
            # Compare to native
            chunk_native = getortho.Chunk(2176, 3232, 'EOX', 13, cache_dir=tmpdir)
            getortho.chunk_getter.submit(chunk_native)
            if chunk_native.ready.wait(20) and chunk_native.data:
                from aoimage import AoImage
                native_img = AoImage.load_from_memory(chunk_native.data)
                native_path = os.path.join(output_dir, "native_mm0.jpg")
                native_img.write_jpg(native_path, quality=95)
                print(f"✓ Native saved: {native_path}")
    
    def test_metadata_storage_verification(self, tmpdir):
        """Verify that images are stored with correct metadata."""
        tile = getortho.Tile(2176, 3232, 'EOX', 13, cache_dir=tmpdir, max_zoom=13)
        
        # Build a mipmap
        img = tile.get_img(2, maxwait=20)
        assert img is not None, "Failed to build mipmap"
        
        # Verify metadata storage
        assert 2 in tile.imgs, "Image should be stored in tile.imgs"
        img_data = tile.imgs[2]
        
        # New format: (image, col, row, zoom)
        assert isinstance(img_data, tuple), "Should be stored as tuple"
        assert len(img_data) == 4, f"Should have 4 elements, got {len(img_data)}"
        
        stored_img, stored_col, stored_row, stored_zoom = img_data
        
        print(f"\nMetadata verification:")
        print(f"  Stored col:  {stored_col}")
        print(f"  Stored row:  {stored_row}")
        print(f"  Stored zoom: {stored_zoom}")
        print(f"  Image size:  {stored_img.size}")
    
