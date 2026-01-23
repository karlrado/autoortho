#!/usr/bin/env python3
"""
Benchmark: Live Tile Build - Python Progressive vs AoPipeline

Compares on-demand DDS build performance between:
1. Python Progressive Path (pydds.gen_mipmaps) - 13+ ctypes calls per tile
2. AoPipeline Fast Path (build_from_jpegs_to_buffer) - 1 native call with buffer pool

Expected results:
- Python path: ~300-400ms per tile
- AoPipeline: ~50-80ms per tile (5-6x faster)

Usage:
    conda activate ao312
    cd autoortho4xplane
    python autoortho/tests/bench_live_aopipeline.py
"""

import os
import sys
import time
import tempfile
import statistics

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from typing import List, Tuple, Optional

# Set console to UTF-8 on Windows
if sys.platform == 'win32':
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    except Exception:
        pass


def print_header(title: str):
    """Print a formatted header."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(name: str, times: List[float], unit: str = "ms"):
    """Print benchmark result with statistics."""
    if not times:
        print(f"  {name}: No valid results")
        return
    
    avg = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0
    min_t = min(times)
    max_t = max(times)
    
    print(f"  {name}:")
    print(f"    Average: {avg:.1f} {unit}")
    print(f"    Std Dev: {std:.1f} {unit}")
    print(f"    Range:   {min_t:.1f} - {max_t:.1f} {unit}")


def load_test_jpegs(cache_dir: str) -> Tuple[List[bytes], int]:
    """
    Load JPEG files from cache directory for benchmark.
    Returns (list of jpeg bytes, count found).
    """
    jpeg_datas = []
    jpg_files = sorted([f for f in os.listdir(cache_dir) if f.endswith('.jpg')])[:256]
    
    for f in jpg_files:
        with open(os.path.join(cache_dir, f), 'rb') as fp:
            jpeg_datas.append(fp.read())
    
    return jpeg_datas, len(jpg_files)


def find_cache_dir() -> Optional[str]:
    """Find a cache directory with JPEG files."""
    # Try common locations
    candidates = [
        r"D:\Games\X-Plane 12\_autoortho\temp",  # Local test cache
        os.path.expanduser("~/.autoortho-data/cache"),
        os.path.expanduser("~/.autoortho/cache"),
        "./cache",
        "../cache",
    ]
    
    for path in candidates:
        if os.path.isdir(path):
            # Check for JPEGs directly in this directory
            jpg_files = [f for f in os.listdir(path) if f.endswith('.jpg')]
            if len(jpg_files) >= 16:
                return path
            
            # Look for subdirectories with JPEGs
            for root, dirs, files in os.walk(path):
                jpg_count = len([f for f in files if f.endswith('.jpg')])
                if jpg_count >= 16:  # At least 16 JPEGs
                    return root
    
    return None


def benchmark_python_progressive(jpeg_datas: List[bytes], iterations: int = 5) -> List[float]:
    """
    Benchmark Python progressive path (pydds.DDS.gen_mipmaps).
    
    This simulates what happens in the live path:
    1. Decode JPEG to image (AoImage)
    2. Compose 256 chunks into 4096x4096 tile
    3. Generate mipmaps with dds.gen_mipmaps() (13+ compress calls)
    """
    print("\n  Benchmarking Python Progressive Path (pydds)...")
    
    try:
        from autoortho import pydds
        from autoortho.aoimage import AoImage
    except ImportError as e:
        print(f"    ERROR: Cannot import required modules: {e}")
        return []
    
    times = []
    missing_color = (66, 77, 55)
    
    for i in range(iterations):
        start = time.perf_counter()
        
        try:
            # Calculate tile dimensions
            chunks_per_side = int(len(jpeg_datas) ** 0.5)
            if chunks_per_side < 1:
                chunks_per_side = 1
            tile_size = 256 * chunks_per_side
            
            # Create base image using AoImage (matching real path)
            base_img = AoImage.new("RGBA", (tile_size, tile_size), missing_color)
            
            # Compose chunks into base image
            for idx, jpeg_data in enumerate(jpeg_datas[:chunks_per_side * chunks_per_side]):
                try:
                    chunk_img = AoImage.load_from_memory(jpeg_data)
                    if chunk_img:
                        x = (idx % chunks_per_side) * 256
                        y = (idx // chunks_per_side) * 256
                        base_img.paste(chunk_img, x, y)
                        chunk_img.close()
                except Exception:
                    pass
            
            # Generate mipmaps using pydds (the progressive path)
            # This is the DDS class that makes 13+ compress calls
            dds = pydds.DDS(tile_size, tile_size)
            dds.gen_mipmaps(base_img)  # This is the slow call
            
            base_img.close()
            
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            print(f"    Iteration {i+1}: {elapsed:.1f}ms")
            
        except Exception as e:
            print(f"    Iteration {i+1}: ERROR - {e}")
            import traceback
            traceback.print_exc()
    
    return times


def benchmark_aopipeline(jpeg_datas: List[bytes], iterations: int = 5) -> List[float]:
    """
    Benchmark AoPipeline fast path (build_from_jpegs_to_buffer).
    
    This uses:
    - Buffer pool for zero-copy builds
    - Single native call for entire pipeline
    - Parallel decode in C
    """
    print("\n  Benchmarking AoPipeline Fast Path...")
    
    try:
        from autoortho.aopipeline import AoDDS
    except ImportError as e:
        print(f"    ERROR: Cannot import AoDDS: {e}")
        return []
    
    if not hasattr(AoDDS, 'build_from_jpegs_to_buffer'):
        print("    ERROR: build_from_jpegs_to_buffer not available")
        return []
    
    # Create buffer pool
    try:
        pool = AoDDS.DDSBufferPool(pool_size=4)
    except Exception as e:
        print(f"    ERROR: Cannot create buffer pool: {e}")
        return []
    
    times = []
    format_type = "BC1"
    missing_color = (66, 77, 55)
    
    for i in range(iterations):
        start = time.perf_counter()
        
        try:
            # Acquire buffer
            acquired = pool.try_acquire()
            if not acquired:
                print(f"    Iteration {i+1}: ERROR - Pool exhausted")
                continue
            
            buffer, buffer_id = acquired
            
            # Build with native pipeline
            result = AoDDS.build_from_jpegs_to_buffer(
                buffer,
                jpeg_datas,
                format=format_type,
                missing_color=missing_color
            )
            
            # Release buffer
            pool.release(buffer_id)
            
            elapsed = (time.perf_counter() - start) * 1000
            
            if result.success and result.bytes_written > 0:
                times.append(elapsed)
                print(f"    Iteration {i+1}: {elapsed:.1f}ms ({result.bytes_written} bytes)")
            else:
                print(f"    Iteration {i+1}: ERROR - {result.error}")
            
        except Exception as e:
            print(f"    Iteration {i+1}: ERROR - {e}")
            import traceback
            traceback.print_exc()
    
    return times


def benchmark_aopipeline_to_file(jpeg_datas: List[bytes], iterations: int = 5) -> List[float]:
    """
    Benchmark AoPipeline direct-to-file path (build_from_jpegs_to_file).
    
    This writes directly to disk, avoiding Python memory copy.
    """
    print("\n  Benchmarking AoPipeline Direct-to-File...")
    
    try:
        from autoortho.aopipeline import AoDDS
    except ImportError as e:
        print(f"    ERROR: Cannot import AoDDS: {e}")
        return []
    
    if not hasattr(AoDDS, 'build_from_jpegs_to_file'):
        print("    ERROR: build_from_jpegs_to_file not available")
        return []
    
    times = []
    format_type = "BC1"
    missing_color = (66, 77, 55)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(iterations):
            output_path = os.path.join(tmpdir, f"test_{i}.dds")
            start = time.perf_counter()
            
            try:
                result = AoDDS.build_from_jpegs_to_file(
                    jpeg_datas,
                    output_path,
                    format=format_type,
                    missing_color=missing_color
                )
                
                elapsed = (time.perf_counter() - start) * 1000
                
                if result.success and result.bytes_written > 0:
                    times.append(elapsed)
                    print(f"    Iteration {i+1}: {elapsed:.1f}ms ({result.bytes_written} bytes)")
                else:
                    print(f"    Iteration {i+1}: ERROR - {result.error}")
                
                # Cleanup
                if os.path.exists(output_path):
                    os.remove(output_path)
                
            except Exception as e:
                print(f"    Iteration {i+1}: ERROR - {e}")
    
    return times


def generate_synthetic_jpegs(count: int = 256) -> List[bytes]:
    """Generate synthetic JPEG data for benchmark when no cache available."""
    print(f"\n  Generating {count} synthetic JPEG chunks...")
    
    try:
        from PIL import Image
        import io
    except ImportError:
        print("    ERROR: Cannot import PIL")
        return []
    
    jpeg_datas = []
    
    # Create simple solid color images with slight variations
    for i in range(count):
        # Vary colors based on position
        row = i // 16
        col = i % 16
        r = int(50 + row * 10)
        g = int(100 + col * 10)
        b = int(80 + (row + col) * 5)
        
        # Create solid color 256x256 image
        img = Image.new("RGB", (256, 256), (r, g, b))
        
        # Convert to JPEG bytes
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        jpeg_datas.append(buffer.getvalue())
        
        img.close()
        
        if (i + 1) % 64 == 0:
            print(f"    Generated {i+1}/{count} chunks...")
    
    print(f"    Generated {count} chunks")
    return jpeg_datas


def main():
    print_header("Live Tile Build Benchmark: Python vs AoPipeline")
    
    print("\n" + "-" * 70)
    print("  NOTE: This benchmark compares build times for on-demand DDS tiles.")
    print("  ")
    print("  Both paths use native compression (ISPC/STB), so the difference is in:")
    print("  - Chunk collection: Python loops vs single native call")
    print("  - Buffer management: Per-call allocation vs buffer pool")
    print("  - JPEG decoding: Sequential vs parallel (in C)")
    print("  ")
    print("  The live aopipeline integration provides additional benefits not")
    print("  measured here: reduced latency when chunks are already in cache/memory.")
    print("-" * 70)
    
    print("\nConfiguration:")
    print(f"  Python: {sys.executable}")
    print(f"  Platform: {sys.platform}")
    
    # Try to load native module info
    try:
        from autoortho.aopipeline import AoDDS
        version = AoDDS.version() if hasattr(AoDDS, 'version') else "unknown"
        print(f"  AoDDS Version: {version}")
    except Exception as e:
        print(f"  AoDDS: Not available ({e})")
    
    # Find or generate test data
    jpeg_datas = []
    cache_dir = find_cache_dir()
    
    if cache_dir:
        print(f"\n  Found cache directory: {cache_dir}")
        jpeg_datas, count = load_test_jpegs(cache_dir)
        print(f"  Loaded {count} JPEG files")
    
    if len(jpeg_datas) < 16:
        print("\n  Insufficient cache data, generating synthetic JPEGs...")
        # Generate 256 chunks for 4096x4096 tile (same as real tiles)
        jpeg_datas = generate_synthetic_jpegs(256)
    
    if len(jpeg_datas) < 16:
        print("\n  ERROR: Cannot generate test data. Aborting.")
        return 1
    
    # Pad to 256 if needed (for 4096x4096 tile)
    while len(jpeg_datas) < 256:
        jpeg_datas.append(jpeg_datas[0])  # Repeat first chunk
    jpeg_datas = jpeg_datas[:256]
    
    print(f"\n  Using {len(jpeg_datas)} JPEG chunks for benchmark")
    
    iterations = 5
    
    # Run benchmarks
    print_header("Running Benchmarks")
    
    # 1. Python progressive path
    python_times = benchmark_python_progressive(jpeg_datas, iterations)
    
    # 2. AoPipeline buffer pool path
    aopipeline_times = benchmark_aopipeline(jpeg_datas, iterations)
    
    # 3. AoPipeline direct-to-file path
    aopipeline_file_times = benchmark_aopipeline_to_file(jpeg_datas, iterations)
    
    # Results
    print_header("Results Summary")
    
    print_result("Python Progressive (pydds)", python_times)
    print_result("AoPipeline Buffer Pool", aopipeline_times)
    print_result("AoPipeline Direct-to-File", aopipeline_file_times)
    
    # Comparison
    if python_times and aopipeline_times:
        python_avg = statistics.mean(python_times)
        aopipeline_avg = statistics.mean(aopipeline_times)
        speedup = python_avg / aopipeline_avg if aopipeline_avg > 0 else 0
        
        print("\n  Comparison:")
        print(f"    Python avg:     {python_avg:.1f}ms")
        print(f"    AoPipeline avg: {aopipeline_avg:.1f}ms")
        print(f"    Speedup:        {speedup:.1f}x faster")
        
        if speedup >= 5:
            print(f"\n  [OK] Excellent! AoPipeline is {speedup:.1f}x faster than Python path")
        elif speedup >= 2:
            print(f"\n  [OK] Good! AoPipeline is {speedup:.1f}x faster than Python path")
        elif speedup >= 1.1:
            print(f"\n  [OK] AoPipeline is {speedup:.1f}x faster than Python path")
            saved_ms = python_avg - aopipeline_avg
            print(f"       Saving ~{saved_ms:.0f}ms per tile ({python_avg:.0f}ms -> {aopipeline_avg:.0f}ms)")
        else:
            print(f"\n  [!] Unexpected: AoPipeline is only {speedup:.1f}x faster")
    
    if aopipeline_times and aopipeline_file_times:
        buffer_avg = statistics.mean(aopipeline_times)
        file_avg = statistics.mean(aopipeline_file_times)
        diff = buffer_avg - file_avg
        
        print(f"\n  Buffer Pool vs Direct-to-File:")
        print(f"    Buffer Pool:    {buffer_avg:.1f}ms")
        print(f"    Direct-to-File: {file_avg:.1f}ms")
        print(f"    Difference:     {diff:+.1f}ms")
    
    print_header("Benchmark Complete")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

