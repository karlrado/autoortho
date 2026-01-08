#!/usr/bin/env python3
"""
bench_live_streaming.py - Performance benchmark for live DDS building

Compares the "live" mode DDS build performance between:
1. Legacy Python progressive build (pydds)
2. Batch aopipeline build (existing)
3. New streaming aopipeline build with fallbacks

This simulates real-world scenarios where X-Plane requests tiles via FUSE
and we need to build DDS textures quickly.

Usage:
    PYTHONPATH=/path/to/autoortho4xplane python autoortho/tests/bench_live_streaming.py

Or run directly:
    python -m autoortho.tests.bench_live_streaming
"""

import os
import sys
import time
import tempfile
import shutil
from io import BytesIO
from typing import List, Optional, Tuple, Dict
import statistics

# Add parent to path for imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_autoortho_dir = os.path.dirname(_script_dir)
_project_dir = os.path.dirname(_autoortho_dir)
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


# ============================================================================
# Test Data Generation
# ============================================================================

def create_test_jpeg(width: int = 256, height: int = 256) -> bytes:
    """Create a realistic JPEG for testing."""
    test_file = os.path.join(_autoortho_dir, 'testfiles', 'test_tile_small.jpg')
    if os.path.exists(test_file):
        with open(test_file, 'rb') as f:
            return f.read()
    
    # Fallback to PIL
    try:
        from PIL import Image
        import random
        random.seed(42)
        
        img = Image.new('RGB', (width, height))
        pixels = img.load()
        for y in range(height):
            for x in range(width):
                r = int((x / width) * 128 + 64) + random.randint(-20, 20)
                g = int((y / height) * 100 + 100) + random.randint(-20, 20)
                b = int(((x + y) / (width + height)) * 100 + 50) + random.randint(-20, 20)
                pixels[x, y] = (
                    min(255, max(0, r)),
                    min(255, max(0, g)),
                    min(255, max(0, b))
                )
        
        buf = BytesIO()
        img.save(buf, format='JPEG', quality=85)
        return buf.getvalue()
    except ImportError:
        pass
    
    # Minimal valid JPEG
    return bytes([
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
        0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xD9
    ])


def create_test_cache(cache_dir: str, chunks_per_side: int = 16,
                      missing_ratio: float = 0.0) -> List[str]:
    """
    Create a test cache directory with JPEG chunks.
    
    Args:
        cache_dir: Directory to create cache in
        chunks_per_side: Number of chunks per side (16 = 256 chunks total)
        missing_ratio: Fraction of chunks to leave missing (0.0-1.0)
    
    Returns:
        List of created file paths
    """
    import random
    random.seed(42)
    
    os.makedirs(cache_dir, exist_ok=True)
    jpeg_data = create_test_jpeg()
    
    paths = []
    total_chunks = chunks_per_side * chunks_per_side
    missing_count = int(total_chunks * missing_ratio)
    missing_indices = set(random.sample(range(total_chunks), missing_count))
    
    for row in range(chunks_per_side):
        for col in range(chunks_per_side):
            idx = row * chunks_per_side + col
            path = os.path.join(cache_dir, f"{col}_{row}_16_BI.jpg")
            
            if idx not in missing_indices:
                with open(path, 'wb') as f:
                    f.write(jpeg_data)
                paths.append(path)
    
    return paths


# ============================================================================
# Build Methods
# ============================================================================

def build_python_progressive(cache_dir: str, chunks_per_side: int = 16) -> Tuple[Optional[bytes], float]:
    """
    Build DDS using Python progressive approach (legacy pydds).
    
    This simulates the old path where we build mipmap by mipmap using Python.
    """
    start = time.perf_counter()
    
    try:
        from autoortho.aoimage import AoImage as aoimage_module
        from autoortho.pydds import DDS
    except ImportError as e:
        return None, 0.0
    
    chunk_size = 256
    tile_size = chunks_per_side * chunk_size
    
    # Read and decode all chunks
    images = []
    for row in range(chunks_per_side):
        for col in range(chunks_per_side):
            path = os.path.join(cache_dir, f"{col}_{row}_16_BI.jpg")
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    jpeg_data = f.read()
                try:
                    img = aoimage_module.load_from_memory(jpeg_data)
                    images.append(img)
                except Exception:
                    images.append(None)
            else:
                images.append(None)
    
    # Compose tile
    tile_img = aoimage_module.new('RGBA', (tile_size, tile_size), (66, 77, 55, 255))
    for i, img in enumerate(images):
        if img is not None:
            img_row = i // chunks_per_side
            img_col = i % chunks_per_side
            x = img_col * chunk_size
            y = img_row * chunk_size
            try:
                tile_img.paste(img, (x, y))
            except Exception:
                pass
    
    # Build DDS with mipmaps
    dds = DDS(tile_size, tile_size, ispc=True, dxt_format="BC1")
    dds.gen_mipmaps(tile_img, startmipmap=0, maxmipmaps=8)
    
    # Collect output
    output = BytesIO()
    output.write(dds.header.getvalue())
    for mipmap in dds.mipmap_list:
        if mipmap.databuffer is not None:
            output.write(mipmap.databuffer.getvalue())
    
    elapsed = time.perf_counter() - start
    return output.getvalue(), elapsed


# Global flag for batch aopipeline availability
_batch_available: Optional[bool] = None


def is_batch_aopipeline_available() -> bool:
    """Check if batch aopipeline is available (cached)."""
    global _batch_available
    if _batch_available is None:
        _batch_available = _check_batch_aopipeline()
    return _batch_available


def _check_batch_aopipeline() -> bool:
    """Actually check if batch aopipeline works."""
    try:
        from autoortho.aopipeline import AoDDS
        if not hasattr(AoDDS, 'is_available'):
            return False
        return AoDDS.is_available()
    except Exception:
        return False


def build_batch_aopipeline(cache_dir: str, chunks_per_side: int = 16) -> Tuple[Optional[bytes], float]:
    """
    Build DDS using batch aopipeline (existing native approach).
    
    This is the hybrid approach: Python reads files, native decodes+compresses.
    """
    # Fast path: check availability
    if not is_batch_aopipeline_available():
        return None, 0.0
    
    start = time.perf_counter()
    
    try:
        from autoortho.aopipeline import AoDDS
    except ImportError:
        return None, 0.0
    
    # Collect JPEG data
    jpeg_datas = []
    for row in range(chunks_per_side):
        for col in range(chunks_per_side):
            path = os.path.join(cache_dir, f"{col}_{row}_16_BI.jpg")
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    jpeg_datas.append(f.read())
            else:
                jpeg_datas.append(None)
    
    # Build using buffer pool
    pool = AoDDS.get_default_pool()
    try:
        buffer, buffer_id = pool.acquire(timeout=5.0)
    except Exception:
        return None, 0.0
    
    try:
        result = AoDDS.build_from_jpegs_to_buffer(
            buffer,
            jpeg_datas,
            format="BC1",
            missing_color=(66, 77, 55)
        )
        
        if result.success:
            dds_bytes = bytes(buffer[:result.bytes_written])
            elapsed = time.perf_counter() - start
            return dds_bytes, elapsed
        else:
            return None, 0.0
    finally:
        pool.release(buffer_id)


# Global flag for streaming builder availability
_streaming_available: Optional[bool] = None


def is_streaming_builder_available() -> bool:
    """Check if streaming builder is available (cached)."""
    global _streaming_available
    if _streaming_available is None:
        _streaming_available = _check_streaming_builder()
    return _streaming_available


def _check_streaming_builder() -> bool:
    """Actually check if streaming builder works."""
    try:
        from autoortho.aopipeline.AoDDS import get_default_builder_pool
        pool = get_default_builder_pool()
        if pool is None:
            return False
        config = {'chunks_per_side': 4, 'format': 'BC1', 'missing_color': (0, 0, 0)}
        builder = pool.acquire(config=config, timeout=0.5)
        if builder is None:
            return False
        builder.release()
        return True
    except Exception:
        return False


def build_streaming_aopipeline(cache_dir: str, chunks_per_side: int = 16) -> Tuple[Optional[bytes], float]:
    """
    Build DDS using new streaming aopipeline with fallback support.
    
    This is the new approach that processes chunks incrementally.
    """
    # Fast path: check availability
    if not is_streaming_builder_available():
        return None, 0.0
    
    start = time.perf_counter()
    
    try:
        from autoortho.aopipeline.AoDDS import (
            get_default_builder_pool, get_default_pool
        )
    except ImportError:
        return None, 0.0
    
    builder_pool = get_default_builder_pool()
    if builder_pool is None:
        return None, 0.0
    
    config = {
        'chunks_per_side': chunks_per_side,
        'format': 'BC1',
        'missing_color': (66, 77, 55)
    }
    
    try:
        builder = builder_pool.acquire(config=config, timeout=5.0)
    except Exception:
        return None, 0.0
    
    if builder is None:
        return None, 0.0
    
    try:
        # Process chunks incrementally
        for row in range(chunks_per_side):
            for col in range(chunks_per_side):
                idx = row * chunks_per_side + col
                path = os.path.join(cache_dir, f"{col}_{row}_16_BI.jpg")
                
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        jpeg_data = f.read()
                    
                    if not builder.add_chunk(idx, jpeg_data):
                        # Decode failed - mark missing (use solid color)
                        builder.mark_missing(idx)
                else:
                    # Missing chunk - mark missing (use solid color)
                    builder.mark_missing(idx)
        
        # Finalize to buffer
        dds_pool = get_default_pool()
        try:
            buffer, buffer_id = dds_pool.acquire(timeout=5.0)
        except Exception:
            return None, 0.0
        
        try:
            result = builder.finalize(buffer)
            if result.success:
                dds_bytes = bytes(buffer[:result.bytes_written])
                elapsed = time.perf_counter() - start
                return dds_bytes, elapsed
            else:
                return None, 0.0
        finally:
            dds_pool.release(buffer_id)
    
    finally:
        builder.release()


# ============================================================================
# Benchmark Functions
# ============================================================================

def run_benchmark(
    name: str,
    build_func,
    cache_dir: str,
    chunks_per_side: int = 16,
    iterations: int = 5,
    warmup: int = 1
) -> Dict:
    """Run a benchmark for a build method."""
    
    # Warmup
    for _ in range(warmup):
        build_func(cache_dir, chunks_per_side)
    
    # Timed runs
    times = []
    sizes = []
    successes = 0
    
    for i in range(iterations):
        result, elapsed = build_func(cache_dir, chunks_per_side)
        times.append(elapsed * 1000)  # Convert to ms
        if result:
            sizes.append(len(result))
            successes += 1
        else:
            sizes.append(0)
    
    return {
        'name': name,
        'times': times,
        'mean_ms': statistics.mean(times) if times else 0,
        'std_ms': statistics.stdev(times) if len(times) > 1 else 0,
        'min_ms': min(times) if times else 0,
        'max_ms': max(times) if times else 0,
        'median_ms': statistics.median(times) if times else 0,
        'success_rate': successes / iterations if iterations > 0 else 0,
        'output_size': sizes[0] if sizes and sizes[0] > 0 else 0,
    }


def print_results(results: List[Dict], baseline_name: str = None,
                  expected_size_kb: Optional[float] = None):
    """Print benchmark results in a formatted table."""
    
    # Find baseline for speedup calculation
    baseline = None
    if baseline_name:
        for r in results:
            if r['name'] == baseline_name:
                baseline = r
                break
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"{'Method':<35} {'Mean':>10} {'Std':>8} {'Min':>8} {'Max':>8} {'Speedup':>8}")
    print("-" * 80)
    
    for r in results:
        speedup = ""
        if baseline and r['mean_ms'] > 0 and baseline['mean_ms'] > 0:
            speedup = f"{baseline['mean_ms'] / r['mean_ms']:.2f}x"
        
        print(f"{r['name']:<35} {r['mean_ms']:>8.1f}ms {r['std_ms']:>6.1f}ms "
              f"{r['min_ms']:>6.1f}ms {r['max_ms']:>6.1f}ms {speedup:>8}")
    
    print("-" * 80)
    
    # Print output sizes with validation
    print(f"\nOutput sizes: ", end="")
    size_mismatch = False
    for r in results:
        if r['output_size'] > 0:
            size_kb = r['output_size'] / 1024
            mark = ""
            if expected_size_kb and abs(size_kb - expected_size_kb) > 10:
                mark = " ⚠️"
                size_mismatch = True
            print(f"{r['name']}: {size_kb:.1f}KB{mark}  ", end="")
    print()
    
    if size_mismatch and expected_size_kb:
        print(f"\n⚠️  WARNING: Some outputs don't match expected size ({expected_size_kb:.1f}KB)")
        print("   This may indicate a bug in that builder's configuration handling.")


def benchmark_all_complete_chunks(iterations: int = 5):
    """Benchmark with all chunks available (best case)."""
    
    print("\n" + "=" * 80)
    print("BENCHMARK: 100% Chunks Available (Best Case)")
    print("=" * 80)
    print(f"Tile: 16x16 chunks = 256 chunks, 4096x4096 pixels")
    print(f"Iterations: {iterations}")
    
    # Create test cache
    cache_dir = tempfile.mkdtemp(prefix="ao_bench_complete_")
    print(f"Cache dir: {cache_dir}")
    
    try:
        paths = create_test_cache(cache_dir, chunks_per_side=16, missing_ratio=0.0)
        print(f"Created {len(paths)} test chunks")
        
        results = []
        
        # Python progressive
        print("\n  Running Python progressive build...")
        result = run_benchmark(
            "Python Progressive (pydds)",
            build_python_progressive,
            cache_dir,
            chunks_per_side=16,
            iterations=iterations
        )
        if result['success_rate'] > 0:
            results.append(result)
        else:
            print("    FAILED - skipping")
        
        # Batch aopipeline
        print("  Running Batch aopipeline build...")
        result = run_benchmark(
            "Batch aopipeline (hybrid)",
            build_batch_aopipeline,
            cache_dir,
            chunks_per_side=16,
            iterations=iterations
        )
        if result['success_rate'] > 0:
            results.append(result)
        else:
            print("    FAILED - skipping")
        
        # Streaming aopipeline
        print("  Running Streaming aopipeline...")
        result = run_benchmark(
            "Streaming aopipeline (native)",
            build_streaming_aopipeline,
            cache_dir,
            chunks_per_side=16,
            iterations=iterations
        )
        if result['success_rate'] > 0:
            results.append(result)
        else:
            print("    FAILED - skipping")
        
        # 16x16 = 256 chunks, 4096x4096 pixels, BC1 ~10923KB expected
        print_results(results, baseline_name="Python Progressive (pydds)",
                      expected_size_kb=10923.0)
        
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def benchmark_with_missing_chunks(missing_ratio: float = 0.1, iterations: int = 5):
    """Benchmark with some chunks missing (fallback test)."""
    
    missing_pct = int(missing_ratio * 100)
    print("\n" + "=" * 80)
    print(f"BENCHMARK: {100 - missing_pct}% Chunks Available ({missing_pct}% Missing)")
    print("=" * 80)
    print(f"Tile: 16x16 chunks = 256 chunks, ~{int(256 * missing_ratio)} missing")
    print(f"Iterations: {iterations}")
    
    # Create test cache with missing chunks
    cache_dir = tempfile.mkdtemp(prefix="ao_bench_missing_")
    print(f"Cache dir: {cache_dir}")
    
    try:
        paths = create_test_cache(cache_dir, chunks_per_side=16, missing_ratio=missing_ratio)
        print(f"Created {len(paths)} test chunks (expected {int(256 * (1-missing_ratio))})")
        
        results = []
        
        # Python progressive
        print("\n  Running Python progressive build...")
        result = run_benchmark(
            "Python Progressive (pydds)",
            build_python_progressive,
            cache_dir,
            chunks_per_side=16,
            iterations=iterations
        )
        if result['success_rate'] > 0:
            results.append(result)
        else:
            print("    FAILED - skipping")
        
        # Batch aopipeline
        print("  Running Batch aopipeline build...")
        result = run_benchmark(
            "Batch aopipeline (hybrid)",
            build_batch_aopipeline,
            cache_dir,
            chunks_per_side=16,
            iterations=iterations
        )
        if result['success_rate'] > 0:
            results.append(result)
        else:
            print("    FAILED - skipping")
        
        # Streaming aopipeline
        print("  Running Streaming aopipeline...")
        result = run_benchmark(
            "Streaming aopipeline (native)",
            build_streaming_aopipeline,
            cache_dir,
            chunks_per_side=16,
            iterations=iterations
        )
        if result['success_rate'] > 0:
            results.append(result)
        else:
            print("    FAILED - skipping")
        
        # 16x16 = 256 chunks, 4096x4096 pixels, BC1 ~10923KB expected
        print_results(results, baseline_name="Python Progressive (pydds)",
                      expected_size_kb=10923.0)
        
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def benchmark_small_tile(iterations: int = 10):
    """Benchmark with smaller tile (faster, good for quick testing)."""
    
    print("\n" + "=" * 80)
    print("BENCHMARK: Small Tile (4x4 = 16 chunks)")
    print("=" * 80)
    print(f"Tile: 4x4 chunks = 16 chunks, 1024x1024 pixels")
    print(f"Iterations: {iterations}")
    
    # Create test cache
    cache_dir = tempfile.mkdtemp(prefix="ao_bench_small_")
    print(f"Cache dir: {cache_dir}")
    
    try:
        paths = create_test_cache(cache_dir, chunks_per_side=4, missing_ratio=0.0)
        print(f"Created {len(paths)} test chunks")
        
        results = []
        
        # Python progressive
        print("\n  Running Python progressive build...")
        result = run_benchmark(
            "Python Progressive (pydds)",
            build_python_progressive,
            cache_dir,
            chunks_per_side=4,
            iterations=iterations
        )
        if result['success_rate'] > 0:
            results.append(result)
        else:
            print("    FAILED - skipping")
        
        # Batch aopipeline
        print("  Running Batch aopipeline build...")
        result = run_benchmark(
            "Batch aopipeline (hybrid)",
            build_batch_aopipeline,
            cache_dir,
            chunks_per_side=4,
            iterations=iterations
        )
        if result['success_rate'] > 0:
            results.append(result)
        else:
            print("    FAILED - skipping")
        
        # Streaming aopipeline
        print("  Running Streaming aopipeline...")
        result = run_benchmark(
            "Streaming aopipeline (native)",
            build_streaming_aopipeline,
            cache_dir,
            chunks_per_side=4,
            iterations=iterations
        )
        if result['success_rate'] > 0:
            results.append(result)
        else:
            print("    FAILED - skipping")
        
        # 4x4 = 16 chunks, 1024x1024 pixels, BC1 ~683KB expected
        print_results(results, baseline_name="Python Progressive (pydds)",
                      expected_size_kb=683.0)
        
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def main():
    """Run all benchmarks."""
    print("=" * 80)
    print("LIVE MODE DDS BUILD BENCHMARK")
    print("=" * 80)
    print("""
This benchmark compares different approaches for building DDS textures
in "live" mode (when X-Plane requests tiles via FUSE):

1. Python Progressive (pydds)
   - Legacy approach: Python reads, decodes, composes, compresses
   - Slowest but most compatible

2. Batch aopipeline (hybrid)
   - Existing optimized approach: Python reads files, native decode+compress
   - Fast when all chunks are available

3. Streaming aopipeline (new)
   - New approach: Incremental chunk processing with fallback support
   - Can apply fallbacks for missing chunks
   - Best for mixed scenarios with some missing chunks
""")
    
    # Check native library availability
    batch_available = is_batch_aopipeline_available()
    streaming_available = is_streaming_builder_available()
    
    print("Native Library Status:")
    print(f"  - Batch aopipeline:     {'✓ Available' if batch_available else '✗ Not available'}")
    print(f"  - Streaming aopipeline: {'✓ Available' if streaming_available else '✗ Not available'}")
    print()
    
    if not batch_available:
        print("NOTE: Native aopipeline library not available.")
        print("      The library may need to be rebuilt.")
        print("      Run one of:")
        print("        make -C autoortho/aopipeline -f Makefile.macos   (macOS)")
        print("        make -C autoortho/aopipeline -f Makefile.linux   (Linux)")
        print("        make -C autoortho/aopipeline -f Makefile.mingw64 (Windows)")
        print()
    elif not streaming_available:
        print("NOTE: Streaming aopipeline not available (batch works).")
        print("      The library needs to be rebuilt with streaming support.")
        print("      Run one of the make commands above to rebuild.")
        print()
        print("      The streaming builder enables:")
        print("        - Incremental chunk processing (no waiting for all chunks)")
        print("        - Integration with fallback resolvers for missing chunks")
        print("        - Better resource management via builder pooling")
        print()
    
    # Quick test first
    print("\n" + "#" * 80)
    print("# QUICK TEST (Small Tile)")
    print("#" * 80)
    benchmark_small_tile(iterations=5)
    
    # Full benchmarks
    print("\n" + "#" * 80)
    print("# FULL BENCHMARK (16x16 Tile)")
    print("#" * 80)
    benchmark_all_complete_chunks(iterations=5)
    
    # With missing chunks
    print("\n" + "#" * 80)
    print("# MISSING CHUNKS BENCHMARK")
    print("#" * 80)
    benchmark_with_missing_chunks(missing_ratio=0.1, iterations=5)
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()

