"""
Performance benchmarks for AOB2 bundle format.

Compares:
- Individual JPEG file reads vs bundle reads
- Bundle creation performance
- DDS building from bundles vs individual files

Run with: python -m autoortho.tests.bench_bundle2
"""

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


# Sample JPEG data - realistic size (~30KB per chunk)
def generate_sample_jpeg(size_bytes: int = 30000) -> bytes:
    """Generate sample JPEG-like data."""
    header = bytes([
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
        0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
    ])
    # Add random-ish data to simulate compression
    import random
    random.seed(42)
    data = bytes(random.randint(0, 255) for _ in range(size_bytes - len(header)))
    return header + data


SAMPLE_JPEG = generate_sample_jpeg()


def setup_individual_jpegs(cache_dir: str, count: int = 256) -> List[str]:
    """Create individual JPEG files for benchmarking."""
    paths = []
    chunks_per_side = int(count ** 0.5)
    
    for i in range(count):
        row = i // chunks_per_side
        col = i % chunks_per_side
        path = os.path.join(cache_dir, f"{col}_{row}_16_BI.jpg")
        with open(path, 'wb') as f:
            f.write(SAMPLE_JPEG)
        paths.append(path)
    
    return paths


def benchmark_individual_reads(paths: List[str], iterations: int = 10) -> dict:
    """Benchmark reading individual JPEG files."""
    times = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        
        data = []
        for path in paths:
            with open(path, 'rb') as f:
                data.append(f.read())
        
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return {
        'method': 'Individual JPEGs',
        'file_count': len(paths),
        'iterations': iterations,
        'avg_time_ms': sum(times) / len(times) * 1000,
        'min_time_ms': min(times) * 1000,
        'max_time_ms': max(times) * 1000,
        'total_bytes': sum(len(open(p, 'rb').read()) for p in paths[:10]) * len(paths) // 10,
    }


def benchmark_bundle_read(bundle_path: str, zoom: int, iterations: int = 10) -> dict:
    """Benchmark reading from bundle."""
    from autoortho.aopipeline.AoBundle2 import Bundle2Python
    
    times = []
    chunk_count = 0
    
    for _ in range(iterations):
        start = time.perf_counter()
        
        bundle = Bundle2Python(bundle_path)
        data = bundle.get_all_chunks(zoom)
        chunk_count = len(data)
        
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return {
        'method': 'Bundle (Python)',
        'file_count': 1,
        'chunk_count': chunk_count,
        'iterations': iterations,
        'avg_time_ms': sum(times) / len(times) * 1000,
        'min_time_ms': min(times) * 1000,
        'max_time_ms': max(times) * 1000,
    }


def benchmark_bundle_creation(cache_dir: str, output_path: str, 
                               chunk_count: int = 256) -> dict:
    """Benchmark bundle creation from individual files."""
    from autoortho.aopipeline.AoBundle2 import create_bundle_python
    
    chunks_per_side = int(chunk_count ** 0.5)
    
    start = time.perf_counter()
    
    create_bundle_python(
        cache_dir=cache_dir,
        tile_row=0,
        tile_col=0,
        maptype="BI",
        zoom=16,
        chunks_per_side=chunks_per_side,
        output_path=output_path
    )
    
    elapsed = time.perf_counter() - start
    
    bundle_size = os.path.getsize(output_path)
    jpeg_sizes = sum(os.path.getsize(f) for f in Path(cache_dir).glob("*.jpg"))
    
    return {
        'method': 'Bundle Creation',
        'chunk_count': chunk_count,
        'time_ms': elapsed * 1000,
        'bundle_size_kb': bundle_size / 1024,
        'jpeg_total_size_kb': jpeg_sizes / 1024,
        'overhead_pct': (bundle_size - jpeg_sizes) / jpeg_sizes * 100,
    }


def run_benchmarks():
    """Run all benchmarks."""
    print("=" * 70)
    print("AOB2 Bundle Performance Benchmarks")
    print("=" * 70)
    print()
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_dir = os.path.join(tmp_dir, "cache")
        os.makedirs(cache_dir)
        
        # Setup test data
        print("Setting up test data...")
        chunk_count = 256  # 16x16 tile
        paths = setup_individual_jpegs(cache_dir, chunk_count)
        print(f"  Created {len(paths)} JPEG files ({len(SAMPLE_JPEG) / 1024:.1f}KB each)")
        print()
        
        # Benchmark 1: Individual file reads
        print("Benchmark 1: Individual JPEG file reads")
        print("-" * 50)
        result = benchmark_individual_reads(paths)
        print(f"  Files: {result['file_count']}")
        print(f"  Iterations: {result['iterations']}")
        print(f"  Avg time: {result['avg_time_ms']:.2f} ms")
        print(f"  Min time: {result['min_time_ms']:.2f} ms")
        print(f"  Max time: {result['max_time_ms']:.2f} ms")
        individual_avg = result['avg_time_ms']
        print()
        
        # Benchmark 2: Bundle creation
        print("Benchmark 2: Bundle creation")
        print("-" * 50)
        bundle_path = os.path.join(tmp_dir, "test.aob2")
        result = benchmark_bundle_creation(cache_dir, bundle_path, chunk_count)
        print(f"  Chunks: {result['chunk_count']}")
        print(f"  Time: {result['time_ms']:.2f} ms")
        print(f"  Bundle size: {result['bundle_size_kb']:.1f} KB")
        print(f"  JPEG total: {result['jpeg_total_size_kb']:.1f} KB")
        print(f"  Overhead: {result['overhead_pct']:.2f}%")
        print()
        
        # Benchmark 3: Bundle reads
        print("Benchmark 3: Bundle reads (Python)")
        print("-" * 50)
        result = benchmark_bundle_read(bundle_path, 16)
        print(f"  Chunks: {result['chunk_count']}")
        print(f"  Iterations: {result['iterations']}")
        print(f"  Avg time: {result['avg_time_ms']:.2f} ms")
        print(f"  Min time: {result['min_time_ms']:.2f} ms")
        print(f"  Max time: {result['max_time_ms']:.2f} ms")
        bundle_avg = result['avg_time_ms']
        print()
        
        # Summary
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        speedup = individual_avg / bundle_avg if bundle_avg > 0 else 0
        print(f"  Individual reads: {individual_avg:.2f} ms")
        print(f"  Bundle reads:     {bundle_avg:.2f} ms")
        print(f"  Speedup:          {speedup:.1f}x")
        print()
        
        if speedup > 1:
            print(f"  Bundle reads are {speedup:.1f}x FASTER than individual file reads")
        else:
            print(f"  Bundle reads are {1/speedup:.1f}x SLOWER than individual file reads")
        print()
        
        # Additional benchmarks if native library available
        try:
            from autoortho.aopipeline import AoBundle2
            if AoBundle2.is_available():
                print("Native library available - running native benchmarks...")
                # Native benchmarks would go here
        except ImportError:
            print("(Native library not available - skipping native benchmarks)")


def run_scaling_benchmark():
    """Benchmark how performance scales with chunk count."""
    print("=" * 70)
    print("Scaling Benchmark: Performance vs Chunk Count")
    print("=" * 70)
    print()
    
    chunk_counts = [16, 64, 256, 1024]  # 4x4, 8x8, 16x16, 32x32
    
    results = []
    
    for count in chunk_counts:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = os.path.join(tmp_dir, "cache")
            os.makedirs(cache_dir)
            
            paths = setup_individual_jpegs(cache_dir, count)
            
            # Individual reads
            ind_result = benchmark_individual_reads(paths, iterations=5)
            
            # Bundle creation
            bundle_path = os.path.join(tmp_dir, "test.aob2")
            create_result = benchmark_bundle_creation(cache_dir, bundle_path, count)
            
            # Bundle reads
            bnd_result = benchmark_bundle_read(bundle_path, 16, iterations=5)
            
            speedup = ind_result['avg_time_ms'] / bnd_result['avg_time_ms']
            
            results.append({
                'chunks': count,
                'individual_ms': ind_result['avg_time_ms'],
                'bundle_ms': bnd_result['avg_time_ms'],
                'speedup': speedup,
                'create_ms': create_result['time_ms'],
            })
    
    # Print table
    print(f"{'Chunks':>8} {'Individual':>12} {'Bundle':>12} {'Speedup':>10} {'Create':>12}")
    print("-" * 60)
    for r in results:
        print(f"{r['chunks']:>8} {r['individual_ms']:>10.2f}ms {r['bundle_ms']:>10.2f}ms "
              f"{r['speedup']:>9.1f}x {r['create_ms']:>10.2f}ms")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AOB2 Bundle Benchmarks")
    parser.add_argument("--scaling", action="store_true", help="Run scaling benchmark")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    args = parser.parse_args()
    
    if args.scaling or args.all:
        run_scaling_benchmark()
        print()
    
    if not args.scaling or args.all:
        run_benchmarks()
