#!/usr/bin/env python3
"""
bench_pipeline.py - Performance benchmarks for native pipeline

Compares native vs Python implementations for:
- Cache file reading
- JPEG decoding
- DDS building
- Complete tile generation

Usage:
    python -m pytest autoortho/tests/bench_pipeline.py -v -s

Or run directly:
    python autoortho/tests/bench_pipeline.py
"""

import os
import sys
import time
import tempfile
from pathlib import Path
from typing import List, Tuple

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Test Data Generation
# ============================================================================

def create_test_jpeg(width: int = 256, height: int = 256) -> bytes:
    """Create a minimal valid JPEG for testing."""
    # Minimal JFIF JPEG header + scan data for a colored image
    # This is a simplified JPEG that most decoders will accept
    return bytes([
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


def create_test_cache_dir(num_chunks: int = 256) -> Tuple[str, List[str]]:
    """Create a temporary cache directory with test JPEG files."""
    tmpdir = tempfile.mkdtemp(prefix="aopipeline_bench_")
    jpeg_data = create_test_jpeg()
    
    paths = []
    for i in range(num_chunks):
        path = os.path.join(tmpdir, f"{i % 16}_{i // 16}_16_BI.jpg")
        with open(path, 'wb') as f:
            f.write(jpeg_data)
        paths.append(path)
    
    return tmpdir, paths


def cleanup_test_cache(cache_dir: str):
    """Remove test cache directory."""
    import shutil
    try:
        shutil.rmtree(cache_dir)
    except OSError:
        pass


# ============================================================================
# Python Baseline Implementations
# ============================================================================

def python_sequential_read(paths: List[str]) -> List[bytes]:
    """Python baseline: sequential file reads."""
    results = []
    for path in paths:
        try:
            with open(path, 'rb') as f:
                results.append(f.read())
        except (FileNotFoundError, OSError):
            results.append(b'')
    return results


def python_batch_read(paths: List[str]) -> List[bytes]:
    """Python baseline: file reads (still sequential due to GIL)."""
    return python_sequential_read(paths)


# ============================================================================
# Benchmark Functions
# ============================================================================

def benchmark_cache_read(num_files: int = 256, iterations: int = 3):
    """Benchmark cache reading: native vs Python."""
    print(f"\n{'='*70}")
    print(f"CACHE READ BENCHMARK ({num_files} files, {iterations} iterations)")
    print(f"{'='*70}")
    
    # Create test data
    cache_dir, paths = create_test_cache_dir(num_files)
    
    try:
        # Python baseline
        python_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            python_sequential_read(paths)
            python_times.append(time.perf_counter() - start)
        
        python_avg = sum(python_times) / len(python_times) * 1000
        print(f"Python sequential: {python_avg:.2f}ms avg")
        
        # Native (if available)
        try:
            from autoortho.aopipeline import AoCache
            if AoCache.is_available():
                # Warmup
                AoCache.batch_read_cache(paths[:10])
                
                native_times = []
                for _ in range(iterations):
                    start = time.perf_counter()
                    AoCache.batch_read_cache(paths)
                    native_times.append(time.perf_counter() - start)
                
                native_avg = sum(native_times) / len(native_times) * 1000
                speedup = python_avg / native_avg if native_avg > 0 else float('inf')
                
                print(f"Native parallel:   {native_avg:.2f}ms avg")
                print(f"Speedup:           {speedup:.1f}x")
            else:
                print("Native: not available")
        except ImportError as e:
            print(f"Native: not available ({e})")
            
    finally:
        cleanup_test_cache(cache_dir)


def benchmark_dds_build(chunks_per_side: int = 4, iterations: int = 3):
    """Benchmark DDS building: native vs Python."""
    print(f"\n{'='*70}")
    print(f"DDS BUILD BENCHMARK ({chunks_per_side}x{chunks_per_side} chunks, {iterations} iterations)")
    print(f"{'='*70}")
    
    num_chunks = chunks_per_side * chunks_per_side
    cache_dir, paths = create_test_cache_dir(num_chunks)
    
    try:
        # Check if native is available
        try:
            from autoortho.aopipeline import AoDDS
            if AoDDS.is_available():
                # Warmup
                try:
                    AoDDS.build_tile_native(
                        cache_dir=cache_dir,
                        row=0, col=0,
                        maptype="BI", zoom=16,
                        chunks_per_side=chunks_per_side,
                        format="BC1"
                    )
                except Exception:
                    pass
                
                native_times = []
                for _ in range(iterations):
                    start = time.perf_counter()
                    try:
                        result = AoDDS.build_tile_native_detailed(
                            cache_dir=cache_dir,
                            row=0, col=0,
                            maptype="BI", zoom=16,
                            chunks_per_side=chunks_per_side,
                            format="BC1"
                        )
                        native_times.append(time.perf_counter() - start)
                    except Exception as e:
                        print(f"Build failed: {e}")
                
                if native_times:
                    native_avg = sum(native_times) / len(native_times) * 1000
                    print(f"Native DDS build:  {native_avg:.2f}ms avg")
                    
                    if result.success:
                        print(f"  - Chunks decoded: {result.chunks_decoded}")
                        print(f"  - Mipmaps: {result.mipmaps}")
                        print(f"  - Output size: {len(result.data)} bytes")
                else:
                    print("Native: build failed")
            else:
                print("Native DDS: not available")
        except ImportError as e:
            print(f"Native DDS: not available ({e})")
            
    finally:
        cleanup_test_cache(cache_dir)


def benchmark_component_availability():
    """Check which native components are available."""
    print(f"\n{'='*70}")
    print("COMPONENT AVAILABILITY")
    print(f"{'='*70}")
    
    components = [
        ("AoCache", "autoortho.aopipeline.AoCache"),
        ("AoDecode", "autoortho.aopipeline.AoDecode"),
        ("AoDDS", "autoortho.aopipeline.AoDDS"),
        ("AoHttp", "autoortho.aopipeline.AoHttp"),
    ]
    
    for name, module_path in components:
        try:
            parts = module_path.rsplit('.', 1)
            mod = __import__(parts[0], fromlist=[parts[1]])
            component = getattr(mod, parts[1])
            if hasattr(component, 'is_available'):
                available = component.is_available()
            elif hasattr(component, 'get_version'):
                available = True
            else:
                available = True
            
            if available:
                if hasattr(component, 'get_version'):
                    version = component.get_version()
                else:
                    version = "available"
                print(f"  {name}: ✓ {version}")
            else:
                print(f"  {name}: ✗ library not found")
        except ImportError as e:
            print(f"  {name}: ✗ import failed ({e})")
        except Exception as e:
            print(f"  {name}: ✗ error ({e})")


def run_all_benchmarks():
    """Run all benchmarks."""
    print("\n" + "="*70)
    print("AUTOORTHO NATIVE PIPELINE BENCHMARKS")
    print("="*70)
    
    benchmark_component_availability()
    benchmark_cache_read(num_files=256, iterations=3)
    benchmark_dds_build(chunks_per_side=4, iterations=3)
    
    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}\n")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    run_all_benchmarks()

