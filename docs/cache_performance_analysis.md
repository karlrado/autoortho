# Cache Reading Performance Analysis

## Executive Summary

This document analyzes the cache reading performance in AutoOrtho and provides recommendations for improvements. The cache system is critical for loading time as it stores JPEG tile chunks that are composed into DDS textures.

**Key Finding**: The current native parallel implementation shows *slower* performance than Python sequential reads in benchmarks (0.4x), which is counterintuitive. This analysis explains why and provides solutions.

---

## Current Architecture

### Cache Reading Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         X-Plane Request                          │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Tile._create_chunks()                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  For each chunk (16-256 per tile):                        │   │
│  │    - Create Chunk object                                   │   │
│  │    - Check cache: Chunk.get_cache()                        │   │
│  │      └─ Path(cache_path).read_bytes()                      │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Current Implementations

| Method | Location | Description |
|--------|----------|-------------|
| `Path.read_bytes()` | `getortho.py:get_cache()` | Python sequential, per-chunk |
| `_batch_read_cache_files()` | `getortho.py` | Native OpenMP parallel batch |
| `aocache_batch_read()` | `aocache.c` | C implementation with OpenMP |
| `aodecode_from_cache()` | `aodecode.c` | Combined read + decode |

---

## Why Native Parallel is Currently Slower

### Benchmark Results (256 files, 11KB each)

| Implementation | Time | Throughput |
|----------------|------|------------|
| Python sequential | 6.7ms | 40.6 MB/s |
| Native parallel | 19.0ms | 14.3 MB/s |
| **Speedup** | **0.35x** | (slower!) |

### Root Causes

#### 1. **OpenMP Thread Overhead** (Dominant Factor)

```c
// Current: Creates thread pool for EACH batch call
#pragma omp parallel for reduction(+:success_count) schedule(dynamic, 4)
for (int32_t i = 0; i < count; i++) {
    // Thread creation/synchronization overhead ~1-5μs per task
    // For 256 files: 256-1280μs just for thread management
}
```

- Thread pool creation: ~1-2ms on macOS
- Dynamic scheduling overhead: ~5-10μs per iteration
- For 256 small files: Thread overhead exceeds I/O time

#### 2. **Small File Penalty**

Files are only 11KB (benchmark) to 20-50KB (real ortho):
- SSD random read: ~50μs latency + 0.1μs/KB transfer
- 11KB file: ~51μs per read
- 256 files sequential: ~13ms I/O
- But filesystem cache: ~1-5μs per cached file

**Key insight**: Files are typically OS-cached, making sequential reads extremely fast.

#### 3. **Memory Allocation Per File**

```c
// Current: malloc per file in aocache.c:read_file_posix()
uint8_t* buffer = (uint8_t*)malloc(size);
// ... read into buffer ...
// Later copied to Python bytes
```

256 mallocs + 256 copies add ~2-5ms overhead.

#### 4. **JPEG Validation Overhead**

```c
// Validates every file
if (!JPEG_SIGNATURE_VALID(data, len)) {
    // Reject non-JPEG
}
```

256 validation checks add ~0.5ms.

#### 5. **ctypes Marshaling**

```python
# Python side: AoCache.py:batch_read_cache()
path_array = (c_char_p * count)()      # Allocate C array
for i, path in enumerate(paths):
    path_array[i] = path.encode('utf-8')  # Encode each path

# ... call C function ...

for i in range(count):
    data = results[i].get_bytes()  # Copy from C to Python
```

For 256 files: ~3-5ms for marshaling overhead.

---

## Recommended Improvements

### Priority 1: Reduce Per-File Overhead (High Impact, Low Risk)

#### 1.1 Use Pre-allocated Buffer Pool for File I/O

```c
// NEW: Add to aocache.c
typedef struct aocache_buffer_pool {
    uint8_t* memory;      // Large contiguous allocation
    size_t buffer_size;   // Max file size (e.g., 256KB)
    int32_t count;        // Number of buffers
    int32_t* free_stack;
    int32_t free_top;
    pthread_mutex_t lock;
} aocache_buffer_pool_t;

// Reuse buffers instead of malloc/free per file
static int read_file_with_pool(const char* path, 
                               aocache_buffer_pool_t* pool,
                               uint8_t** out_data, 
                               uint32_t* out_len) {
    uint8_t* buffer = acquire_buffer(pool);  // O(1) from pool
    // ... read file ...
    *out_data = buffer;
    return 1;
}
```

**Expected gain**: 2-3ms for 256 files

#### 1.2 Avoid Thread Pool Recreation

```c
// NEW: Persistent thread pool
static int pool_initialized = 0;

AOCACHE_API void aocache_init_pool(int32_t num_threads) {
    if (!pool_initialized) {
        omp_set_num_threads(num_threads);
        #pragma omp parallel
        {
            // Warm up thread pool
        }
        pool_initialized = 1;
    }
}

AOCACHE_API int32_t aocache_batch_read(...) {
    // Pool already warm - no creation overhead
    #pragma omp parallel for schedule(static)  // Use static for predictable workload
    ...
}
```

**Expected gain**: 1-2ms for first batch, eliminates thread creation overhead

### Priority 2: Optimize I/O Pattern (Medium Impact, Medium Risk)

#### 2.1 Use `preadv` / Scatter-Gather I/O (Linux/macOS)

```c
// NEW: Read multiple files with single syscall
#include <sys/uio.h>

static int batch_preadv(int* fds, struct iovec* iovecs, 
                        int count, off_t* sizes) {
    // Open all files first
    for (int i = 0; i < count; i++) {
        fds[i] = open(paths[i], O_RDONLY | O_NONBLOCK);
    }
    
    // Read all in parallel using io_submit (Linux) or kqueue (macOS)
    // ...
}
```

#### 2.2 Memory-Mapped I/O for Large Files Only

Current code uses mmap for files > 64KB, but most cache files are < 64KB.

```c
// CHANGE: Lower threshold or remove mmap for cache files
// Cache files are small and frequently accessed - mmap overhead not worth it
#define MMAP_THRESHOLD (1024 * 1024)  // Only for > 1MB files
```

#### 2.3 Use `posix_fadvise` for Sequential Access

```c
// NEW: Hint to kernel about access pattern
int fd = open(path, O_RDONLY);
posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);  // Will read sequentially
posix_fadvise(fd, 0, 0, POSIX_FADV_WILLNEED);    // Prefetch into cache
```

### Priority 3: Reduce Python/C Boundary Overhead (Medium Impact)

#### 3.1 Return Memory Views Instead of Copying

```python
# NEW: Use memoryview to avoid copies
class CacheResult(Structure):
    def get_memoryview(self) -> memoryview:
        """Get data as memoryview (no copy)."""
        if not self.success or not self.data:
            return memoryview(b'')
        # Create memoryview pointing to C memory
        # Requires careful lifetime management
        return (ctypes.c_char * self.length).from_address(
            ctypes.addressof(self.data.contents)
        )
```

#### 3.2 Batch String Encoding

```python
# NEW: Encode all paths at once
def batch_read_cache(paths: List[str], ...) -> ...:
    # Pre-encode all paths
    encoded = [p.encode('utf-8') for p in paths]
    path_array = (c_char_p * count)(*encoded)  # Single allocation
    ...
```

### Priority 4: Architecture Changes (High Impact, High Risk)

#### 4.1 Combined Read-Decode Pipeline

The `aodecode_from_cache()` function already does this but adds overhead. Optimize:

```c
// IMPROVED: Stream directly from disk to decoder
AODECODE_API int32_t aodecode_from_cache_streaming(
    const char** cache_paths,
    int32_t count,
    aodecode_image_t* images,
    aodecode_pool_t* pool,
    int32_t max_threads
) {
    #pragma omp parallel
    {
        tjhandle tjh = tjInitDecompress();
        
        #pragma omp for schedule(dynamic, 8)  // Larger chunk size
        for (int32_t i = 0; i < count; i++) {
            // Read directly into decode buffer
            int fd = open(cache_paths[i], O_RDONLY);
            if (fd < 0) continue;
            
            struct stat st;
            fstat(fd, &st);
            
            // Map file directly (avoid copy)
            void* mapped = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
            close(fd);
            
            if (mapped != MAP_FAILED) {
                // Decode directly from mmap'd region
                tjDecompress2(tjh, mapped, st.st_size, ...);
                munmap(mapped, st.st_size);
            }
        }
        
        tjDestroy(tjh);
    }
}
```

#### 4.2 Async I/O with io_uring (Linux 5.1+)

For Linux systems with modern kernels:

```c
#include <liburing.h>

// NEW: True async I/O without thread overhead
int async_batch_read(const char** paths, int count, ...) {
    struct io_uring ring;
    io_uring_queue_init(256, &ring, 0);
    
    // Submit all reads at once
    for (int i = 0; i < count; i++) {
        struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
        io_uring_prep_read(sqe, fds[i], buffers[i], sizes[i], 0);
        io_uring_sqe_set_data(sqe, &results[i]);
    }
    
    io_uring_submit(&ring);
    
    // Collect completions
    for (int i = 0; i < count; i++) {
        struct io_uring_cqe *cqe;
        io_uring_wait_cqe(&ring, &cqe);
        // Process completion
        io_uring_cqe_seen(&ring, cqe);
    }
}
```

**Expected gain**: 2-3x for I/O-bound workloads, but Linux-only.

### Priority 5: Cache Architecture Improvements

#### 5.1 Cache File Consolidation

Instead of one file per chunk, consolidate into larger files:

```
Current: cache/123_456_16_BI.jpg (11KB)
         cache/124_456_16_BI.jpg (11KB)
         ... 256 files per tile ...

Proposed: cache/tile_123_456_16.dat (2.8MB consolidated)
          Contains: [header][chunk0][chunk1]...[chunk255]
```

Benefits:
- Single `open()` + `read()` instead of 256
- Better sequential read performance
- Reduced filesystem metadata overhead

#### 5.2 Memory-Mapped Cache Index

```c
// NEW: Index file for fast lookups
typedef struct cache_index {
    uint32_t magic;
    uint32_t version;
    uint32_t entry_count;
    struct {
        uint64_t key;       // Hash of chunk ID
        uint32_t offset;    // Offset in data file
        uint32_t size;      // Compressed size
    } entries[];
} cache_index_t;

// Memory-map the index for O(1) lookups
cache_index_t* idx = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
```

---

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 days)

1. **Implement buffer pooling** in aocache.c
2. **Switch to static OpenMP scheduling** for predictable workloads
3. **Pre-warm thread pool** on module load
4. **Optimize ctypes marshaling** in AoCache.py

### Phase 2: I/O Optimization (3-5 days)

1. **Add posix_fadvise hints** for better kernel prefetching
2. **Implement streaming decode** (read directly into decoder)
3. **Benchmark mmap vs read** for different file sizes

### Phase 3: Architecture (1-2 weeks)

1. **Prototype cache consolidation** (files → single archive)
2. **Implement memory-mapped index** for O(1) lookups
3. **Add io_uring support** for Linux (optional)

---

## Benchmarking Recommendations

### Test Scenarios

1. **Cold cache**: Files not in OS cache (most realistic)
2. **Warm cache**: Files in OS cache (current benchmark)
3. **Mixed**: Some cached, some not

### Metrics to Track

- Wall-clock time
- CPU time (user + system)
- I/O wait time
- Memory allocations
- Context switches

### Benchmark Command

```bash
# Clear OS cache before benchmark (requires root)
sudo purge  # macOS
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches  # Linux

# Run benchmark
PYTHONPATH=. python autoortho/tests/bench_pipeline.py
```

---

## Conclusion

The current native parallel implementation is slower than Python sequential due to:

1. **Thread overhead** dominating small-file I/O
2. **Memory allocation** per file
3. **ctypes marshaling** overhead

Recommended approach:
1. **Keep Python sequential for small batches** (< 32 files)
2. **Use native parallel for large batches** (> 64 files) with optimizations
3. **Long-term**: Consolidate cache files to reduce per-file overhead

The most impactful single change would be **buffer pooling + static scheduling**, which should recover most of the lost performance and provide speedups for larger batches.

