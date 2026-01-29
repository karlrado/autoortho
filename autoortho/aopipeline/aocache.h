/**
 * aocache.h - Native Parallel Cache I/O for AutoOrtho
 * 
 * Provides high-performance batch file reading with:
 * - OpenMP-based parallelism for concurrent file reads
 * - Memory-mapped I/O for large files
 * - JPEG signature validation
 * - Thread-safe operation
 * 
 * This module bypasses Python's GIL by handling all file I/O
 * in native C code with true parallelism.
 */

#ifndef AOCACHE_H
#define AOCACHE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Export/Import macros */
#ifdef _WIN32
  #ifdef AOPIPELINE_EXPORTS
    #define AOCACHE_API __declspec(dllexport)
  #else
    #define AOCACHE_API __declspec(dllimport)
  #endif
#else
  #define AOCACHE_API __attribute__((__visibility__("default")))
#endif

/**
 * Result structure for batch cache read operations.
 * 
 * Each result contains either:
 * - Valid data (success=1, data points to malloc'd buffer)
 * - Error state (success=0, error contains message)
 */
typedef struct {
    uint8_t* data;      /**< Pointer to file contents (caller must free via aocache_batch_free) */
    uint32_t length;    /**< Length of data in bytes */
    int32_t  success;   /**< 1 = success, 0 = failure */
    char     error[64]; /**< Error message if success=0 */
} aocache_result_t;

/**
 * Read multiple cache files in parallel using OpenMP.
 * 
 * This is the main entry point for batch cache reads. It reads all
 * specified files concurrently, validates JPEG signatures, and
 * returns results in the pre-allocated array.
 * 
 * @param paths         Array of null-terminated file path strings
 * @param count         Number of paths in the array
 * @param results       Pre-allocated array of count results (caller provides)
 * @param max_threads   Maximum parallel threads (0 = auto-detect from CPU cores)
 * 
 * @return Number of successfully read files (0 to count)
 * 
 * Thread Safety: This function is thread-safe and can be called from
 * multiple Python threads simultaneously.
 * 
 * Memory: On success, results[i].data points to malloc'd memory that
 * MUST be freed via aocache_batch_free() after use.
 * 
 * Example:
 *     const char* paths[] = {"cache/a.jpg", "cache/b.jpg", "cache/c.jpg"};
 *     aocache_result_t results[3];
 *     int32_t ok = aocache_batch_read(paths, 3, results, 0);
 *     // Use results...
 *     aocache_batch_free(results, 3);
 */
AOCACHE_API int32_t aocache_batch_read(
    const char** paths,
    int32_t count,
    aocache_result_t* results,
    int32_t max_threads
);

/**
 * Read multiple cache files in parallel, skipping JPEG validation.
 * 
 * Same as aocache_batch_read but does not validate JPEG signatures.
 * Use this when reading non-JPEG files or when validation is not needed.
 * 
 * @param paths         Array of null-terminated file path strings
 * @param count         Number of paths in the array  
 * @param results       Pre-allocated array of count results
 * @param max_threads   Maximum parallel threads (0 = auto)
 * 
 * @return Number of successfully read files
 */
AOCACHE_API int32_t aocache_batch_read_raw(
    const char** paths,
    int32_t count,
    aocache_result_t* results,
    int32_t max_threads
);

/**
 * Free memory allocated by batch read operations.
 * 
 * This function safely frees all data buffers in the results array.
 * It is safe to call on results that have success=0 (no-op for those).
 * 
 * @param results   Array of results from aocache_batch_read
 * @param count     Number of results in the array
 */
AOCACHE_API void aocache_batch_free(
    aocache_result_t* results,
    int32_t count
);

/**
 * Validate JPEG headers in batch without reading full files.
 * 
 * This is a fast check that reads only the first 3 bytes of each file
 * to verify the JPEG signature (FFD8FF). Useful for pre-filtering
 * cache entries before expensive decode operations.
 * 
 * @param paths         Array of null-terminated file path strings
 * @param count         Number of paths in the array
 * @param valid_flags   Output array: 1 = valid JPEG, 0 = invalid/missing
 * @param max_threads   Maximum parallel threads (0 = auto)
 * 
 * @return Number of valid JPEG files found
 */
AOCACHE_API int32_t aocache_validate_jpegs(
    const char** paths,
    int32_t count,
    int32_t* valid_flags,
    int32_t max_threads
);

/**
 * Check if a single cache file exists and is readable.
 * 
 * @param path  Null-terminated file path
 * @return 1 if file exists and is readable, 0 otherwise
 */
AOCACHE_API int32_t aocache_file_exists(const char* path);

/**
 * Get the size of a cache file in bytes.
 * 
 * @param path  Null-terminated file path
 * @return File size in bytes, or -1 if file doesn't exist or error
 */
AOCACHE_API int64_t aocache_file_size(const char* path);

/**
 * Read a single cache file (convenience wrapper).
 * 
 * This is a simpler interface for reading one file when batch
 * operations aren't needed.
 * 
 * @param path      Null-terminated file path
 * @param out_data  Output: pointer to allocated data (caller must free)
 * @param out_len   Output: length of data in bytes
 * 
 * @return 1 on success, 0 on failure
 */
AOCACHE_API int32_t aocache_read_file(
    const char* path,
    uint8_t** out_data,
    uint32_t* out_len
);

/**
 * Write data to a cache file atomically.
 * 
 * Uses atomic write pattern (write to temp, then rename) to prevent
 * partial/corrupt files in case of crash or interruption.
 * 
 * @param path      Null-terminated destination file path
 * @param data      Data to write
 * @param length    Length of data in bytes
 * 
 * @return 1 on success, 0 on failure
 */
AOCACHE_API int32_t aocache_write_file_atomic(
    const char* path,
    const uint8_t* data,
    uint32_t length
);

/**
 * Get version information for the aocache module.
 * 
 * @return Static string with version info (do not free)
 */
AOCACHE_API const char* aocache_version(void);

/**
 * Warm up the thread pool to avoid creation overhead on first batch read.
 * 
 * Call this early in application startup (e.g., during module import).
 * This pre-creates the OpenMP thread pool so subsequent batch reads
 * don't pay the thread creation penalty.
 * 
 * @param num_threads  Number of threads to use (0 = auto-detect from CPU)
 */
AOCACHE_API void aocache_warmup_threads(int32_t num_threads);

#ifdef __cplusplus
}
#endif

#endif /* AOCACHE_H */

