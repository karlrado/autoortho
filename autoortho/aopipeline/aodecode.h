/**
 * aodecode.h - Native Parallel JPEG Decoding for AutoOrtho
 * 
 * Provides high-performance batch JPEG decoding with:
 * - OpenMP-based parallelism for concurrent decoding
 * - libturbojpeg for optimized JPEG decompression
 * - Buffer pooling for zero-allocation decode paths
 * - Direct integration with cache reading
 * 
 * This module bypasses Python's GIL by handling all JPEG decoding
 * in native C code with true parallelism.
 */

#ifndef AODECODE_H
#define AODECODE_H

#include <stdint.h>
#include "aocache.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Export/Import macros */
#ifdef _WIN32
  #ifdef AOPIPELINE_EXPORTS
    #define AODECODE_API __declspec(dllexport)
  #else
    #define AODECODE_API __declspec(dllimport)
  #endif
#else
  #define AODECODE_API __attribute__((__visibility__("default")))
#endif

/**
 * Decoded image structure.
 * 
 * Holds the result of JPEG decoding - raw RGBA pixel data.
 * The data pointer is either from a buffer pool or malloc'd.
 */
typedef struct {
    uint8_t* data;      /**< RGBA pixel data (4 bytes per pixel) */
    int32_t width;      /**< Image width in pixels */
    int32_t height;     /**< Image height in pixels */
    int32_t stride;     /**< Bytes per row (usually width * 4) */
    int32_t channels;   /**< Number of channels (always 4 for RGBA) */
    int32_t from_pool;  /**< 1 if data is from buffer pool, 0 if malloc'd */
} aodecode_image_t;

/**
 * Decode request structure for batch operations.
 * 
 * Contains input JPEG data and receives output decoded image.
 */
typedef struct {
    /* Input */
    const uint8_t* jpeg_data;   /**< Pointer to JPEG data in memory */
    uint32_t jpeg_length;       /**< Length of JPEG data in bytes */
    
    /* Output */
    aodecode_image_t image;     /**< Decoded image (populated on success) */
    int32_t success;            /**< 1 = success, 0 = failure */
    char error[64];             /**< Error message if success=0 */
} aodecode_request_t;

/**
 * Buffer pool for zero-allocation decoding.
 * 
 * Pre-allocates a fixed number of RGBA buffers for chunk images.
 * This avoids malloc/free overhead during hot paths.
 */
typedef struct aodecode_pool aodecode_pool_t;

/**
 * Create a buffer pool for chunk decoding.
 * 
 * Each buffer is sized for a 256x256 RGBA image (256KB).
 * 
 * @param count     Number of buffers to pre-allocate
 * @return Pool handle, or NULL on allocation failure
 */
AODECODE_API aodecode_pool_t* aodecode_create_pool(int32_t count);

/**
 * Destroy a buffer pool and free all memory.
 * 
 * @param pool  Pool handle (may be NULL)
 */
AODECODE_API void aodecode_destroy_pool(aodecode_pool_t* pool);

/**
 * Acquire a buffer from the pool.
 * 
 * Returns a pre-allocated 256x256 RGBA buffer. If the pool is empty,
 * falls back to malloc.
 * 
 * @param pool  Pool handle
 * @return Pointer to RGBA buffer (caller must release or free)
 */
AODECODE_API uint8_t* aodecode_acquire_buffer(aodecode_pool_t* pool);

/**
 * Release a buffer back to the pool.
 * 
 * If the buffer was malloc'd (pool exhausted), this frees it.
 * 
 * @param pool      Pool handle
 * @param buffer    Buffer to release
 */
AODECODE_API void aodecode_release_buffer(aodecode_pool_t* pool, uint8_t* buffer);

/**
 * Get pool statistics.
 * 
 * @param pool          Pool handle
 * @param out_total     Output: total buffers in pool
 * @param out_available Output: currently available buffers
 * @param out_acquired  Output: currently acquired buffers
 */
AODECODE_API void aodecode_pool_stats(
    aodecode_pool_t* pool,
    int32_t* out_total,
    int32_t* out_available,
    int32_t* out_acquired
);

/**
 * Decode multiple JPEGs in parallel.
 * 
 * This is the main batch decode function. It decodes all input JPEGs
 * concurrently using OpenMP, with each thread using its own turbojpeg
 * decompressor instance for thread safety.
 * 
 * @param requests      Array of decode requests (input/output)
 * @param count         Number of requests
 * @param pool          Buffer pool for output images (may be NULL for malloc)
 * @param max_threads   Maximum parallel threads (0 = auto)
 * 
 * @return Number of successfully decoded images
 * 
 * Note: Caller is responsible for freeing decoded images either by
 * returning buffers to the pool or calling aodecode_free_image().
 */
AODECODE_API int32_t aodecode_batch(
    aodecode_request_t* requests,
    int32_t count,
    aodecode_pool_t* pool,
    int32_t max_threads
);

/**
 * Combined: read cache files and decode in one native call.
 * 
 * This is the optimal path for loading cached chunks - it reads
 * multiple cache files and decodes them in parallel, all in native
 * code without Python involvement.
 * 
 * @param cache_paths   Array of cache file paths
 * @param count         Number of paths
 * @param images        Output array of decoded images (caller provides)
 * @param pool          Buffer pool (may be NULL)
 * @param max_threads   Maximum parallel threads (0 = auto)
 * 
 * @return Number of successfully decoded images
 * 
 * Note: images[i].data will be NULL for files that failed to read/decode.
 * Check images[i].width > 0 to determine success.
 */
AODECODE_API int32_t aodecode_from_cache(
    const char** cache_paths,
    int32_t count,
    aodecode_image_t* images,
    aodecode_pool_t* pool,
    int32_t max_threads
);

/**
 * Decode a single JPEG from memory.
 * 
 * Convenience function for decoding one JPEG without batch overhead.
 * 
 * @param jpeg_data     Pointer to JPEG data
 * @param jpeg_length   Length of JPEG data
 * @param image         Output image structure
 * @param pool          Buffer pool (may be NULL)
 * 
 * @return 1 on success, 0 on failure
 */
AODECODE_API int32_t aodecode_single(
    const uint8_t* jpeg_data,
    uint32_t jpeg_length,
    aodecode_image_t* image,
    aodecode_pool_t* pool
);

/**
 * Free a decoded image.
 * 
 * If the image buffer was from a pool, returns it to the pool.
 * Otherwise, frees the malloc'd memory.
 * 
 * @param image     Image to free
 * @param pool      Pool the image may have come from (may be NULL)
 */
AODECODE_API void aodecode_free_image(
    aodecode_image_t* image,
    aodecode_pool_t* pool
);

/**
 * Get version information for the aodecode module.
 * 
 * @return Static string with version info (do not free)
 */
AODECODE_API const char* aodecode_version(void);

#ifdef __cplusplus
}
#endif

#endif /* AODECODE_H */

