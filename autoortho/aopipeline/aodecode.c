/**
 * aodecode.c - Native Parallel JPEG Decoding Implementation
 * 
 * Provides high-performance batch JPEG decoding using libturbojpeg
 * with OpenMP parallelism and buffer pooling.
 */

#include "aodecode.h"
#include "aocache.h"
#include "internal.h"
#include <stdio.h>
#include <turbojpeg.h>

/* Version string */
#define AODECODE_VERSION "1.0.0"

/* Standard chunk dimensions */
#define CHUNK_SIZE 256
#define CHUNK_RGBA_BYTES (CHUNK_SIZE * CHUNK_SIZE * 4)

/*============================================================================
 * Buffer Pool Implementation
 *============================================================================*/

/**
 * Buffer pool structure.
 * 
 * Uses a simple stack-based free list for O(1) acquire/release.
 * Thread safety is provided by a mutex around the free list.
 */
struct aodecode_pool {
    uint8_t* memory;        /* Contiguous allocation for all buffers */
    int32_t buffer_size;    /* Size of each buffer (CHUNK_RGBA_BYTES) */
    int32_t count;          /* Total number of buffers */
    int32_t* free_stack;    /* Stack of free buffer indices */
    int32_t free_top;       /* Top of free stack (number of free buffers) */
#ifdef AOPIPELINE_WINDOWS
    CRITICAL_SECTION lock;
#else
    pthread_mutex_t lock;
#endif
};

AODECODE_API aodecode_pool_t* aodecode_create_pool(int32_t count) {
    if (count <= 0) return NULL;
    
    aodecode_pool_t* pool = (aodecode_pool_t*)malloc(sizeof(aodecode_pool_t));
    if (!pool) return NULL;
    
    pool->buffer_size = CHUNK_RGBA_BYTES;
    pool->count = count;
    
    /* Allocate contiguous memory for all buffers */
    pool->memory = (uint8_t*)malloc((size_t)count * CHUNK_RGBA_BYTES);
    if (!pool->memory) {
        free(pool);
        return NULL;
    }
    
    /* Allocate free stack */
    pool->free_stack = (int32_t*)malloc((size_t)count * sizeof(int32_t));
    if (!pool->free_stack) {
        free(pool->memory);
        free(pool);
        return NULL;
    }
    
    /* Initialize free stack with all buffer indices */
    for (int32_t i = 0; i < count; i++) {
        pool->free_stack[i] = i;
    }
    pool->free_top = count;
    
    /* Initialize mutex */
#ifdef AOPIPELINE_WINDOWS
    InitializeCriticalSection(&pool->lock);
#else
    pthread_mutex_init(&pool->lock, NULL);
#endif
    
    return pool;
}

AODECODE_API void aodecode_destroy_pool(aodecode_pool_t* pool) {
    if (!pool) return;
    
#ifdef AOPIPELINE_WINDOWS
    DeleteCriticalSection(&pool->lock);
#else
    pthread_mutex_destroy(&pool->lock);
#endif
    
    free(pool->free_stack);
    free(pool->memory);
    free(pool);
}

AODECODE_API uint8_t* aodecode_acquire_buffer(aodecode_pool_t* pool) {
    if (!pool) {
        /* No pool - fallback to malloc */
        return (uint8_t*)malloc(CHUNK_RGBA_BYTES);
    }
    
    uint8_t* buffer = NULL;
    
#ifdef AOPIPELINE_WINDOWS
    EnterCriticalSection(&pool->lock);
#else
    pthread_mutex_lock(&pool->lock);
#endif
    
    if (pool->free_top > 0) {
        /* Pop from free stack */
        pool->free_top--;
        int32_t idx = pool->free_stack[pool->free_top];
        buffer = pool->memory + (idx * CHUNK_RGBA_BYTES);
    }
    
#ifdef AOPIPELINE_WINDOWS
    LeaveCriticalSection(&pool->lock);
#else
    pthread_mutex_unlock(&pool->lock);
#endif
    
    if (!buffer) {
        /* Pool exhausted - fallback to malloc */
        buffer = (uint8_t*)malloc(CHUNK_RGBA_BYTES);
    }
    
    return buffer;
}

AODECODE_API void aodecode_release_buffer(aodecode_pool_t* pool, uint8_t* buffer) {
    if (!buffer) return;
    
    if (!pool) {
        /* No pool - must have been malloc'd */
        free(buffer);
        return;
    }
    
    /* Check if buffer is from the pool */
    if (buffer >= pool->memory && 
        buffer < pool->memory + ((size_t)pool->count * CHUNK_RGBA_BYTES)) {
        /* Calculate buffer index */
        int32_t idx = (int32_t)((buffer - pool->memory) / CHUNK_RGBA_BYTES);
        
#ifdef AOPIPELINE_WINDOWS
        EnterCriticalSection(&pool->lock);
#else
        pthread_mutex_lock(&pool->lock);
#endif
        
        /* Push back to free stack */
        if (pool->free_top < pool->count) {
            pool->free_stack[pool->free_top] = idx;
            pool->free_top++;
        }
        
#ifdef AOPIPELINE_WINDOWS
        LeaveCriticalSection(&pool->lock);
#else
        pthread_mutex_unlock(&pool->lock);
#endif
    } else {
        /* Not from pool - must have been malloc'd */
        free(buffer);
    }
}

AODECODE_API void aodecode_pool_stats(
    aodecode_pool_t* pool,
    int32_t* out_total,
    int32_t* out_available,
    int32_t* out_acquired
) {
    if (!pool) {
        if (out_total) *out_total = 0;
        if (out_available) *out_available = 0;
        if (out_acquired) *out_acquired = 0;
        return;
    }
    
#ifdef AOPIPELINE_WINDOWS
    EnterCriticalSection(&pool->lock);
#else
    pthread_mutex_lock(&pool->lock);
#endif
    
    if (out_total) *out_total = pool->count;
    if (out_available) *out_available = pool->free_top;
    if (out_acquired) *out_acquired = pool->count - pool->free_top;
    
#ifdef AOPIPELINE_WINDOWS
    LeaveCriticalSection(&pool->lock);
#else
    pthread_mutex_unlock(&pool->lock);
#endif
}

/*============================================================================
 * JPEG Decoding Implementation
 *============================================================================*/

/**
 * Internal: Decode a single JPEG using provided turbojpeg handle.
 * 
 * This function is designed to be called from OpenMP parallel loops
 * where each thread has its own tjhandle.
 */
static int decode_jpeg_internal(
    tjhandle tjh,
    const uint8_t* jpeg_data,
    uint32_t jpeg_length,
    aodecode_image_t* image,
    aodecode_pool_t* pool,
    char* error_buf
) {
    if (!tjh || !jpeg_data || jpeg_length == 0 || !image) {
        if (error_buf) safe_strcpy(error_buf, "Invalid parameters", 64);
        return 0;
    }
    
    /* Get JPEG dimensions */
    int width, height, subsamp, colorspace;
    if (tjDecompressHeader3(tjh, jpeg_data, jpeg_length,
                            &width, &height, &subsamp, &colorspace) < 0) {
        if (error_buf) {
            const char* tj_err = tjGetErrorStr2(tjh);
            safe_strcpy(error_buf, tj_err ? tj_err : "Header parse failed", 64);
        }
        return 0;
    }
    
    /* Validate dimensions */
    if (width <= 0 || height <= 0 || width > 4096 || height > 4096) {
        if (error_buf) safe_strcpy(error_buf, "Invalid image dimensions", 64);
        return 0;
    }
    
    /* Acquire buffer for decoded image */
    uint8_t* buffer;
    int from_pool = 0;
    
    if (pool && width == CHUNK_SIZE && height == CHUNK_SIZE) {
        /* Standard chunk size - try to use pool */
        buffer = aodecode_acquire_buffer(pool);
        /* Check if it's actually from the pool */
        if (pool->memory && buffer >= pool->memory && 
            buffer < pool->memory + ((size_t)pool->count * CHUNK_RGBA_BYTES)) {
            from_pool = 1;
        }
    } else {
        /* Non-standard size - must malloc */
        buffer = (uint8_t*)malloc((size_t)width * height * 4);
    }
    
    if (!buffer) {
        if (error_buf) safe_strcpy(error_buf, "Buffer allocation failed", 64);
        return 0;
    }
    
    /* Decode JPEG to RGBA */
    if (tjDecompress2(tjh, jpeg_data, jpeg_length,
                      buffer, width, 0, height,
                      TJPF_RGBA, TJFLAG_FASTDCT) < 0) {
        if (error_buf) {
            const char* tj_err = tjGetErrorStr2(tjh);
            safe_strcpy(error_buf, tj_err ? tj_err : "Decompress failed", 64);
        }
        if (from_pool) {
            aodecode_release_buffer(pool, buffer);
        } else {
            free(buffer);
        }
        return 0;
    }
    
    /* Fill output structure */
    image->data = buffer;
    image->width = width;
    image->height = height;
    image->stride = width * 4;
    image->channels = 4;
    image->from_pool = from_pool;
    
    return 1;
}

AODECODE_API int32_t aodecode_batch(
    aodecode_request_t* requests,
    int32_t count,
    aodecode_pool_t* pool,
    int32_t max_threads
) {
    if (!requests || count <= 0) return 0;
    
    int32_t success_count = 0;
    
#if AOPIPELINE_HAS_OPENMP
    if (max_threads > 0) {
        omp_set_num_threads(max_threads);
    }
#endif
    
#pragma omp parallel reduction(+:success_count)
    {
        /* Each thread gets its own turbojpeg handle */
        tjhandle tjh = tjInitDecompress();
        if (!tjh) {
            /* Thread can't decode - skip all its work */
        } else {
#pragma omp for schedule(dynamic, 4)
            for (int32_t i = 0; i < count; i++) {
                /* Initialize output */
                requests[i].image.data = NULL;
                requests[i].image.width = 0;
                requests[i].image.height = 0;
                requests[i].image.stride = 0;
                requests[i].image.channels = 0;
                requests[i].image.from_pool = 0;
                requests[i].success = 0;
                requests[i].error[0] = '\0';
                
                if (!requests[i].jpeg_data || requests[i].jpeg_length == 0) {
                    safe_strcpy(requests[i].error, "No JPEG data", 64);
                    continue;
                }
                
                if (decode_jpeg_internal(tjh, 
                                         requests[i].jpeg_data,
                                         requests[i].jpeg_length,
                                         &requests[i].image,
                                         pool,
                                         requests[i].error)) {
                    requests[i].success = 1;
                    success_count++;
                }
            }
            
            tjDestroy(tjh);
        }
    }
    
    return success_count;
}

AODECODE_API int32_t aodecode_from_cache(
    const char** cache_paths,
    int32_t count,
    aodecode_image_t* images,
    aodecode_pool_t* pool,
    int32_t max_threads
) {
    if (!cache_paths || !images || count <= 0) return 0;
    
    /* First, batch read all cache files */
    aocache_result_t* cache_results = (aocache_result_t*)malloc(
        (size_t)count * sizeof(aocache_result_t)
    );
    if (!cache_results) return 0;
    
    aocache_batch_read(cache_paths, count, cache_results, max_threads);
    
    int32_t success_count = 0;
    
#if AOPIPELINE_HAS_OPENMP
    if (max_threads > 0) {
        omp_set_num_threads(max_threads);
    }
#endif
    
#pragma omp parallel reduction(+:success_count)
    {
        tjhandle tjh = tjInitDecompress();
        if (tjh) {
#pragma omp for schedule(dynamic, 4)
            for (int32_t i = 0; i < count; i++) {
                /* Initialize output */
                images[i].data = NULL;
                images[i].width = 0;
                images[i].height = 0;
                images[i].stride = 0;
                images[i].channels = 0;
                images[i].from_pool = 0;
                
                if (!cache_results[i].success) {
                    continue;
                }
                
                char error[64];
                if (decode_jpeg_internal(tjh,
                                         cache_results[i].data,
                                         cache_results[i].length,
                                         &images[i],
                                         pool,
                                         error)) {
                    success_count++;
                }
            }
            
            tjDestroy(tjh);
        }
    }
    
    /* Free cache file data */
    aocache_batch_free(cache_results, count);
    free(cache_results);
    
    return success_count;
}

AODECODE_API int32_t aodecode_single(
    const uint8_t* jpeg_data,
    uint32_t jpeg_length,
    aodecode_image_t* image,
    aodecode_pool_t* pool
) {
    if (!jpeg_data || jpeg_length == 0 || !image) return 0;
    
    /* Initialize output */
    image->data = NULL;
    image->width = 0;
    image->height = 0;
    image->stride = 0;
    image->channels = 0;
    image->from_pool = 0;
    
    tjhandle tjh = tjInitDecompress();
    if (!tjh) return 0;
    
    char error[64];
    int result = decode_jpeg_internal(tjh, jpeg_data, jpeg_length, 
                                       image, pool, error);
    
    tjDestroy(tjh);
    return result;
}

AODECODE_API void aodecode_free_image(
    aodecode_image_t* image,
    aodecode_pool_t* pool
) {
    if (!image || !image->data) return;
    
    if (image->from_pool && pool) {
        aodecode_release_buffer(pool, image->data);
    } else {
        free(image->data);
    }
    
    image->data = NULL;
    image->width = 0;
    image->height = 0;
    image->stride = 0;
    image->channels = 0;
    image->from_pool = 0;
}

AODECODE_API const char* aodecode_version(void) {
    return "aodecode " AODECODE_VERSION
#if AOPIPELINE_HAS_OPENMP
           " (OpenMP enabled)"
#else
           " (OpenMP disabled)"
#endif
#ifdef AOPIPELINE_WINDOWS
           " [Windows]"
#elif defined(AOPIPELINE_MACOS)
           " [macOS]"
#else
           " [Linux]"
#endif
    ;
}

