/**
 * aodds.h - Native DDS Texture Building for AutoOrtho
 * 
 * Provides complete native DDS generation pipeline:
 * - Cache reading + JPEG decoding + image composition
 * - Mipmap generation with efficient image reduction
 * - DXT1/DXT5 (BC1/BC3) compression via ISPC texcomp
 * - DDS header generation compliant with X-Plane requirements
 * 
 * This module provides the highest performance impact by moving
 * the entire DDS building pipeline to native code.
 */

#ifndef AODDS_H
#define AODDS_H

#include <stdint.h>
#include "aodecode.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Export/Import macros */
#ifdef _WIN32
  #ifdef AOPIPELINE_EXPORTS
    #define AODDS_API __declspec(dllexport)
  #else
    #define AODDS_API __declspec(dllimport)
  #endif
#else
  #define AODDS_API __attribute__((__visibility__("default")))
#endif

/**
 * DDS compression format.
 */
typedef enum {
    DDS_FORMAT_BC1 = 0,  /**< DXT1, 8 bytes per 4x4 block, no alpha */
    DDS_FORMAT_BC3 = 1,  /**< DXT5, 16 bytes per 4x4 block, with alpha */
} dds_format_t;

/**
 * Statistics from DDS building.
 */
typedef struct {
    int32_t chunks_found;       /**< Number of cache hits */
    int32_t chunks_decoded;     /**< Number successfully decoded */
    int32_t chunks_failed;      /**< Number that used fallback color */
    int32_t mipmaps_generated;  /**< Number of mipmap levels created */
    double  elapsed_ms;         /**< Total time in milliseconds */
} dds_stats_t;

/**
 * Tile build request structure.
 * 
 * Contains all parameters needed to build a complete DDS tile
 * from cached JPEG chunks.
 */
typedef struct {
    /* Input parameters */
    const char* cache_dir;      /**< Directory containing cached JPEGs */
    int32_t tile_row;           /**< Tile row coordinate (in tile units) */
    int32_t tile_col;           /**< Tile column coordinate (in tile units) */
    const char* maptype;        /**< Map source identifier (e.g., "BI", "EOX") */
    int32_t zoom;               /**< Zoom level for chunk fetching */
    int32_t chunks_per_side;    /**< Chunks per side (typically 16 for ZL16) */
    dds_format_t format;        /**< Output compression format */
    
    /* Fallback color for missing chunks (RGB) */
    uint8_t missing_r;
    uint8_t missing_g;
    uint8_t missing_b;
    
    /* Output buffer (caller provides) */
    uint8_t* dds_buffer;        /**< Pre-allocated output buffer */
    uint32_t dds_buffer_size;   /**< Size of output buffer in bytes */
    uint32_t dds_written;       /**< Actual bytes written (output) */
    
    /* Statistics (output) */
    dds_stats_t stats;
    
    /* Error handling */
    int32_t success;            /**< 1 = success, 0 = failure */
    char error[256];            /**< Error message if failed */
} dds_tile_request_t;

/**
 * Build a complete DDS tile from cached JPEGs.
 * 
 * This is the main entry point for tile building. It performs the entire
 * pipeline in native code:
 * 1. Batch read all chunk cache files
 * 2. Parallel decode all JPEGs
 * 3. Compose chunks into full tile image
 * 4. Generate all mipmap levels
 * 5. Compress each mipmap with ISPC texcomp
 * 6. Write complete DDS file to output buffer
 * 
 * @param request   Tile build request (input/output)
 * @param pool      Buffer pool for decoding (may be NULL)
 * 
 * @return 1 on success, 0 on failure (check request->error)
 * 
 * Memory: The output buffer must be pre-allocated by caller.
 * Use aodds_calc_dds_size() to determine required size.
 * 
 * Thread Safety: This function is thread-safe and can be called
 * from multiple Python threads simultaneously.
 */
AODDS_API int32_t aodds_build_tile(
    dds_tile_request_t* request,
    aodecode_pool_t* pool
);

/**
 * Calculate required DDS buffer size.
 * 
 * @param width         Image width in pixels
 * @param height        Image height in pixels
 * @param mipmap_count  Number of mipmap levels (0 = auto-calculate)
 * @param format        Compression format
 * 
 * @return Required buffer size in bytes including DDS header
 */
AODDS_API uint32_t aodds_calc_dds_size(
    int32_t width,
    int32_t height,
    int32_t mipmap_count,
    dds_format_t format
);

/**
 * Calculate number of mipmap levels for given dimensions.
 * 
 * @param width     Image width
 * @param height    Image height
 * 
 * @return Number of mipmap levels (minimum 1)
 */
AODDS_API int32_t aodds_calc_mipmap_count(int32_t width, int32_t height);

/**
 * Write DDS header to buffer.
 * 
 * @param buffer        Output buffer
 * @param width         Image width
 * @param height        Image height
 * @param mipmap_count  Number of mipmap levels
 * @param format        Compression format
 * 
 * @return Number of bytes written (128 for standard DDS header)
 */
AODDS_API int32_t aodds_write_header(
    uint8_t* buffer,
    int32_t width,
    int32_t height,
    int32_t mipmap_count,
    dds_format_t format
);

/**
 * Compose multiple chunk images into a single tile image.
 * 
 * @param chunks            Array of decoded chunk images
 * @param chunks_per_side   Number of chunks per side
 * @param output            Output image (caller must allocate data)
 * 
 * @return 1 on success, 0 on failure
 * 
 * Note: Output image data must be pre-allocated to 
 * (chunks_per_side * 256) ^ 2 * 4 bytes
 */
AODDS_API int32_t aodds_compose_chunks(
    aodecode_image_t* chunks,
    int32_t chunks_per_side,
    aodecode_image_t* output
);

/**
 * Fill missing chunk positions with solid color.
 * 
 * @param chunks            Array of chunk images (NULL = missing)
 * @param chunks_per_side   Number of chunks per side
 * @param output            Output image to fill
 * @param r, g, b           Fill color
 */
AODDS_API void aodds_fill_missing(
    aodecode_image_t* chunks,
    int32_t chunks_per_side,
    aodecode_image_t* output,
    uint8_t r, uint8_t g, uint8_t b
);

/**
 * Reduce image by half (generate next mipmap level).
 * 
 * Uses 2x2 box filter averaging for smooth mipmap transitions.
 * 
 * @param input     Source image
 * @param output    Output image (dimensions = input/2)
 * 
 * @return 1 on success, 0 on failure
 * 
 * Note: Output data must be pre-allocated to (width/2) * (height/2) * 4 bytes
 */
AODDS_API int32_t aodds_reduce_half(
    const aodecode_image_t* input,
    aodecode_image_t* output
);

/**
 * Compress RGBA image to BC1/BC3 format.
 * 
 * Uses ISPC texcomp library for high-performance SIMD compression.
 * 
 * @param image     Source RGBA image
 * @param format    Target compression format
 * @param output    Output buffer (must be correctly sized)
 * 
 * @return Number of bytes written, or 0 on failure
 * 
 * Required output size:
 * - BC1: (width/4) * (height/4) * 8 bytes
 * - BC3: (width/4) * (height/4) * 16 bytes
 */
AODDS_API uint32_t aodds_compress(
    const aodecode_image_t* image,
    dds_format_t format,
    uint8_t* output
);

/**
 * Build DDS from pre-decoded images.
 * 
 * Lower-level function for custom pipelines. Assumes chunks are
 * already decoded and in the correct order.
 * 
 * @param chunks            Array of decoded chunk images
 * @param chunks_per_side   Number of chunks per side
 * @param format            Compression format
 * @param missing_color     RGB color for missing chunks
 * @param dds_output        Pre-allocated output buffer
 * @param output_size       Output buffer size
 * @param bytes_written     Actual bytes written (output)
 * 
 * @return 1 on success, 0 on failure
 */
AODDS_API int32_t aodds_build_from_chunks(
    aodecode_image_t* chunks,
    int32_t chunks_per_side,
    dds_format_t format,
    uint8_t missing_color[3],
    uint8_t* dds_output,
    uint32_t output_size,
    uint32_t* bytes_written
);

/**
 * Initialize the ISPC compression library.
 * 
 * This loads the ISPC texcomp dynamic library. Called automatically
 * on first use, but can be called explicitly for early initialization.
 * 
 * @return 1 if ISPC loaded successfully, 0 if failed (fallback to stb_dxt)
 */
AODDS_API int32_t aodds_init_ispc(void);

/**
 * Get version information for the aodds module.
 * 
 * @return Static string with version info (do not free)
 */
AODDS_API const char* aodds_version(void);

#ifdef __cplusplus
}
#endif

#endif /* AODDS_H */

