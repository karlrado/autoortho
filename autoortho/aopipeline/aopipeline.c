/**
 * aopipeline.c - Unified Native Pipeline Implementation
 * 
 * Combines all pipeline components into a single unified API.
 * 
 * Components:
 * - AoCache: Parallel batch cache file I/O
 * - AoDecode: Parallel JPEG decoding with libturbojpeg
 * - AoDDS: Native DDS texture building
 */

#include "aopipeline.h"
#include "internal.h"
#include <stdio.h>
#include <string.h>

/* Version string */
#define AOPIPELINE_VERSION "1.1.0"

/* Static version buffer */
static char version_buffer[1024] = {0};

/*============================================================================
 * Initialization
 *============================================================================*/

AOPIPELINE_API int32_t aopipeline_init(void) {
    int32_t count = 0;
    
    /* Initialize ISPC for DDS building */
    if (aodds_init_ispc()) {
        count++;
    }
    
    return count + 2;  /* Cache and decode are always available */
}

AOPIPELINE_API void aopipeline_shutdown(void) {
    /* Nothing to shutdown - cache/decode/dds don't have global state */
}

AOPIPELINE_API const char* aopipeline_version(void) {
    if (version_buffer[0] == '\0') {
        snprintf(version_buffer, sizeof(version_buffer),
                 "aopipeline " AOPIPELINE_VERSION "\n"
                 "  %s\n"
                 "  %s\n"
                 "  %s",
                 aocache_version(),
                 aodecode_version(),
                 aodds_version()
        );
    }
    return version_buffer;
}

AOPIPELINE_API void aopipeline_check_components(
    int32_t* out_cache,
    int32_t* out_decode,
    int32_t* out_dds
) {
    if (out_cache) *out_cache = 1;  /* Always available */
    if (out_decode) *out_decode = 1;  /* Always available */
    if (out_dds) *out_dds = 1;  /* Always available (may use fallback compression) */
}

/*============================================================================
 * High-Level Pipeline Functions
 *============================================================================*/

AOPIPELINE_API int32_t aopipeline_build_cached_tile(
    const char* cache_dir,
    int32_t tile_row,
    int32_t tile_col,
    const char* maptype,
    int32_t zoom,
    int32_t chunks_per_side,
    dds_format_t format,
    const uint8_t* missing_color,
    uint8_t* dds_output,
    uint32_t output_size,
    uint32_t* bytes_written,
    aopipeline_stats_t* stats
) {
    if (!cache_dir || !dds_output || !bytes_written) {
        return 0;
    }
    
    /* Initialize stats if provided */
    if (stats) {
        memset(stats, 0, sizeof(aopipeline_stats_t));
    }
    
    /* Build tile request */
    dds_tile_request_t req = {0};
    req.cache_dir = cache_dir;
    req.tile_row = tile_row;
    req.tile_col = tile_col;
    req.maptype = maptype;
    req.zoom = zoom;
    req.chunks_per_side = chunks_per_side;
    req.format = format;
    
    if (missing_color) {
        req.missing_r = missing_color[0];
        req.missing_g = missing_color[1];
        req.missing_b = missing_color[2];
    } else {
        /* Default missing color */
        req.missing_r = 66;
        req.missing_g = 77;
        req.missing_b = 55;
    }
    
    req.dds_buffer = dds_output;
    req.dds_buffer_size = output_size;
    
    /* Build the tile */
    int32_t result = aodds_build_tile(&req, NULL);
    
    *bytes_written = req.dds_written;
    
    /* Copy stats if requested */
    if (stats && result) {
        stats->files_read = req.stats.chunks_found;
        stats->files_failed = req.stats.chunks_failed;
        stats->jpegs_decoded = req.stats.chunks_decoded;
        stats->jpegs_failed = req.stats.chunks_failed;
        stats->chunks_composed = req.stats.chunks_decoded;
        stats->chunks_missing = req.stats.chunks_failed;
        stats->mipmaps_generated = req.stats.mipmaps_generated;
        stats->total_ms = req.stats.elapsed_ms;
    }
    
    return result;
}

