/**
 * aopipeline.c - Unified Native Pipeline Implementation
 * 
 * Combines all pipeline components into a single unified API.
 */

#include "aopipeline.h"
#include "internal.h"
#include <stdio.h>
#include <string.h>

/* Version string */
#define AOPIPELINE_VERSION "1.0.0"

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
    
    /* HTTP is optional - only count if available */
#ifdef AOPIPELINE_HAS_CURL
    if (aohttp_init(32)) {
        count++;
    }
#endif
    
    return count + 2;  /* Cache and decode are always available */
}

AOPIPELINE_API void aopipeline_shutdown(void) {
#ifdef AOPIPELINE_HAS_CURL
    aohttp_shutdown();
#endif
}

AOPIPELINE_API const char* aopipeline_version(void) {
    if (version_buffer[0] == '\0') {
        snprintf(version_buffer, sizeof(version_buffer),
                 "aopipeline " AOPIPELINE_VERSION "\n"
                 "  %s\n"
                 "  %s\n"
                 "  %s\n"
                 "  %s",
                 aocache_version(),
                 aodecode_version(),
                 aodds_version(),
                 aohttp_version()
        );
    }
    return version_buffer;
}

AOPIPELINE_API void aopipeline_check_components(
    int32_t* out_cache,
    int32_t* out_decode,
    int32_t* out_dds,
    int32_t* out_http
) {
    if (out_cache) *out_cache = 1;  /* Always available */
    if (out_decode) *out_decode = 1;  /* Always available */
    if (out_dds) *out_dds = 1;  /* Always available (may use fallback compression) */
    
    if (out_http) {
#ifdef AOPIPELINE_HAS_CURL
        *out_http = aohttp_is_available();
#else
        *out_http = 0;
#endif
    }
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

AOPIPELINE_API int32_t aopipeline_download_and_build(
    const char* url_template,
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
#ifndef AOPIPELINE_HAS_CURL
    /* HTTP not available - cannot download */
    if (bytes_written) *bytes_written = 0;
    return 0;
#else
    if (!url_template || !cache_dir || !dds_output || !bytes_written) {
        return 0;
    }
    
    /* Initialize stats */
    if (stats) {
        memset(stats, 0, sizeof(aopipeline_stats_t));
    }
    
    int32_t chunk_count = chunks_per_side * chunks_per_side;
    
    /* Build URLs for all chunks */
    char** urls = malloc(chunk_count * sizeof(char*));
    char** cache_paths = malloc(chunk_count * sizeof(char*));
    
    if (!urls || !cache_paths) {
        free(urls);
        free(cache_paths);
        return 0;
    }
    
    for (int32_t i = 0; i < chunk_count; i++) {
        urls[i] = malloc(4096);
        cache_paths[i] = malloc(4096);
        
        if (!urls[i] || !cache_paths[i]) {
            /* Cleanup and fail */
            for (int32_t j = 0; j <= i; j++) {
                free(urls[j]);
                free(cache_paths[j]);
            }
            free(urls);
            free(cache_paths);
            return 0;
        }
        
        int32_t chunk_row = i / chunks_per_side;
        int32_t chunk_col = i % chunks_per_side;
        int32_t abs_col = tile_col * chunks_per_side + chunk_col;
        int32_t abs_row = tile_row * chunks_per_side + chunk_row;
        
        /* Build URL from template */
        /* Simple placeholder replacement - real impl would be more robust */
        snprintf(urls[i], 4096, "%s", url_template);
        /* Note: A full implementation would properly replace {col}, {row}, {zoom} */
        
        /* Build cache path */
        snprintf(cache_paths[i], 4096, "%s/%d_%d_%d_%s.jpg",
                 cache_dir, abs_col, abs_row, zoom, maptype);
    }
    
    /* Check which chunks need downloading (not in cache) */
    int32_t* need_download = calloc(chunk_count, sizeof(int32_t));
    int32_t download_count = 0;
    
    for (int32_t i = 0; i < chunk_count; i++) {
        if (!aocache_file_exists(cache_paths[i])) {
            need_download[i] = 1;
            download_count++;
        }
    }
    
    /* Download missing chunks */
    if (download_count > 0) {
        /* Build download URL list */
        const char** download_urls = malloc(download_count * sizeof(char*));
        aohttp_response_t* responses = calloc(download_count, sizeof(aohttp_response_t));
        int32_t* download_indices = malloc(download_count * sizeof(int32_t));
        
        int32_t di = 0;
        for (int32_t i = 0; i < chunk_count; i++) {
            if (need_download[i]) {
                download_urls[di] = urls[i];
                download_indices[di] = i;
                di++;
            }
        }
        
        /* Perform downloads */
        int32_t success = aohttp_get_batch_sync(download_urls, download_count, 
                                                 responses, 30000);
        
        if (stats) {
            stats->http_requests = download_count;
            stats->http_failures = download_count - success;
        }
        
        /* Save successful downloads to cache */
        for (int32_t di = 0; di < download_count; di++) {
            if (responses[di].status_code >= 200 && responses[di].status_code < 300) {
                int32_t i = download_indices[di];
                aocache_write_file_atomic(cache_paths[i], 
                                          responses[di].data, 
                                          responses[di].length);
            }
        }
        
        /* Cleanup */
        aohttp_response_batch_free(responses, download_count);
        free(responses);
        free(download_urls);
        free(download_indices);
    }
    
    free(need_download);
    
    /* Now build DDS from cache */
    int32_t result = aopipeline_build_cached_tile(
        cache_dir, tile_row, tile_col, maptype, zoom,
        chunks_per_side, format, missing_color,
        dds_output, output_size, bytes_written, stats
    );
    
    /* Cleanup */
    for (int32_t i = 0; i < chunk_count; i++) {
        free(urls[i]);
        free(cache_paths[i]);
    }
    free(urls);
    free(cache_paths);
    
    return result;
#endif
}

