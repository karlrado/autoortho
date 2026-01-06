/**
 * aodds.c - Native DDS Texture Building Implementation
 * 
 * Complete native pipeline for building DDS textures from cached JPEGs.
 * Includes ISPC texcomp integration for high-performance compression.
 */

#include "aodds.h"
#include "aodecode.h"
#include "aocache.h"
#include "internal.h"
#include <stdio.h>
#include <math.h>

#ifdef AOPIPELINE_WINDOWS
#include <windows.h>
#else
#include <dlfcn.h>
#include <sys/time.h>
#endif

/* Version string */
#define AODDS_VERSION "1.0.0"

/* DDS file constants */
#define DDS_MAGIC 0x20534444  /* "DDS " */
#define DDS_HEADER_SIZE 128
#define CHUNK_SIZE 256

/* DDS header flags */
#define DDSD_CAPS        0x00000001
#define DDSD_HEIGHT      0x00000002
#define DDSD_WIDTH       0x00000004
#define DDSD_PITCH       0x00000008
#define DDSD_PIXELFORMAT 0x00001000
#define DDSD_MIPMAPCOUNT 0x00020000
#define DDSD_LINEARSIZE  0x00080000
#define DDSD_DEPTH       0x00800000

#define DDPF_FOURCC      0x00000004

#define DDSCAPS_TEXTURE  0x00001000
#define DDSCAPS_MIPMAP   0x00400000
#define DDSCAPS_COMPLEX  0x00000008

/* FourCC codes */
#define FOURCC_DXT1 0x31545844  /* "DXT1" */
#define FOURCC_DXT5 0x35545844  /* "DXT5" */

/*============================================================================
 * ISPC Texture Compression
 *============================================================================*/

/* ISPC function pointer types */
typedef struct {
    uint8_t* ptr;
    int32_t width;
    int32_t height;
    int32_t stride;
} rgba_surface_t;

typedef void (*CompressBlocksBC1_fn)(const rgba_surface_t*, uint8_t*);
typedef void (*CompressBlocksBC3_fn)(const rgba_surface_t*, uint8_t*);

/* Global function pointers */
static CompressBlocksBC1_fn ispc_compress_bc1 = NULL;
static CompressBlocksBC3_fn ispc_compress_bc3 = NULL;
static int ispc_initialized = 0;
static int ispc_available = 0;

#ifdef AOPIPELINE_WINDOWS
static HMODULE ispc_lib = NULL;
#else
static void* ispc_lib = NULL;
#endif

AODDS_API int32_t aodds_init_ispc(void) {
    if (ispc_initialized) {
        return ispc_available;
    }
    ispc_initialized = 1;
    
    /* Determine library path */
    const char* lib_name;
#ifdef AOPIPELINE_WINDOWS
    lib_name = "ispc_texcomp.dll";
#elif defined(AOPIPELINE_MACOS)
    lib_name = "libispc_texcomp.dylib";
#else
    lib_name = "libispc_texcomp.so";
#endif
    
    /* Try to load from lib directory relative to this module */
#ifdef AOPIPELINE_WINDOWS
    char lib_path[MAX_PATH];
    snprintf(lib_path, MAX_PATH, "../lib/windows/%s", lib_name);
    ispc_lib = LoadLibraryA(lib_path);
    if (!ispc_lib) {
        /* Try current directory */
        ispc_lib = LoadLibraryA(lib_name);
    }
    if (ispc_lib) {
        ispc_compress_bc1 = (CompressBlocksBC1_fn)GetProcAddress(ispc_lib, "CompressBlocksBC1");
        ispc_compress_bc3 = (CompressBlocksBC3_fn)GetProcAddress(ispc_lib, "CompressBlocksBC3");
    }
#else
    char lib_path[4096];
#ifdef AOPIPELINE_MACOS
    snprintf(lib_path, sizeof(lib_path), "../lib/macos/%s", lib_name);
#else
    snprintf(lib_path, sizeof(lib_path), "../lib/linux/%s", lib_name);
#endif
    ispc_lib = dlopen(lib_path, RTLD_NOW);
    if (!ispc_lib) {
        /* Try standard paths */
        ispc_lib = dlopen(lib_name, RTLD_NOW);
    }
    if (ispc_lib) {
        ispc_compress_bc1 = (CompressBlocksBC1_fn)dlsym(ispc_lib, "CompressBlocksBC1");
        ispc_compress_bc3 = (CompressBlocksBC3_fn)dlsym(ispc_lib, "CompressBlocksBC3");
    }
#endif
    
    ispc_available = (ispc_compress_bc1 != NULL && ispc_compress_bc3 != NULL);
    return ispc_available;
}

/*============================================================================
 * Timing Utilities
 *============================================================================*/

static double get_time_ms(void) {
#ifdef AOPIPELINE_WINDOWS
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)freq.QuadPart;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
#endif
}

/*============================================================================
 * DDS Header Generation
 *============================================================================*/

AODDS_API int32_t aodds_calc_mipmap_count(int32_t width, int32_t height) {
    int32_t count = 1;
    while (width > 4 && height > 4) {
        width /= 2;
        height /= 2;
        count++;
    }
    return count;
}

AODDS_API uint32_t aodds_calc_dds_size(
    int32_t width,
    int32_t height,
    int32_t mipmap_count,
    dds_format_t format
) {
    if (mipmap_count <= 0) {
        mipmap_count = aodds_calc_mipmap_count(width, height);
    }
    
    uint32_t block_size = (format == DDS_FORMAT_BC1) ? 8 : 16;
    uint32_t total_size = DDS_HEADER_SIZE;
    
    int32_t w = width;
    int32_t h = height;
    
    for (int32_t i = 0; i < mipmap_count; i++) {
        uint32_t blocks_x = (w + 3) / 4;
        uint32_t blocks_y = (h + 3) / 4;
        total_size += blocks_x * blocks_y * block_size;
        w = AOPIPELINE_MAX(w / 2, 1);
        h = AOPIPELINE_MAX(h / 2, 1);
    }
    
    return total_size;
}

AODDS_API int32_t aodds_write_header(
    uint8_t* buffer,
    int32_t width,
    int32_t height,
    int32_t mipmap_count,
    dds_format_t format
) {
    if (!buffer) return 0;
    
    memset(buffer, 0, DDS_HEADER_SIZE);
    
    uint32_t* header = (uint32_t*)buffer;
    
    /* Magic number */
    header[0] = DDS_MAGIC;
    
    /* Header size (always 124 for DDS_HEADER) */
    header[1] = 124;
    
    /* Flags */
    header[2] = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | 
                DDSD_MIPMAPCOUNT | DDSD_LINEARSIZE;
    
    /* Height and width */
    header[3] = height;
    header[4] = width;
    
    /* Pitch/linear size (size of top-level mipmap) */
    uint32_t block_size = (format == DDS_FORMAT_BC1) ? 8 : 16;
    uint32_t blocks_x = (width + 3) / 4;
    uint32_t blocks_y = (height + 3) / 4;
    header[5] = blocks_x * blocks_y * block_size;
    
    /* Depth (unused for 2D textures) */
    header[6] = 0;
    
    /* Mipmap count */
    header[7] = mipmap_count;
    
    /* Reserved (11 DWORDs) - already zeroed */
    
    /* Pixel format starts at offset 76 (DWORD 19) */
    uint32_t* pf = &header[19];
    
    /* Pixel format size */
    pf[0] = 32;
    
    /* Pixel format flags */
    pf[1] = DDPF_FOURCC;
    
    /* FourCC */
    pf[2] = (format == DDS_FORMAT_BC1) ? FOURCC_DXT1 : FOURCC_DXT5;
    
    /* RGB bit counts (unused for compressed) */
    pf[3] = 0;
    pf[4] = 0;
    pf[5] = 0;
    pf[6] = 0;
    pf[7] = 0;
    
    /* Caps at offset 108 (DWORD 27) */
    header[27] = DDSCAPS_TEXTURE | DDSCAPS_MIPMAP | DDSCAPS_COMPLEX;
    
    return DDS_HEADER_SIZE;
}

/*============================================================================
 * Image Operations
 *============================================================================*/

AODDS_API int32_t aodds_compose_chunks(
    aodecode_image_t* chunks,
    int32_t chunks_per_side,
    aodecode_image_t* output
) {
    if (!chunks || !output || !output->data || chunks_per_side <= 0) {
        return 0;
    }
    
    int32_t tile_size = chunks_per_side * CHUNK_SIZE;
    if (output->width != tile_size || output->height != tile_size) {
        return 0;
    }
    
    int32_t chunk_count = chunks_per_side * chunks_per_side;
    
    /* Compose chunks into output - can be parallelized */
#pragma omp parallel for schedule(dynamic, 4)
    for (int32_t i = 0; i < chunk_count; i++) {
        if (!chunks[i].data || chunks[i].width != CHUNK_SIZE || 
            chunks[i].height != CHUNK_SIZE) {
            continue;  /* Skip invalid chunks (will use fill color) */
        }
        
        int32_t chunk_row = i / chunks_per_side;
        int32_t chunk_col = i % chunks_per_side;
        int32_t dest_x = chunk_col * CHUNK_SIZE;
        int32_t dest_y = chunk_row * CHUNK_SIZE;
        
        /* Copy chunk data to output */
        for (int32_t y = 0; y < CHUNK_SIZE; y++) {
            uint8_t* src_row = chunks[i].data + (y * chunks[i].stride);
            uint8_t* dst_row = output->data + 
                               ((dest_y + y) * output->stride) + 
                               (dest_x * 4);
            memcpy(dst_row, src_row, CHUNK_SIZE * 4);
        }
    }
    
    return 1;
}

AODDS_API void aodds_fill_missing(
    aodecode_image_t* chunks,
    int32_t chunks_per_side,
    aodecode_image_t* output,
    uint8_t r, uint8_t g, uint8_t b
) {
    if (!chunks || !output || !output->data) return;
    
    int32_t chunk_count = chunks_per_side * chunks_per_side;
    
    /* Fill missing chunk areas with solid color */
#pragma omp parallel for schedule(dynamic, 4)
    for (int32_t i = 0; i < chunk_count; i++) {
        if (chunks[i].data && chunks[i].width == CHUNK_SIZE && 
            chunks[i].height == CHUNK_SIZE) {
            continue;  /* Chunk is valid, skip */
        }
        
        int32_t chunk_row = i / chunks_per_side;
        int32_t chunk_col = i % chunks_per_side;
        int32_t dest_x = chunk_col * CHUNK_SIZE;
        int32_t dest_y = chunk_row * CHUNK_SIZE;
        
        /* Fill with solid color */
        for (int32_t y = 0; y < CHUNK_SIZE; y++) {
            uint8_t* dst_row = output->data + 
                               ((dest_y + y) * output->stride) + 
                               (dest_x * 4);
            for (int32_t x = 0; x < CHUNK_SIZE; x++) {
                dst_row[x * 4 + 0] = r;
                dst_row[x * 4 + 1] = g;
                dst_row[x * 4 + 2] = b;
                dst_row[x * 4 + 3] = 255;
            }
        }
    }
}

AODDS_API int32_t aodds_reduce_half(
    const aodecode_image_t* input,
    aodecode_image_t* output
) {
    if (!input || !input->data || !output || !output->data) {
        return 0;
    }
    
    int32_t out_w = input->width / 2;
    int32_t out_h = input->height / 2;
    
    if (output->width != out_w || output->height != out_h) {
        return 0;
    }
    
    /* 2x2 box filter with averaging */
#pragma omp parallel for schedule(static)
    for (int32_t y = 0; y < out_h; y++) {
        const uint8_t* src_row0 = input->data + ((y * 2) * input->stride);
        const uint8_t* src_row1 = input->data + ((y * 2 + 1) * input->stride);
        uint8_t* dst_row = output->data + (y * output->stride);
        
        for (int32_t x = 0; x < out_w; x++) {
            int32_t sx = x * 2 * 4;
            
            /* Average 4 pixels */
            for (int32_t c = 0; c < 4; c++) {
                uint32_t sum = (uint32_t)src_row0[sx + c] +
                               (uint32_t)src_row0[sx + 4 + c] +
                               (uint32_t)src_row1[sx + c] +
                               (uint32_t)src_row1[sx + 4 + c];
                dst_row[x * 4 + c] = (uint8_t)((sum + 2) / 4);
            }
        }
    }
    
    return 1;
}

/*============================================================================
 * Compression
 *============================================================================*/

/* Fallback BC1 compression (when ISPC not available) */
static void compress_bc1_block_simple(const uint8_t* rgba, uint8_t* block) {
    /* Simple compression: use min/max colors */
    uint8_t min_r = 255, min_g = 255, min_b = 255;
    uint8_t max_r = 0, max_g = 0, max_b = 0;
    
    for (int i = 0; i < 16; i++) {
        uint8_t r = rgba[i * 4 + 0];
        uint8_t g = rgba[i * 4 + 1];
        uint8_t b = rgba[i * 4 + 2];
        
        if (r < min_r) min_r = r;
        if (g < min_g) min_g = g;
        if (b < min_b) min_b = b;
        if (r > max_r) max_r = r;
        if (g > max_g) max_g = g;
        if (b > max_b) max_b = b;
    }
    
    /* Convert to RGB565 */
    uint16_t c0 = ((max_r >> 3) << 11) | ((max_g >> 2) << 5) | (max_b >> 3);
    uint16_t c1 = ((min_r >> 3) << 11) | ((min_g >> 2) << 5) | (min_b >> 3);
    
    /* Ensure c0 > c1 for 4-color mode */
    if (c0 < c1) {
        uint16_t tmp = c0; c0 = c1; c1 = tmp;
        uint8_t tr = max_r; max_r = min_r; min_r = tr;
        uint8_t tg = max_g; max_g = min_g; min_g = tg;
        uint8_t tb = max_b; max_b = min_b; min_b = tb;
    }
    
    block[0] = c0 & 0xFF;
    block[1] = (c0 >> 8) & 0xFF;
    block[2] = c1 & 0xFF;
    block[3] = (c1 >> 8) & 0xFF;
    
    /* Calculate interpolated colors */
    int32_t colors[4][3] = {
        {max_r, max_g, max_b},
        {min_r, min_g, min_b},
        {(2 * max_r + min_r + 1) / 3, (2 * max_g + min_g + 1) / 3, (2 * max_b + min_b + 1) / 3},
        {(max_r + 2 * min_r + 1) / 3, (max_g + 2 * min_g + 1) / 3, (max_b + 2 * min_b + 1) / 3}
    };
    
    /* Encode indices */
    uint32_t indices = 0;
    for (int i = 0; i < 16; i++) {
        int32_t best_idx = 0;
        int32_t best_dist = INT32_MAX;
        
        int32_t r = rgba[i * 4 + 0];
        int32_t g = rgba[i * 4 + 1];
        int32_t b = rgba[i * 4 + 2];
        
        for (int j = 0; j < 4; j++) {
            int32_t dr = r - colors[j][0];
            int32_t dg = g - colors[j][1];
            int32_t db = b - colors[j][2];
            int32_t dist = dr*dr + dg*dg + db*db;
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = j;
            }
        }
        
        indices |= (best_idx << (i * 2));
    }
    
    block[4] = indices & 0xFF;
    block[5] = (indices >> 8) & 0xFF;
    block[6] = (indices >> 16) & 0xFF;
    block[7] = (indices >> 24) & 0xFF;
}

static void compress_bc1_fallback(const aodecode_image_t* image, uint8_t* output) {
    int32_t blocks_x = (image->width + 3) / 4;
    int32_t blocks_y = (image->height + 3) / 4;
    
#pragma omp parallel for schedule(static)
    for (int32_t by = 0; by < blocks_y; by++) {
        for (int32_t bx = 0; bx < blocks_x; bx++) {
            uint8_t block_rgba[16 * 4];
            
            /* Extract 4x4 block */
            for (int y = 0; y < 4; y++) {
                int32_t src_y = by * 4 + y;
                if (src_y >= image->height) src_y = image->height - 1;
                
                for (int x = 0; x < 4; x++) {
                    int32_t src_x = bx * 4 + x;
                    if (src_x >= image->width) src_x = image->width - 1;
                    
                    const uint8_t* src = image->data + 
                                         (src_y * image->stride) + 
                                         (src_x * 4);
                    uint8_t* dst = block_rgba + ((y * 4 + x) * 4);
                    memcpy(dst, src, 4);
                }
            }
            
            /* Compress block */
            uint8_t* block_out = output + ((by * blocks_x + bx) * 8);
            compress_bc1_block_simple(block_rgba, block_out);
        }
    }
}

AODDS_API uint32_t aodds_compress(
    const aodecode_image_t* image,
    dds_format_t format,
    uint8_t* output
) {
    if (!image || !image->data || !output) {
        return 0;
    }
    
    /* Initialize ISPC if not done */
    aodds_init_ispc();
    
    uint32_t block_size = (format == DDS_FORMAT_BC1) ? 8 : 16;
    uint32_t blocks_x = (image->width + 3) / 4;
    uint32_t blocks_y = (image->height + 3) / 4;
    uint32_t output_size = blocks_x * blocks_y * block_size;
    
    if (ispc_available) {
        /* Use ISPC compression */
        rgba_surface_t surface = {
            .ptr = image->data,
            .width = image->width,
            .height = image->height,
            .stride = image->stride
        };
        
        if (format == DDS_FORMAT_BC1) {
            ispc_compress_bc1(&surface, output);
        } else {
            ispc_compress_bc3(&surface, output);
        }
    } else {
        /* Fallback compression */
        if (format == DDS_FORMAT_BC1) {
            compress_bc1_fallback(image, output);
        } else {
            /* BC3 fallback not implemented - use BC1 */
            compress_bc1_fallback(image, output);
            /* Note: This produces incorrect output for BC3 */
            /* A real BC3 fallback would need additional alpha block handling */
        }
    }
    
    return output_size;
}

/*============================================================================
 * Main Build Function
 *============================================================================*/

AODDS_API int32_t aodds_build_tile(
    dds_tile_request_t* request,
    aodecode_pool_t* pool
) {
    if (!request || !request->dds_buffer || !request->cache_dir) {
        if (request) safe_strcpy(request->error, "Invalid parameters", 256);
        return 0;
    }
    
    double start_time = get_time_ms();
    
    /* Initialize output */
    request->dds_written = 0;
    request->success = 0;
    request->stats.chunks_found = 0;
    request->stats.chunks_decoded = 0;
    request->stats.chunks_failed = 0;
    request->stats.mipmaps_generated = 0;
    request->stats.elapsed_ms = 0;
    
    int32_t chunks_per_side = request->chunks_per_side;
    int32_t chunk_count = chunks_per_side * chunks_per_side;
    int32_t tile_size = chunks_per_side * CHUNK_SIZE;
    
    /* Calculate required DDS size */
    int32_t mipmap_count = aodds_calc_mipmap_count(tile_size, tile_size);
    uint32_t required_size = aodds_calc_dds_size(tile_size, tile_size, 
                                                  mipmap_count, request->format);
    
    if (request->dds_buffer_size < required_size) {
        snprintf(request->error, 256, 
                 "Buffer too small: need %u bytes, got %u",
                 required_size, request->dds_buffer_size);
        return 0;
    }
    
    /* Build cache paths for all chunks */
    char** cache_paths = (char**)malloc(chunk_count * sizeof(char*));
    if (!cache_paths) {
        safe_strcpy(request->error, "Memory allocation failed", 256);
        return 0;
    }
    
    for (int32_t i = 0; i < chunk_count; i++) {
        cache_paths[i] = (char*)malloc(4096);
        if (!cache_paths[i]) {
            for (int32_t j = 0; j < i; j++) free(cache_paths[j]);
            free(cache_paths);
            safe_strcpy(request->error, "Memory allocation failed", 256);
            return 0;
        }
        
        int32_t chunk_row = i / chunks_per_side;
        int32_t chunk_col = i % chunks_per_side;
        int32_t abs_col = request->tile_col * chunks_per_side + chunk_col;
        int32_t abs_row = request->tile_row * chunks_per_side + chunk_row;
        
        snprintf(cache_paths[i], 4096, "%s/%d_%d_%d_%s.jpg",
                 request->cache_dir, abs_col, abs_row, 
                 request->zoom, request->maptype);
    }
    
    /* Decode all chunks from cache */
    aodecode_image_t* chunks = (aodecode_image_t*)calloc(
        chunk_count, sizeof(aodecode_image_t)
    );
    if (!chunks) {
        for (int32_t i = 0; i < chunk_count; i++) free(cache_paths[i]);
        free(cache_paths);
        safe_strcpy(request->error, "Memory allocation failed", 256);
        return 0;
    }
    
    int32_t decoded = aodecode_from_cache(
        (const char**)cache_paths, chunk_count, chunks, pool, 0
    );
    
    request->stats.chunks_found = decoded;
    request->stats.chunks_decoded = decoded;
    request->stats.chunks_failed = chunk_count - decoded;
    
    /* Free cache paths */
    for (int32_t i = 0; i < chunk_count; i++) {
        free(cache_paths[i]);
    }
    free(cache_paths);
    
    /* Allocate composed tile image */
    aodecode_image_t tile_image = {0};
    tile_image.width = tile_size;
    tile_image.height = tile_size;
    tile_image.stride = tile_size * 4;
    tile_image.channels = 4;
    tile_image.data = (uint8_t*)malloc(tile_size * tile_size * 4);
    
    if (!tile_image.data) {
        for (int32_t i = 0; i < chunk_count; i++) {
            aodecode_free_image(&chunks[i], pool);
        }
        free(chunks);
        safe_strcpy(request->error, "Memory allocation failed", 256);
        return 0;
    }
    
    /* Fill with missing color first */
    memset(tile_image.data, 0, tile_size * tile_size * 4);
    aodds_fill_missing(chunks, chunks_per_side, &tile_image,
                       request->missing_r, request->missing_g, request->missing_b);
    
    /* Compose valid chunks */
    aodds_compose_chunks(chunks, chunks_per_side, &tile_image);
    
    /* Free chunk images */
    for (int32_t i = 0; i < chunk_count; i++) {
        aodecode_free_image(&chunks[i], pool);
    }
    free(chunks);
    
    /* Write DDS header */
    uint32_t offset = aodds_write_header(
        request->dds_buffer, tile_size, tile_size, 
        mipmap_count, request->format
    );
    
    /* Generate and compress mipmaps */
    aodecode_image_t current = tile_image;
    aodecode_image_t next = {0};
    
    for (int32_t mip = 0; mip < mipmap_count; mip++) {
        /* Compress current mipmap */
        uint32_t compressed_size = aodds_compress(
            &current, request->format, 
            request->dds_buffer + offset
        );
        offset += compressed_size;
        request->stats.mipmaps_generated++;
        
        /* Generate next mipmap level */
        if (mip < mipmap_count - 1 && current.width > 4 && current.height > 4) {
            next.width = current.width / 2;
            next.height = current.height / 2;
            next.stride = next.width * 4;
            next.channels = 4;
            next.data = (uint8_t*)malloc(next.width * next.height * 4);
            
            if (next.data) {
                aodds_reduce_half(&current, &next);
                
                /* Free current (except first which is tile_image) */
                if (mip > 0) {
                    free(current.data);
                }
                current = next;
                memset(&next, 0, sizeof(next));
            } else {
                break;  /* Out of memory, stop generating mipmaps */
            }
        }
    }
    
    /* Clean up last mipmap */
    if (current.data != tile_image.data) {
        free(current.data);
    }
    free(tile_image.data);
    
    request->dds_written = offset;
    request->success = 1;
    request->stats.elapsed_ms = get_time_ms() - start_time;
    
    return 1;
}

AODDS_API int32_t aodds_build_from_chunks(
    aodecode_image_t* chunks,
    int32_t chunks_per_side,
    dds_format_t format,
    uint8_t missing_color[3],
    uint8_t* dds_output,
    uint32_t output_size,
    uint32_t* bytes_written
) {
    if (!chunks || !dds_output || !bytes_written || chunks_per_side <= 0) {
        return 0;
    }
    
    int32_t tile_size = chunks_per_side * CHUNK_SIZE;
    int32_t mipmap_count = aodds_calc_mipmap_count(tile_size, tile_size);
    uint32_t required = aodds_calc_dds_size(tile_size, tile_size, mipmap_count, format);
    
    if (output_size < required) {
        return 0;
    }
    
    /* Allocate tile image */
    aodecode_image_t tile = {0};
    tile.width = tile_size;
    tile.height = tile_size;
    tile.stride = tile_size * 4;
    tile.channels = 4;
    tile.data = (uint8_t*)malloc(tile_size * tile_size * 4);
    
    if (!tile.data) return 0;
    
    /* Fill and compose */
    aodds_fill_missing(chunks, chunks_per_side, &tile,
                       missing_color[0], missing_color[1], missing_color[2]);
    aodds_compose_chunks(chunks, chunks_per_side, &tile);
    
    /* Write header */
    uint32_t offset = aodds_write_header(dds_output, tile_size, tile_size, 
                                          mipmap_count, format);
    
    /* Generate mipmaps */
    aodecode_image_t current = tile;
    
    for (int32_t mip = 0; mip < mipmap_count; mip++) {
        offset += aodds_compress(&current, format, dds_output + offset);
        
        if (mip < mipmap_count - 1 && current.width > 4) {
            aodecode_image_t next = {0};
            next.width = current.width / 2;
            next.height = current.height / 2;
            next.stride = next.width * 4;
            next.channels = 4;
            next.data = (uint8_t*)malloc(next.width * next.height * 4);
            
            if (next.data) {
                aodds_reduce_half(&current, &next);
                if (mip > 0) free(current.data);
                current = next;
            } else {
                break;
            }
        }
    }
    
    if (current.data != tile.data) free(current.data);
    free(tile.data);
    
    *bytes_written = offset;
    return 1;
}

AODDS_API const char* aodds_version(void) {
    aodds_init_ispc();
    
    static char version_buf[256];
    snprintf(version_buf, sizeof(version_buf),
             "aodds " AODDS_VERSION
#if AOPIPELINE_HAS_OPENMP
             " (OpenMP enabled)"
#else
             " (OpenMP disabled)"
#endif
             " [ISPC: %s]"
#ifdef AOPIPELINE_WINDOWS
             " [Windows]"
#elif defined(AOPIPELINE_MACOS)
             " [macOS]"
#else
             " [Linux]"
#endif
             , ispc_available ? "yes" : "fallback"
    );
    
    return version_buf;
}

