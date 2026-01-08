/**
 * aodds.c - Native DDS Texture Building Implementation
 * 
 * Complete native pipeline for building DDS textures from cached JPEGs.
 * Includes ISPC texcomp integration for high-performance compression.
 */

/* Enable dladdr() on Linux/glibc */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "aodds.h"
#include "aodecode.h"
#include "aocache.h"
#include "internal.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <turbojpeg.h>

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
static int force_fallback = 0;  /* When set, use fallback even if ISPC is available */

#ifdef AOPIPELINE_WINDOWS
static HMODULE ispc_lib = NULL;
#else
static void* ispc_lib = NULL;
#endif

/* Helper to get directory containing a path */
static void get_directory(char* dest, size_t dest_size, const char* path) {
    /* Use snprintf for safe copy with guaranteed null-termination */
    snprintf(dest, dest_size, "%s", path);
    
    /* Find last separator */
    char* last_sep = NULL;
    for (char* p = dest; *p; p++) {
#ifdef AOPIPELINE_WINDOWS
        if (*p == '\\' || *p == '/') last_sep = p;
#else
        if (*p == '/') last_sep = p;
#endif
    }
    if (last_sep) {
        *last_sep = '\0';
    }
}

AODDS_API int32_t aodds_init_ispc(void) {
    if (ispc_initialized) {
        return ispc_available;
    }
    ispc_initialized = 1;
    
    /* Determine library name and subdirectory */
    const char* lib_name;
    const char* lib_subdir;
#ifdef AOPIPELINE_WINDOWS
    lib_name = "ispc_texcomp.dll";
    lib_subdir = "windows";
#elif defined(AOPIPELINE_MACOS)
    lib_name = "libispc_texcomp.dylib";
    lib_subdir = "macos";
#else
    lib_name = "libispc_texcomp.so";
    lib_subdir = "linux";
#endif
    
    char module_path[4096] = {0};
    char module_dir[4096] = {0};
    char lib_path[4096] = {0};
    
#ifdef AOPIPELINE_WINDOWS
    /* Get path to this DLL (aopipeline.dll) */
    HMODULE self_module = NULL;
    GetModuleHandleExA(
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        (LPCSTR)aodds_init_ispc,
        &self_module
    );
    if (self_module) {
        GetModuleFileNameA(self_module, module_path, sizeof(module_path));
        get_directory(module_dir, sizeof(module_dir), module_path);
        
        /* 
         * aopipeline.dll is in: autoortho/aopipeline/lib/windows/
         * ISPC lib is in:       autoortho/lib/windows/
         * So we need: ../../lib/windows/ispc_texcomp.dll
         */
        snprintf(lib_path, sizeof(lib_path), "%s\\..\\..\\..\\lib\\%s\\%s", 
                 module_dir, lib_subdir, lib_name);
        ispc_lib = LoadLibraryA(lib_path);
        
        /* Also try the aopipeline lib dir (in case user copied it there) */
        if (!ispc_lib) {
            snprintf(lib_path, sizeof(lib_path), "%s\\%s", module_dir, lib_name);
            ispc_lib = LoadLibraryA(lib_path);
        }
    }
    
    /* Fallback: try just the library name (relies on PATH) */
    if (!ispc_lib) {
        ispc_lib = LoadLibraryA(lib_name);
    }
    
    if (ispc_lib) {
        ispc_compress_bc1 = (CompressBlocksBC1_fn)GetProcAddress(ispc_lib, "CompressBlocksBC1");
        ispc_compress_bc3 = (CompressBlocksBC3_fn)GetProcAddress(ispc_lib, "CompressBlocksBC3");
    }
#else
    /* Unix: use dladdr to find this shared library's path */
    Dl_info dl_info;
    if (dladdr((void*)aodds_init_ispc, &dl_info) && dl_info.dli_fname) {
        get_directory(module_dir, sizeof(module_dir), dl_info.dli_fname);
        
        /* 
         * libaopipeline.so is in: autoortho/aopipeline/lib/{linux,macos}/
         * ISPC lib is in:         autoortho/lib/{linux,macos}/
         * So we need: ../../../lib/{platform}/libispc_texcomp.{so,dylib}
         */
        snprintf(lib_path, sizeof(lib_path), "%s/../../../lib/%s/%s", 
                 module_dir, lib_subdir, lib_name);
        ispc_lib = dlopen(lib_path, RTLD_NOW);
        
        /* Also try the aopipeline lib dir (in case user copied it there) */
        if (!ispc_lib) {
            snprintf(lib_path, sizeof(lib_path), "%s/%s", module_dir, lib_name);
            ispc_lib = dlopen(lib_path, RTLD_NOW);
        }
    }
    
    /* Fallback: try standard library paths */
    if (!ispc_lib) {
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

AODDS_API void aodds_set_use_ispc(int32_t use_ispc) {
    force_fallback = !use_ispc;
}

AODDS_API int32_t aodds_get_use_ispc(void) {
    return ispc_available && !force_fallback;
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

/*============================================================================
 * BC3 Fallback Compression (DXT5)
 * 
 * BC3 block layout (16 bytes total):
 *   - Bytes 0-7:  Alpha block (DXT5 alpha compression)
 *   - Bytes 8-15: Color block (same as BC1/DXT1)
 * 
 * Alpha block format:
 *   - Byte 0: alpha0 (reference alpha value)
 *   - Byte 1: alpha1 (reference alpha value)
 *   - Bytes 2-7: 16 3-bit indices (48 bits) selecting from 8 derived alphas
 * 
 * Alpha derivation (when alpha0 > alpha1):
 *   alpha[0] = alpha0
 *   alpha[1] = alpha1
 *   alpha[2] = (6*alpha0 + 1*alpha1) / 7
 *   alpha[3] = (5*alpha0 + 2*alpha1) / 7
 *   alpha[4] = (4*alpha0 + 3*alpha1) / 7
 *   alpha[5] = (3*alpha0 + 4*alpha1) / 7
 *   alpha[6] = (2*alpha0 + 5*alpha1) / 7
 *   alpha[7] = (1*alpha0 + 6*alpha1) / 7
 *============================================================================*/

static void compress_alpha_block(const uint8_t* rgba, uint8_t* block) {
    /* Find min/max alpha values in the 4x4 block */
    uint8_t min_alpha = 255;
    uint8_t max_alpha = 0;
    
    for (int i = 0; i < 16; i++) {
        uint8_t a = rgba[i * 4 + 3];  /* Alpha is 4th component */
        if (a < min_alpha) min_alpha = a;
        if (a > max_alpha) max_alpha = a;
    }
    
    /* Set reference alpha values
     * Use alpha0 > alpha1 mode for full 8-value interpolation */
    uint8_t alpha0, alpha1;
    if (max_alpha == min_alpha) {
        /* Solid alpha - avoid division issues */
        alpha0 = max_alpha;
        alpha1 = max_alpha > 0 ? max_alpha - 1 : 0;
    } else {
        alpha0 = max_alpha;
        alpha1 = min_alpha;
    }
    
    /* Ensure alpha0 > alpha1 for 8-value interpolation mode */
    if (alpha0 <= alpha1) {
        alpha0 = alpha1 + 1;
        if (alpha0 == 0) alpha0 = 1;  /* Handle overflow */
    }
    
    block[0] = alpha0;
    block[1] = alpha1;
    
    /* Calculate 8 interpolated alpha values */
    int32_t alphas[8];
    alphas[0] = alpha0;
    alphas[1] = alpha1;
    alphas[2] = (6 * alpha0 + 1 * alpha1 + 3) / 7;
    alphas[3] = (5 * alpha0 + 2 * alpha1 + 3) / 7;
    alphas[4] = (4 * alpha0 + 3 * alpha1 + 3) / 7;
    alphas[5] = (3 * alpha0 + 4 * alpha1 + 3) / 7;
    alphas[6] = (2 * alpha0 + 5 * alpha1 + 3) / 7;
    alphas[7] = (1 * alpha0 + 6 * alpha1 + 3) / 7;
    
    /* Encode 16 3-bit indices into 48 bits (6 bytes) */
    uint64_t indices = 0;
    for (int i = 0; i < 16; i++) {
        int32_t a = rgba[i * 4 + 3];
        
        /* Find closest alpha value */
        int32_t best_idx = 0;
        int32_t best_dist = INT32_MAX;
        for (int j = 0; j < 8; j++) {
            int32_t dist = (a - alphas[j]) * (a - alphas[j]);
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = j;
            }
        }
        
        indices |= ((uint64_t)best_idx << (i * 3));
    }
    
    /* Write 48 bits (6 bytes) of indices */
    block[2] = (indices >> 0) & 0xFF;
    block[3] = (indices >> 8) & 0xFF;
    block[4] = (indices >> 16) & 0xFF;
    block[5] = (indices >> 24) & 0xFF;
    block[6] = (indices >> 32) & 0xFF;
    block[7] = (indices >> 40) & 0xFF;
}

static void compress_bc3_fallback(const aodecode_image_t* image, uint8_t* output) {
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
            
            /* BC3 block = 16 bytes: 8 bytes alpha + 8 bytes color */
            uint8_t* block_out = output + ((by * blocks_x + bx) * 16);
            
            /* Compress alpha block (bytes 0-7) */
            compress_alpha_block(block_rgba, block_out);
            
            /* Compress color block (bytes 8-15) - same as BC1 */
            compress_bc1_block_simple(block_rgba, block_out + 8);
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
    
    if (ispc_available && !force_fallback) {
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
        /* Fallback compression (simple inline implementation) */
        if (format == DDS_FORMAT_BC1) {
            compress_bc1_fallback(image, output);
        } else {
            /* BC3 = alpha block + BC1 color block */
            compress_bc3_fallback(image, output);
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
    
    /* ═══════════════════════════════════════════════════════════════════════
     * PATH STRING OPTIMIZATION (Phase 2.1)
     * ═══════════════════════════════════════════════════════════════════════
     * BEFORE: 256 mallocs of 4KB each = 1MB total, 257 allocations
     * AFTER:  1 contiguous buffer with 512-byte slots = 128KB, 2 allocations
     * 
     * Path format: {cache_dir}/{col}_{row}_{zoom}_{maptype}.jpg
     * Typical path: ~60-80 chars, 512 bytes is generous with room for long paths
     * ═══════════════════════════════════════════════════════════════════════*/
    #define PATH_SLOT_SIZE 512
    
    char* path_buffer = (char*)malloc(chunk_count * PATH_SLOT_SIZE);
    char** cache_paths = (char**)malloc(chunk_count * sizeof(char*));
    
    if (!path_buffer || !cache_paths) {
        free(path_buffer);
        free(cache_paths);
        safe_strcpy(request->error, "Memory allocation failed", 256);
        return 0;
    }
    
    for (int32_t i = 0; i < chunk_count; i++) {
        cache_paths[i] = path_buffer + (i * PATH_SLOT_SIZE);
        
        int32_t chunk_row = i / chunks_per_side;
        int32_t chunk_col = i % chunks_per_side;
        int32_t abs_col = request->tile_col * chunks_per_side + chunk_col;
        int32_t abs_row = request->tile_row * chunks_per_side + chunk_row;
        
        snprintf(cache_paths[i], PATH_SLOT_SIZE, "%s/%d_%d_%d_%s.jpg",
                 request->cache_dir, abs_col, abs_row, 
                 request->zoom, request->maptype);
    }
    
    /* Decode all chunks from cache */
    aodecode_image_t* chunks = (aodecode_image_t*)calloc(
        chunk_count, sizeof(aodecode_image_t)
    );
    if (!chunks) {
        free(path_buffer);
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
    
    /* Free path buffers (now just 2 frees instead of 257) */
    free(path_buffer);
    free(cache_paths);
    
    #undef PATH_SLOT_SIZE
    
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
    
    /* ═══════════════════════════════════════════════════════════════════════
     * MIPMAP BUFFER REUSE (Phase 2.2)
     * ═══════════════════════════════════════════════════════════════════════
     * BEFORE: Allocate new buffer for each mipmap level (~12 allocations)
     * AFTER:  Pre-allocate two ping-pong buffers (2 allocations)
     * 
     * For 4096x4096 tile:
     * - mip_buf_a: 16MB (for levels 1, 3, 5, 7, 9, 11 - odd levels)
     * - mip_buf_b: 4MB (for levels 2, 4, 6, 8, 10 - even levels)
     * 
     * Level 0: compress tile_image
     * Level 1: reduce tile→mip_a, compress mip_a
     * Level 2: reduce mip_a→mip_b, compress mip_b
     * Level 3: reduce mip_b→mip_a (fits in 16MB), compress mip_a
     * ... alternating between buffers
     * ═══════════════════════════════════════════════════════════════════════*/
    
    /* Allocate ping-pong buffers for mipmaps */
    int32_t mip1_size = (tile_size / 2) * (tile_size / 2) * 4;  /* Level 1 size */
    int32_t mip2_size = (tile_size / 4) * (tile_size / 4) * 4;  /* Level 2 size */
    
    uint8_t* mip_buf_a = (mipmap_count > 1) ? (uint8_t*)malloc(mip1_size) : NULL;
    uint8_t* mip_buf_b = (mipmap_count > 2) ? (uint8_t*)malloc(mip2_size) : NULL;
    
    aodecode_image_t current = tile_image;
    aodecode_image_t next = {0};
    int use_buf_a = 1;  /* Toggle for ping-pong */
    
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
            
            /* Use pre-allocated ping-pong buffers */
            if (use_buf_a && mip_buf_a) {
                next.data = mip_buf_a;
            } else if (!use_buf_a && mip_buf_b) {
                next.data = mip_buf_b;
            } else {
                /* Fallback: allocate if ping-pong buffer unavailable */
                next.data = (uint8_t*)malloc(next.width * next.height * 4);
            }
            
            if (next.data) {
                aodds_reduce_half(&current, &next);
                current = next;
                memset(&next, 0, sizeof(next));
                use_buf_a = !use_buf_a;  /* Toggle for next iteration */
            } else {
                break;  /* Out of memory, stop generating mipmaps */
            }
        }
    }
    
    /* Clean up - only free the pre-allocated buffers, not the reused ones */
    free(mip_buf_a);
    free(mip_buf_b);
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
    
    /* ═══════════════════════════════════════════════════════════════════════
     * MIPMAP BUFFER REUSE (Phase 2.2)
     * Pre-allocate ping-pong buffers instead of allocating per mipmap level
     * ═══════════════════════════════════════════════════════════════════════*/
    int32_t mip1_size = (tile_size / 2) * (tile_size / 2) * 4;
    int32_t mip2_size = (tile_size / 4) * (tile_size / 4) * 4;
    
    uint8_t* mip_buf_a = (mipmap_count > 1) ? (uint8_t*)malloc(mip1_size) : NULL;
    uint8_t* mip_buf_b = (mipmap_count > 2) ? (uint8_t*)malloc(mip2_size) : NULL;
    
    aodecode_image_t current = tile;
    aodecode_image_t next = {0};
    int use_buf_a = 1;
    
    for (int32_t mip = 0; mip < mipmap_count; mip++) {
        offset += aodds_compress(&current, format, dds_output + offset);
        
        if (mip < mipmap_count - 1 && current.width > 4) {
            next.width = current.width / 2;
            next.height = current.height / 2;
            next.stride = next.width * 4;
            next.channels = 4;
            
            /* Use pre-allocated ping-pong buffers */
            if (use_buf_a && mip_buf_a) {
                next.data = mip_buf_a;
            } else if (!use_buf_a && mip_buf_b) {
                next.data = mip_buf_b;
            } else {
                next.data = (uint8_t*)malloc(next.width * next.height * 4);
            }
            
            if (next.data) {
                aodds_reduce_half(&current, &next);
                current = next;
                memset(&next, 0, sizeof(next));
                use_buf_a = !use_buf_a;
            } else {
                break;
            }
        }
    }
    
    free(mip_buf_a);
    free(mip_buf_b);
    free(tile.data);
    
    *bytes_written = offset;
    return 1;
}

/*============================================================================
 * Hybrid Pipeline: Build DDS from pre-read JPEG data
 * 
 * This is the optimal approach:
 * - Python reads cache files (fast for OS-cached files)
 * - Native decodes + composes + compresses (parallelism helps)
 *============================================================================*/

AODDS_API int32_t aodds_build_from_jpegs(
    const uint8_t** jpeg_data,
    const uint32_t* jpeg_sizes,
    int32_t chunk_count,
    dds_format_t format,
    uint8_t missing_r,
    uint8_t missing_g,
    uint8_t missing_b,
    uint8_t* dds_output,
    uint32_t output_size,
    uint32_t* bytes_written,
    aodecode_pool_t* pool
) {
    if (!jpeg_data || !jpeg_sizes || !dds_output || !bytes_written || chunk_count <= 0) {
        return 0;
    }
    
    /* Calculate chunks per side (must be perfect square) */
    int32_t chunks_per_side = (int32_t)sqrt((double)chunk_count);
    if (chunks_per_side * chunks_per_side != chunk_count) {
        return 0;  /* Not a perfect square */
    }
    
    int32_t tile_size = chunks_per_side * CHUNK_SIZE;
    int32_t mipmap_count = aodds_calc_mipmap_count(tile_size, tile_size);
    uint32_t required = aodds_calc_dds_size(tile_size, tile_size, mipmap_count, format);
    
    if (output_size < required) {
        return 0;
    }
    
    /* Allocate chunk image array */
    aodecode_image_t* chunks = (aodecode_image_t*)calloc(
        chunk_count, sizeof(aodecode_image_t)
    );
    if (!chunks) {
        return 0;
    }
    
    /* Parallel decode all JPEGs */
    int32_t decoded = 0;
    
#if AOPIPELINE_HAS_OPENMP
    #pragma omp parallel reduction(+:decoded)
    {
        /* Each thread gets its own turbojpeg handle */
        tjhandle tjh = tjInitDecompress();
        if (tjh) {
            #pragma omp for schedule(static)
            for (int32_t i = 0; i < chunk_count; i++) {
                if (!jpeg_data[i] || jpeg_sizes[i] == 0) {
                    continue;  /* Missing chunk - will use fill color */
                }
                
                /* Get JPEG dimensions */
                int width, height, subsamp, colorspace;
                if (tjDecompressHeader3(tjh, jpeg_data[i], jpeg_sizes[i],
                                        &width, &height, &subsamp, &colorspace) < 0) {
                    continue;
                }
                
                /* Validate dimensions */
                if (width != CHUNK_SIZE || height != CHUNK_SIZE) {
                    continue;  /* Unexpected size */
                }
                
                /* Acquire buffer from pool or malloc */
                uint8_t* buffer;
                int from_pool = 0;
                if (pool) {
                    buffer = aodecode_acquire_buffer(pool);
                    from_pool = (buffer != NULL);
                } else {
                    buffer = (uint8_t*)malloc(CHUNK_SIZE * CHUNK_SIZE * 4);
                }
                
                if (!buffer) continue;
                
                /* Decode JPEG to RGBA */
                if (tjDecompress2(tjh, jpeg_data[i], jpeg_sizes[i],
                                  buffer, width, 0, height,
                                  TJPF_RGBA, TJFLAG_FASTDCT) < 0) {
                    if (from_pool) {
                        aodecode_release_buffer(pool, buffer);
                    } else {
                        free(buffer);
                    }
                    continue;
                }
                
                /* Fill chunk structure */
                chunks[i].data = buffer;
                chunks[i].width = width;
                chunks[i].height = height;
                chunks[i].stride = width * 4;
                chunks[i].channels = 4;
                chunks[i].from_pool = from_pool;
                decoded++;
            }
            tjDestroy(tjh);
        }
    }
#else
    /* Non-OpenMP fallback: sequential decode */
    tjhandle tjh = tjInitDecompress();
    if (tjh) {
        for (int32_t i = 0; i < chunk_count; i++) {
            if (!jpeg_data[i] || jpeg_sizes[i] == 0) {
                continue;
            }
            
            int width, height, subsamp, colorspace;
            if (tjDecompressHeader3(tjh, jpeg_data[i], jpeg_sizes[i],
                                    &width, &height, &subsamp, &colorspace) < 0) {
                continue;
            }
            
            if (width != CHUNK_SIZE || height != CHUNK_SIZE) {
                continue;
            }
            
            uint8_t* buffer = (uint8_t*)malloc(CHUNK_SIZE * CHUNK_SIZE * 4);
            if (!buffer) continue;
            
            if (tjDecompress2(tjh, jpeg_data[i], jpeg_sizes[i],
                              buffer, width, 0, height,
                              TJPF_RGBA, TJFLAG_FASTDCT) < 0) {
                free(buffer);
                continue;
            }
            
            chunks[i].data = buffer;
            chunks[i].width = width;
            chunks[i].height = height;
            chunks[i].stride = width * 4;
            chunks[i].channels = 4;
            chunks[i].from_pool = 0;
            decoded++;
        }
        tjDestroy(tjh);
    }
#endif
    
    /* Allocate tile image */
    aodecode_image_t tile = {0};
    tile.width = tile_size;
    tile.height = tile_size;
    tile.stride = tile_size * 4;
    tile.channels = 4;
    tile.data = (uint8_t*)malloc(tile_size * tile_size * 4);
    
    if (!tile.data) {
        for (int32_t i = 0; i < chunk_count; i++) {
            aodecode_free_image(&chunks[i], pool);
        }
        free(chunks);
        return 0;
    }
    
    /* Fill with missing color and compose chunks */
    uint8_t missing_color[3] = {missing_r, missing_g, missing_b};
    aodds_fill_missing(chunks, chunks_per_side, &tile,
                       missing_color[0], missing_color[1], missing_color[2]);
    aodds_compose_chunks(chunks, chunks_per_side, &tile);
    
    /* Free chunk images */
    for (int32_t i = 0; i < chunk_count; i++) {
        aodecode_free_image(&chunks[i], pool);
    }
    free(chunks);
    
    /* Write header and generate mipmaps */
    uint32_t offset = aodds_write_header(dds_output, tile_size, tile_size, 
                                          mipmap_count, format);
    
    /* ═══════════════════════════════════════════════════════════════════════
     * MIPMAP BUFFER REUSE (Phase 2.2)
     * Pre-allocate ping-pong buffers instead of allocating per mipmap level
     * ═══════════════════════════════════════════════════════════════════════*/
    int32_t mip1_size = (tile_size / 2) * (tile_size / 2) * 4;
    int32_t mip2_size = (tile_size / 4) * (tile_size / 4) * 4;
    
    uint8_t* mip_buf_a = (mipmap_count > 1) ? (uint8_t*)malloc(mip1_size) : NULL;
    uint8_t* mip_buf_b = (mipmap_count > 2) ? (uint8_t*)malloc(mip2_size) : NULL;
    
    aodecode_image_t current = tile;
    aodecode_image_t next = {0};
    int use_buf_a = 1;
    
    for (int32_t mip = 0; mip < mipmap_count; mip++) {
        offset += aodds_compress(&current, format, dds_output + offset);
        
        if (mip < mipmap_count - 1 && current.width > 4) {
            next.width = current.width / 2;
            next.height = current.height / 2;
            next.stride = next.width * 4;
            next.channels = 4;
            
            /* Use pre-allocated ping-pong buffers */
            if (use_buf_a && mip_buf_a) {
                next.data = mip_buf_a;
            } else if (!use_buf_a && mip_buf_b) {
                next.data = mip_buf_b;
            } else {
                next.data = (uint8_t*)malloc(next.width * next.height * 4);
            }
            
            if (next.data) {
                aodds_reduce_half(&current, &next);
                current = next;
                memset(&next, 0, sizeof(next));
                use_buf_a = !use_buf_a;
            } else {
                break;
            }
        }
    }
    
    free(mip_buf_a);
    free(mip_buf_b);
    free(tile.data);
    
    *bytes_written = offset;
    return 1;
}

/*============================================================================
 * Direct-to-File Pipeline: Build DDS and write directly to disk
 * 
 * PERFORMANCE OPTIMIZATION:
 * This eliminates the ~65ms Python copy overhead by writing DDS data
 * directly to the disk cache file. The flow becomes:
 *   1. Decode JPEGs to RGBA (parallel, in C)
 *   2. Compose tile image
 *   3. For each mipmap: compress and fwrite to file
 *   4. No Python involvement, no memory copy
 * 
 * ATOMICITY:
 * Uses temp file + rename pattern to prevent corrupt files on crash.
 *============================================================================*/

AODDS_API int32_t aodds_build_from_jpegs_to_file(
    const uint8_t** jpeg_data,
    const uint32_t* jpeg_sizes,
    int32_t chunk_count,
    dds_format_t format,
    uint8_t missing_r,
    uint8_t missing_g,
    uint8_t missing_b,
    const char* output_path,
    uint32_t* bytes_written,
    aodecode_pool_t* pool
) {
    if (!jpeg_data || !jpeg_sizes || !output_path || !bytes_written || chunk_count <= 0) {
        return 0;
    }
    
    *bytes_written = 0;
    
    /* Calculate chunks per side (must be perfect square) */
    int32_t chunks_per_side = (int32_t)sqrt((double)chunk_count);
    if (chunks_per_side * chunks_per_side != chunk_count) {
        return 0;
    }
    
    int32_t tile_size = chunks_per_side * CHUNK_SIZE;
    int32_t mipmap_count = aodds_calc_mipmap_count(tile_size, tile_size);
    
    /* Allocate chunk image array */
    aodecode_image_t* chunks = (aodecode_image_t*)calloc(
        chunk_count, sizeof(aodecode_image_t)
    );
    if (!chunks) {
        return 0;
    }
    
    /* Parallel decode all JPEGs (same as aodds_build_from_jpegs) */
    int32_t decoded = 0;
    
#if AOPIPELINE_HAS_OPENMP
    #pragma omp parallel reduction(+:decoded)
    {
        tjhandle tjh = tjInitDecompress();
        if (tjh) {
            #pragma omp for schedule(static)
            for (int32_t i = 0; i < chunk_count; i++) {
                if (!jpeg_data[i] || jpeg_sizes[i] == 0) {
                    continue;
                }
                
                int width, height, subsamp, colorspace;
                if (tjDecompressHeader3(tjh, jpeg_data[i], jpeg_sizes[i],
                                        &width, &height, &subsamp, &colorspace) < 0) {
                    continue;
                }
                
                if (width != CHUNK_SIZE || height != CHUNK_SIZE) {
                    continue;
                }
                
                uint8_t* buffer;
                int from_pool = 0;
                if (pool) {
                    buffer = aodecode_acquire_buffer(pool);
                    from_pool = (buffer != NULL);
                }
                if (!from_pool) {
                    buffer = (uint8_t*)malloc(CHUNK_SIZE * CHUNK_SIZE * 4);
                }
                
                if (!buffer) continue;
                
                if (tjDecompress2(tjh, jpeg_data[i], jpeg_sizes[i],
                                  buffer, width, 0, height,
                                  TJPF_RGBA, TJFLAG_FASTDCT) < 0) {
                    if (from_pool) {
                        aodecode_release_buffer(pool, buffer);
                    } else {
                        free(buffer);
                    }
                    continue;
                }
                
                chunks[i].data = buffer;
                chunks[i].width = width;
                chunks[i].height = height;
                chunks[i].stride = width * 4;
                chunks[i].channels = 4;
                chunks[i].from_pool = from_pool;
                decoded++;
            }
            tjDestroy(tjh);
        }
    }
#else
    /* Sequential fallback */
    tjhandle tjh = tjInitDecompress();
    if (tjh) {
        for (int32_t i = 0; i < chunk_count; i++) {
            if (!jpeg_data[i] || jpeg_sizes[i] == 0) continue;
            
            int width, height, subsamp, colorspace;
            if (tjDecompressHeader3(tjh, jpeg_data[i], jpeg_sizes[i],
                                    &width, &height, &subsamp, &colorspace) < 0) {
                continue;
            }
            
            if (width != CHUNK_SIZE || height != CHUNK_SIZE) continue;
            
            uint8_t* buffer = (uint8_t*)malloc(CHUNK_SIZE * CHUNK_SIZE * 4);
            if (!buffer) continue;
            
            if (tjDecompress2(tjh, jpeg_data[i], jpeg_sizes[i],
                              buffer, width, 0, height,
                              TJPF_RGBA, TJFLAG_FASTDCT) < 0) {
                free(buffer);
                continue;
            }
            
            chunks[i].data = buffer;
            chunks[i].width = width;
            chunks[i].height = height;
            chunks[i].stride = width * 4;
            chunks[i].channels = 4;
            chunks[i].from_pool = 0;
            decoded++;
        }
        tjDestroy(tjh);
    }
#endif
    
    /* Allocate tile image */
    aodecode_image_t tile = {0};
    tile.width = tile_size;
    tile.height = tile_size;
    tile.stride = tile_size * 4;
    tile.channels = 4;
    tile.data = (uint8_t*)malloc(tile_size * tile_size * 4);
    
    if (!tile.data) {
        for (int32_t i = 0; i < chunk_count; i++) {
            aodecode_free_image(&chunks[i], pool);
        }
        free(chunks);
        return 0;
    }
    
    /* Fill with missing color and compose chunks */
    aodds_fill_missing(chunks, chunks_per_side, &tile, missing_r, missing_g, missing_b);
    aodds_compose_chunks(chunks, chunks_per_side, &tile);
    
    /* Free chunk images */
    for (int32_t i = 0; i < chunk_count; i++) {
        aodecode_free_image(&chunks[i], pool);
    }
    free(chunks);
    
    /* ═══════════════════════════════════════════════════════════════════════
     * FILE OUTPUT: Write DDS directly to disk
     * Uses temp file + rename for atomicity
     * ═══════════════════════════════════════════════════════════════════════*/
    
    /* Create temp file path */
    char temp_path[4096];
    snprintf(temp_path, sizeof(temp_path), "%s.tmp", output_path);
    
    /* Open temp file for writing */
    FILE* fp = fopen(temp_path, "wb");
    if (!fp) {
        free(tile.data);
        return 0;
    }
    
    /* Write DDS header */
    uint8_t header[DDS_HEADER_SIZE];
    aodds_write_header(header, tile_size, tile_size, mipmap_count, format);
    if (fwrite(header, 1, DDS_HEADER_SIZE, fp) != DDS_HEADER_SIZE) {
        fclose(fp);
        remove(temp_path);
        free(tile.data);
        return 0;
    }
    
    uint32_t total_written = DDS_HEADER_SIZE;
    
    /* Allocate compression buffer (largest mipmap size) */
    uint32_t block_size = (format == DDS_FORMAT_BC1) ? 8 : 16;
    uint32_t max_blocks_x = (tile_size + 3) / 4;
    uint32_t max_blocks_y = (tile_size + 3) / 4;
    uint32_t max_compressed_size = max_blocks_x * max_blocks_y * block_size;
    uint8_t* compress_buffer = (uint8_t*)malloc(max_compressed_size);
    
    if (!compress_buffer) {
        fclose(fp);
        remove(temp_path);
        free(tile.data);
        return 0;
    }
    
    /* ═══════════════════════════════════════════════════════════════════════
     * MIPMAP BUFFER REUSE (Phase 2.2)
     * Pre-allocate ping-pong buffers instead of allocating per mipmap level
     * ═══════════════════════════════════════════════════════════════════════*/
    int32_t mip1_size = (tile_size / 2) * (tile_size / 2) * 4;
    int32_t mip2_size = (tile_size / 4) * (tile_size / 4) * 4;
    
    uint8_t* mip_buf_a = (mipmap_count > 1) ? (uint8_t*)malloc(mip1_size) : NULL;
    uint8_t* mip_buf_b = (mipmap_count > 2) ? (uint8_t*)malloc(mip2_size) : NULL;
    
    /* Generate and write mipmaps */
    aodecode_image_t current = tile;
    aodecode_image_t next = {0};
    int success = 1;
    int use_buf_a = 1;
    
    for (int32_t mip = 0; mip < mipmap_count && success; mip++) {
        /* Compress current mipmap to buffer */
        uint32_t compressed_size = aodds_compress(&current, format, compress_buffer);
        
        /* Write compressed data to file */
        if (fwrite(compress_buffer, 1, compressed_size, fp) != compressed_size) {
            success = 0;
            break;
        }
        total_written += compressed_size;
        
        /* Generate next mipmap level */
        if (mip < mipmap_count - 1 && current.width > 4) {
            next.width = current.width / 2;
            next.height = current.height / 2;
            next.stride = next.width * 4;
            next.channels = 4;
            
            /* Use pre-allocated ping-pong buffers */
            if (use_buf_a && mip_buf_a) {
                next.data = mip_buf_a;
            } else if (!use_buf_a && mip_buf_b) {
                next.data = mip_buf_b;
            } else {
                next.data = (uint8_t*)malloc(next.width * next.height * 4);
            }
            
            if (next.data) {
                aodds_reduce_half(&current, &next);
                current = next;
                memset(&next, 0, sizeof(next));
                use_buf_a = !use_buf_a;
            } else {
                success = 0;
            }
        }
    }
    
    /* Cleanup - free ping-pong buffers and tile data */
    free(compress_buffer);
    free(mip_buf_a);
    free(mip_buf_b);
    free(tile.data);
    fclose(fp);
    
    if (!success) {
        remove(temp_path);
        return 0;
    }
    
    /* Atomic rename: temp file -> final path */
#ifdef AOPIPELINE_WINDOWS
    /* Windows: remove target first (rename fails if target exists) */
    remove(output_path);
#endif
    if (rename(temp_path, output_path) != 0) {
        remove(temp_path);
        return 0;
    }
    
    *bytes_written = total_written;
    return 1;
}

/*============================================================================
 * Native Direct-to-File: Build DDS from cache files and write to disk
 * 
 * PERFORMANCE OPTIMIZATION:
 * Same as aodds_build_tile() but writes directly to disk instead of buffer.
 * Eliminates ~65ms Python copy overhead for native mode predictive builds.
 *============================================================================*/

AODDS_API int32_t aodds_build_tile_to_file(
    const char* cache_dir,
    int32_t tile_row,
    int32_t tile_col,
    const char* maptype,
    int32_t zoom,
    int32_t chunks_per_side,
    dds_format_t format,
    uint8_t missing_r,
    uint8_t missing_g,
    uint8_t missing_b,
    const char* output_path,
    uint32_t* bytes_written,
    aodecode_pool_t* pool
) {
    if (!cache_dir || !maptype || !output_path || !bytes_written || chunks_per_side <= 0) {
        return 0;
    }
    
    *bytes_written = 0;
    
    int32_t chunk_count = chunks_per_side * chunks_per_side;
    int32_t tile_size = chunks_per_side * CHUNK_SIZE;
    int32_t mipmap_count = aodds_calc_mipmap_count(tile_size, tile_size);
    
    /* ═══════════════════════════════════════════════════════════════════════
     * PATH STRING OPTIMIZATION (Phase 2.1)
     * Contiguous buffer instead of individual allocations
     * ═══════════════════════════════════════════════════════════════════════*/
    #define PATH_SLOT_SIZE 512
    
    char* path_buffer = (char*)malloc(chunk_count * PATH_SLOT_SIZE);
    char** cache_paths = (char**)malloc(chunk_count * sizeof(char*));
    
    if (!path_buffer || !cache_paths) {
        free(path_buffer);
        free(cache_paths);
        return 0;
    }
    
    for (int32_t i = 0; i < chunk_count; i++) {
        cache_paths[i] = path_buffer + (i * PATH_SLOT_SIZE);
        
        int32_t chunk_row = i / chunks_per_side;
        int32_t chunk_col = i % chunks_per_side;
        int32_t abs_col = tile_col * chunks_per_side + chunk_col;
        int32_t abs_row = tile_row * chunks_per_side + chunk_row;
        
        snprintf(cache_paths[i], PATH_SLOT_SIZE, "%s/%d_%d_%d_%s.jpg",
                 cache_dir, abs_col, abs_row, zoom, maptype);
    }
    
    /* Decode all chunks from cache */
    aodecode_image_t* chunks = (aodecode_image_t*)calloc(
        chunk_count, sizeof(aodecode_image_t)
    );
    if (!chunks) {
        free(path_buffer);
        free(cache_paths);
        return 0;
    }
    
    aodecode_from_cache((const char**)cache_paths, chunk_count, chunks, pool, 0);
    
    /* Free path buffers */
    free(path_buffer);
    free(cache_paths);
    
    #undef PATH_SLOT_SIZE
    
    /* Allocate composed tile image */
    aodecode_image_t tile = {0};
    tile.width = tile_size;
    tile.height = tile_size;
    tile.stride = tile_size * 4;
    tile.channels = 4;
    tile.data = (uint8_t*)malloc(tile_size * tile_size * 4);
    
    if (!tile.data) {
        for (int32_t i = 0; i < chunk_count; i++) {
            aodecode_free_image(&chunks[i], pool);
        }
        free(chunks);
        return 0;
    }
    
    /* Fill with missing color and compose chunks */
    memset(tile.data, 0, tile_size * tile_size * 4);
    aodds_fill_missing(chunks, chunks_per_side, &tile, missing_r, missing_g, missing_b);
    aodds_compose_chunks(chunks, chunks_per_side, &tile);
    
    /* Free chunk images */
    for (int32_t i = 0; i < chunk_count; i++) {
        aodecode_free_image(&chunks[i], pool);
    }
    free(chunks);
    
    /* ═══════════════════════════════════════════════════════════════════════
     * FILE OUTPUT: Write DDS directly to disk with temp file + rename
     * ═══════════════════════════════════════════════════════════════════════*/
    
    char temp_path[4096];
    snprintf(temp_path, sizeof(temp_path), "%s.tmp", output_path);
    
    FILE* fp = fopen(temp_path, "wb");
    if (!fp) {
        free(tile.data);
        return 0;
    }
    
    /* Write DDS header */
    uint8_t header[DDS_HEADER_SIZE];
    aodds_write_header(header, tile_size, tile_size, mipmap_count, format);
    if (fwrite(header, 1, DDS_HEADER_SIZE, fp) != DDS_HEADER_SIZE) {
        fclose(fp);
        remove(temp_path);
        free(tile.data);
        return 0;
    }
    
    uint32_t total_written = DDS_HEADER_SIZE;
    
    /* Allocate compression buffer */
    uint32_t block_size = (format == DDS_FORMAT_BC1) ? 8 : 16;
    uint32_t max_blocks_x = (tile_size + 3) / 4;
    uint32_t max_blocks_y = (tile_size + 3) / 4;
    uint32_t max_compressed_size = max_blocks_x * max_blocks_y * block_size;
    uint8_t* compress_buffer = (uint8_t*)malloc(max_compressed_size);
    
    if (!compress_buffer) {
        fclose(fp);
        remove(temp_path);
        free(tile.data);
        return 0;
    }
    
    /* ═══════════════════════════════════════════════════════════════════════
     * MIPMAP BUFFER REUSE (Phase 2.2)
     * ═══════════════════════════════════════════════════════════════════════*/
    int32_t mip1_size = (tile_size / 2) * (tile_size / 2) * 4;
    int32_t mip2_size = (tile_size / 4) * (tile_size / 4) * 4;
    
    uint8_t* mip_buf_a = (mipmap_count > 1) ? (uint8_t*)malloc(mip1_size) : NULL;
    uint8_t* mip_buf_b = (mipmap_count > 2) ? (uint8_t*)malloc(mip2_size) : NULL;
    
    aodecode_image_t current = tile;
    aodecode_image_t next = {0};
    int success = 1;
    int use_buf_a = 1;
    
    for (int32_t mip = 0; mip < mipmap_count && success; mip++) {
        uint32_t compressed_size = aodds_compress(&current, format, compress_buffer);
        
        if (fwrite(compress_buffer, 1, compressed_size, fp) != compressed_size) {
            success = 0;
            break;
        }
        total_written += compressed_size;
        
        if (mip < mipmap_count - 1 && current.width > 4) {
            next.width = current.width / 2;
            next.height = current.height / 2;
            next.stride = next.width * 4;
            next.channels = 4;
            
            if (use_buf_a && mip_buf_a) {
                next.data = mip_buf_a;
            } else if (!use_buf_a && mip_buf_b) {
                next.data = mip_buf_b;
            } else {
                next.data = (uint8_t*)malloc(next.width * next.height * 4);
            }
            
            if (next.data) {
                aodds_reduce_half(&current, &next);
                current = next;
                memset(&next, 0, sizeof(next));
                use_buf_a = !use_buf_a;
            } else {
                success = 0;
            }
        }
    }
    
    /* Cleanup */
    free(compress_buffer);
    free(mip_buf_a);
    free(mip_buf_b);
    free(tile.data);
    fclose(fp);
    
    if (!success) {
        remove(temp_path);
        return 0;
    }
    
    /* Atomic rename */
#ifdef AOPIPELINE_WINDOWS
    remove(output_path);
#endif
    if (rename(temp_path, output_path) != 0) {
        remove(temp_path);
        return 0;
    }
    
    *bytes_written = total_written;
    return 1;
}

/*============================================================================
 * STREAMING TILE BUILDER IMPLEMENTATION
 *============================================================================*/

/**
 * Internal structure for streaming tile builder.
 */
struct aodds_builder_s {
    /* Configuration */
    aodds_builder_config_t config;
    int32_t chunk_count;  /* chunks_per_side^2 */
    
    /* Chunk storage - decoded images */
    aodecode_image_t* chunks;  /* Array of chunk_count images */
    uint8_t* chunk_status;     /* Array of aodds_chunk_status_t values */
    
    /* Deferred JPEG storage for parallel decode at finalize */
    uint8_t** jpeg_buffers;     /* Array of JPEG byte buffers (NULL if decoded or missing) */
    uint32_t* jpeg_sizes;       /* Array of JPEG sizes */
    uint32_t jpeg_storage_size; /* Allocated size of jpeg arrays */
    
    /* Thread safety */
#ifdef AOPIPELINE_WINDOWS
    CRITICAL_SECTION lock;
#else
    pthread_mutex_t lock;
#endif
    int lock_initialized;
    
    /* Tile composition buffer (allocated on first finalize) */
    aodecode_image_t tile_image;
    int tile_allocated;
    
    /* Compression work buffers (allocated on first finalize, reused) */
    uint8_t* compress_buffer;       /* BC1/BC3 compression output */
    uint32_t compress_buffer_size;  /* Current allocated size */
    
    /* Mipmap reduction buffers (ping-pong pattern) */
    uint8_t* mip_buf_a;             /* First mipmap buffer */
    uint8_t* mip_buf_b;             /* Second mipmap buffer */
    uint32_t mip_buf_a_size;        /* Allocated size of mip_buf_a */
    uint32_t mip_buf_b_size;        /* Allocated size of mip_buf_b */
    
    /* Decode pool reference */
    aodecode_pool_t* pool;
    
    /* Statistics */
    aodds_builder_status_t status;
};

/* Helper: Initialize lock */
static void builder_init_lock(aodds_builder_t* builder) {
    if (builder->lock_initialized) return;
#ifdef AOPIPELINE_WINDOWS
    InitializeCriticalSection(&builder->lock);
#else
    pthread_mutex_init(&builder->lock, NULL);
#endif
    builder->lock_initialized = 1;
}

/* Helper: Lock the builder */
static void builder_lock(aodds_builder_t* builder) {
#ifdef AOPIPELINE_WINDOWS
    EnterCriticalSection(&builder->lock);
#else
    pthread_mutex_lock(&builder->lock);
#endif
}

/* Helper: Unlock the builder */
static void builder_unlock(aodds_builder_t* builder) {
#ifdef AOPIPELINE_WINDOWS
    LeaveCriticalSection(&builder->lock);
#else
    pthread_mutex_unlock(&builder->lock);
#endif
}

/* Helper: Destroy lock */
static void builder_destroy_lock(aodds_builder_t* builder) {
    if (!builder->lock_initialized) return;
#ifdef AOPIPELINE_WINDOWS
    DeleteCriticalSection(&builder->lock);
#else
    pthread_mutex_destroy(&builder->lock);
#endif
    builder->lock_initialized = 0;
}

AODDS_API aodds_builder_t* aodds_builder_create(
    const aodds_builder_config_t* config,
    aodecode_pool_t* decode_pool
) {
    if (!config || config->chunks_per_side <= 0) {
        return NULL;
    }
    
    aodds_builder_t* builder = (aodds_builder_t*)calloc(1, sizeof(aodds_builder_t));
    if (!builder) {
        return NULL;
    }
    
    /* Copy configuration */
    builder->config = *config;
    builder->chunk_count = config->chunks_per_side * config->chunks_per_side;
    builder->pool = decode_pool;
    
    /* Allocate chunk storage */
    builder->chunks = (aodecode_image_t*)calloc(builder->chunk_count, sizeof(aodecode_image_t));
    builder->chunk_status = (uint8_t*)calloc(builder->chunk_count, sizeof(uint8_t));
    
    if (!builder->chunks || !builder->chunk_status) {
        free(builder->chunks);
        free(builder->chunk_status);
        free(builder);
        return NULL;
    }
    
    /* Allocate JPEG storage arrays for deferred decode */
    builder->jpeg_buffers = (uint8_t**)calloc(builder->chunk_count, sizeof(uint8_t*));
    builder->jpeg_sizes = (uint32_t*)calloc(builder->chunk_count, sizeof(uint32_t));
    builder->jpeg_storage_size = builder->chunk_count;
    
    if (!builder->jpeg_buffers || !builder->jpeg_sizes) {
        free(builder->jpeg_buffers);
        free(builder->jpeg_sizes);
        free(builder->chunks);
        free(builder->chunk_status);
        free(builder);
        return NULL;
    }
    
    /* Initialize lock */
    builder_init_lock(builder);
    
    /* Initialize work buffers (lazy allocation on first finalize) */
    builder->compress_buffer = NULL;
    builder->compress_buffer_size = 0;
    builder->mip_buf_a = NULL;
    builder->mip_buf_b = NULL;
    builder->mip_buf_a_size = 0;
    builder->mip_buf_b_size = 0;
    
    /* Initialize status */
    builder->status.chunks_total = builder->chunk_count;
    builder->status.chunks_received = 0;
    builder->status.chunks_decoded = 0;
    builder->status.chunks_failed = 0;
    builder->status.chunks_fallback = 0;
    builder->status.chunks_missing = 0;
    
    return builder;
}

AODDS_API void aodds_builder_reset(
    aodds_builder_t* builder,
    const aodds_builder_config_t* config
) {
    if (!builder || !config) {
        return;
    }
    
    builder_lock(builder);
    
    int32_t new_chunk_count = config->chunks_per_side * config->chunks_per_side;
    int32_t new_tile_size = config->chunks_per_side * CHUNK_SIZE;
    int32_t old_tile_size = builder->config.chunks_per_side * CHUNK_SIZE;
    
    /* Free existing chunk data and any pending JPEG buffers */
    for (int32_t i = 0; i < builder->chunk_count; i++) {
        if (builder->chunks[i].data) {
            aodecode_free_image(&builder->chunks[i], builder->pool);
        }
        /* Free any deferred JPEG data */
        if (builder->jpeg_buffers && builder->jpeg_buffers[i]) {
            free(builder->jpeg_buffers[i]);
            builder->jpeg_buffers[i] = NULL;
        }
        if (builder->jpeg_sizes) {
            builder->jpeg_sizes[i] = 0;
        }
    }
    
    /* Free tile image if tile size changed */
    if (builder->tile_allocated && new_tile_size != old_tile_size) {
        if (builder->tile_image.data) {
            free(builder->tile_image.data);
            builder->tile_image.data = NULL;
        }
        builder->tile_allocated = 0;
        memset(&builder->tile_image, 0, sizeof(builder->tile_image));
    }
    
    /* Free work buffers if tile size increased (they'll be reallocated in finalize)
     * If tile size decreased or unchanged, keep existing buffers for reuse */
    if (new_tile_size > old_tile_size) {
        free(builder->compress_buffer);
        builder->compress_buffer = NULL;
        builder->compress_buffer_size = 0;
        
        free(builder->mip_buf_a);
        free(builder->mip_buf_b);
        builder->mip_buf_a = NULL;
        builder->mip_buf_b = NULL;
        builder->mip_buf_a_size = 0;
        builder->mip_buf_b_size = 0;
    }
    
    /* Reallocate if chunk count changed */
    if (new_chunk_count != builder->chunk_count) {
        free(builder->chunks);
        free(builder->chunk_status);
        free(builder->jpeg_buffers);
        free(builder->jpeg_sizes);
        
        builder->chunks = (aodecode_image_t*)calloc(new_chunk_count, sizeof(aodecode_image_t));
        builder->chunk_status = (uint8_t*)calloc(new_chunk_count, sizeof(uint8_t));
        builder->jpeg_buffers = (uint8_t**)calloc(new_chunk_count, sizeof(uint8_t*));
        builder->jpeg_sizes = (uint32_t*)calloc(new_chunk_count, sizeof(uint32_t));
        builder->jpeg_storage_size = new_chunk_count;
        
        if (!builder->chunks || !builder->chunk_status || 
            !builder->jpeg_buffers || !builder->jpeg_sizes) {
            /* Allocation failed - leave builder in invalid state */
            builder->chunk_count = 0;
            builder_unlock(builder);
            return;
        }
    } else {
        /* Just clear existing arrays */
        memset(builder->chunks, 0, new_chunk_count * sizeof(aodecode_image_t));
        memset(builder->chunk_status, 0, new_chunk_count * sizeof(uint8_t));
        memset(builder->jpeg_buffers, 0, new_chunk_count * sizeof(uint8_t*));
        memset(builder->jpeg_sizes, 0, new_chunk_count * sizeof(uint32_t));
    }
    
    /* Update configuration */
    builder->config = *config;
    builder->chunk_count = new_chunk_count;
    
    /* Reset status */
    builder->status.chunks_total = new_chunk_count;
    builder->status.chunks_received = 0;
    builder->status.chunks_decoded = 0;
    builder->status.chunks_failed = 0;
    builder->status.chunks_fallback = 0;
    builder->status.chunks_missing = 0;
    
    builder_unlock(builder);
}

AODDS_API void aodds_builder_destroy(aodds_builder_t* builder) {
    if (!builder) {
        return;
    }
    
    /* Free chunk data and any pending JPEG buffers */
    for (int32_t i = 0; i < builder->chunk_count; i++) {
        if (builder->chunks[i].data) {
            aodecode_free_image(&builder->chunks[i], builder->pool);
        }
        if (builder->jpeg_buffers && builder->jpeg_buffers[i]) {
            free(builder->jpeg_buffers[i]);
        }
    }
    
    free(builder->chunks);
    free(builder->chunk_status);
    free(builder->jpeg_buffers);
    free(builder->jpeg_sizes);
    
    /* Free tile image if allocated */
    if (builder->tile_allocated && builder->tile_image.data) {
        free(builder->tile_image.data);
    }
    
    /* Free work buffers */
    free(builder->compress_buffer);
    free(builder->mip_buf_a);
    free(builder->mip_buf_b);
    
    /* Destroy lock */
    builder_destroy_lock(builder);
    
    free(builder);
}

AODDS_API int32_t aodds_builder_add_chunk(
    aodds_builder_t* builder,
    int32_t chunk_index,
    const uint8_t* jpeg_data,
    uint32_t jpeg_size
) {
    if (!builder || !jpeg_data || jpeg_size == 0) {
        return 0;
    }
    
    if (chunk_index < 0 || chunk_index >= builder->chunk_count) {
        return 0;
    }
    
    builder_lock(builder);
    
    /* Check if already set */
    if (builder->chunk_status[chunk_index] != CHUNK_STATUS_EMPTY) {
        builder_unlock(builder);
        return 0;
    }
    
    /* Allocate JPEG buffer and copy data for deferred decode */
    uint8_t* jpeg_copy = (uint8_t*)malloc(jpeg_size);
    if (!jpeg_copy) {
        builder_unlock(builder);
        return 0;
    }
    memcpy(jpeg_copy, jpeg_data, jpeg_size);
    
    /* Store for parallel decode at finalize time */
    builder->jpeg_buffers[chunk_index] = jpeg_copy;
    builder->jpeg_sizes[chunk_index] = jpeg_size;
    builder->chunk_status[chunk_index] = CHUNK_STATUS_PENDING_DECODE;
    builder->status.chunks_received++;
    
    builder_unlock(builder);
    return 1;
}

AODDS_API int32_t aodds_builder_add_chunks_batch(
    aodds_builder_t* builder,
    int32_t count,
    const int32_t* indices,
    const uint8_t** jpeg_data,
    const uint32_t* jpeg_sizes
) {
    if (!builder || !indices || !jpeg_data || !jpeg_sizes || count <= 0) {
        return 0;
    }
    
    int32_t added = 0;
    
    builder_lock(builder);
    
    for (int32_t i = 0; i < count; i++) {
        int32_t idx = indices[i];
        
        /* Validate index */
        if (idx < 0 || idx >= builder->chunk_count) {
            continue;
        }
        
        /* Skip if already set */
        if (builder->chunk_status[idx] != CHUNK_STATUS_EMPTY) {
            continue;
        }
        
        /* Skip empty data */
        if (!jpeg_data[i] || jpeg_sizes[i] == 0) {
            continue;
        }
        
        /* Allocate and copy JPEG for deferred decode */
        uint8_t* jpeg_copy = (uint8_t*)malloc(jpeg_sizes[i]);
        if (!jpeg_copy) {
            continue;
        }
        memcpy(jpeg_copy, jpeg_data[i], jpeg_sizes[i]);
        
        /* Store for parallel decode at finalize */
        builder->jpeg_buffers[idx] = jpeg_copy;
        builder->jpeg_sizes[idx] = jpeg_sizes[i];
        builder->chunk_status[idx] = CHUNK_STATUS_PENDING_DECODE;
        builder->status.chunks_received++;
        added++;
    }
    
    builder_unlock(builder);
    return added;
}

AODDS_API int32_t aodds_builder_add_fallback_image(
    aodds_builder_t* builder,
    int32_t chunk_index,
    const uint8_t* rgba_data,
    int32_t width,
    int32_t height
) {
    if (!builder || !rgba_data) {
        return 0;
    }
    
    if (chunk_index < 0 || chunk_index >= builder->chunk_count) {
        return 0;
    }
    
    /* Validate dimensions */
    if (width != CHUNK_WIDTH || height != CHUNK_HEIGHT) {
        return 0;
    }
    
    builder_lock(builder);
    
    /* Check if already set */
    if (builder->chunk_status[chunk_index] != CHUNK_STATUS_EMPTY) {
        builder_unlock(builder);
        return 0;
    }
    
    /* Allocate and copy image data */
    uint8_t* data = NULL;
    if (builder->pool) {
        data = aodecode_acquire_buffer(builder->pool);
    }
    if (!data) {
        data = (uint8_t*)malloc(CHUNK_BUFFER_SIZE);
    }
    
    if (!data) {
        builder_unlock(builder);
        return 0;
    }
    
    memcpy(data, rgba_data, CHUNK_BUFFER_SIZE);
    
    builder->chunks[chunk_index].data = data;
    builder->chunks[chunk_index].width = width;
    builder->chunks[chunk_index].height = height;
    builder->chunks[chunk_index].stride = width * 4;
    builder->chunks[chunk_index].channels = 4;
    builder->chunks[chunk_index].from_pool = (builder->pool != NULL);
    
    builder->chunk_status[chunk_index] = CHUNK_STATUS_FALLBACK;
    builder->status.chunks_received++;
    builder->status.chunks_fallback++;
    
    builder_unlock(builder);
    return 1;
}

AODDS_API void aodds_builder_mark_missing(
    aodds_builder_t* builder,
    int32_t chunk_index
) {
    if (!builder) {
        return;
    }
    
    if (chunk_index < 0 || chunk_index >= builder->chunk_count) {
        return;
    }
    
    builder_lock(builder);
    
    /* Only mark if not already set */
    if (builder->chunk_status[chunk_index] == CHUNK_STATUS_EMPTY) {
        builder->chunk_status[chunk_index] = CHUNK_STATUS_MISSING;
        builder->status.chunks_received++;
        builder->status.chunks_missing++;
    }
    
    builder_unlock(builder);
}

AODDS_API void aodds_builder_get_status(
    const aodds_builder_t* builder,
    aodds_builder_status_t* status
) {
    if (!builder || !status) {
        return;
    }
    
    /* Cast away const for locking (status read is thread-safe) */
    aodds_builder_t* b = (aodds_builder_t*)builder;
    builder_lock(b);
    *status = builder->status;
    builder_unlock(b);
}

AODDS_API int32_t aodds_builder_is_complete(const aodds_builder_t* builder) {
    if (!builder) {
        return 0;
    }
    
    aodds_builder_t* b = (aodds_builder_t*)builder;
    builder_lock(b);
    int32_t complete = (builder->status.chunks_received >= builder->status.chunks_total);
    builder_unlock(b);
    return complete;
}

AODDS_API int32_t aodds_builder_finalize(
    aodds_builder_t* builder,
    uint8_t* dds_output,
    uint32_t output_size,
    uint32_t* bytes_written
) {
    if (!builder || !dds_output || !bytes_written) {
        return 0;
    }
    
    int32_t chunks_per_side = builder->config.chunks_per_side;
    int32_t tile_size = chunks_per_side * CHUNK_SIZE;
    int32_t mipmap_count = aodds_calc_mipmap_count(tile_size, tile_size);
    
    uint32_t required_size = aodds_calc_dds_size(tile_size, tile_size, mipmap_count, builder->config.format);
    if (output_size < required_size) {
        return 0;
    }
    
    /* Allocate tile image if not already */
    if (!builder->tile_allocated) {
        builder->tile_image.width = tile_size;
        builder->tile_image.height = tile_size;
        builder->tile_image.stride = tile_size * 4;
        builder->tile_image.channels = 4;
        builder->tile_image.data = (uint8_t*)malloc(tile_size * tile_size * 4);
        if (!builder->tile_image.data) {
            return 0;
        }
        builder->tile_allocated = 1;
    }
    
    /* ═══════════════════════════════════════════════════════════════════════
     * PARALLEL DECODE: Decode all pending JPEGs at once using OpenMP
     * This is the key performance optimization - 8 threads decode ~32 chunks each
     * instead of 256 sequential decodes during add_chunk() calls.
     * ═══════════════════════════════════════════════════════════════════════ */
    int32_t pending_count = 0;
    int32_t decode_success_count = 0;
    int32_t decode_fail_count = 0;
    
    /* Count pending chunks first */
    for (int32_t i = 0; i < builder->chunk_count; i++) {
        if (builder->chunk_status[i] == CHUNK_STATUS_PENDING_DECODE) {
            pending_count++;
        }
    }
    
    if (pending_count > 0) {
        /* Parallel decode all pending JPEGs */
        #pragma omp parallel for schedule(dynamic, 4) reduction(+:decode_success_count, decode_fail_count)
        for (int32_t i = 0; i < builder->chunk_count; i++) {
            if (builder->chunk_status[i] == CHUNK_STATUS_PENDING_DECODE) {
                aodecode_image_t decoded = {0};
                int32_t success = aodecode_single(
                    builder->jpeg_buffers[i],
                    builder->jpeg_sizes[i],
                    &decoded,
                    builder->pool  /* Pool is thread-safe */
                );
                
                if (success && decoded.data) {
                    builder->chunks[i] = decoded;
                    builder->chunk_status[i] = CHUNK_STATUS_JPEG;
                    decode_success_count++;
                } else {
                    builder->chunk_status[i] = CHUNK_STATUS_MISSING;
                    decode_fail_count++;
                }
                
                /* Free JPEG buffer after decode - no longer needed */
                free(builder->jpeg_buffers[i]);
                builder->jpeg_buffers[i] = NULL;
                builder->jpeg_sizes[i] = 0;
            }
        }
        
        /* Update stats after parallel region */
        builder->status.chunks_decoded += decode_success_count;
        builder->status.chunks_failed += decode_fail_count;
        builder->status.chunks_missing += decode_fail_count;
    }
    
    /* Fill missing chunks with color and compose */
    aodds_fill_missing(builder->chunks, chunks_per_side, &builder->tile_image,
                       builder->config.missing_r, builder->config.missing_g, builder->config.missing_b);
    aodds_compose_chunks(builder->chunks, chunks_per_side, &builder->tile_image);
    
    /* Write DDS header */
    int32_t header_written = aodds_write_header(dds_output, tile_size, tile_size, 
                                                 mipmap_count, builder->config.format);
    if (header_written != DDS_HEADER_SIZE) {
        return 0;
    }
    
    uint32_t total_written = DDS_HEADER_SIZE;
    uint8_t* output_ptr = dds_output + DDS_HEADER_SIZE;
    
    /* Block size for compression size calculations */
    uint32_t block_size = (builder->config.format == DDS_FORMAT_BC1) ? 8 : 16;
    
    /* Ensure mipmap buffers are allocated and large enough (persistent in builder) */
    int32_t mip1_size = (tile_size / 2) * (tile_size / 2) * 4;
    int32_t mip2_size = (tile_size / 4) * (tile_size / 4) * 4;
    
    if (mipmap_count > 1) {
        if (builder->mip_buf_a_size < (uint32_t)mip1_size) {
            free(builder->mip_buf_a);
            builder->mip_buf_a = (uint8_t*)malloc(mip1_size);
            builder->mip_buf_a_size = builder->mip_buf_a ? mip1_size : 0;
        }
    }
    if (mipmap_count > 2) {
        if (builder->mip_buf_b_size < (uint32_t)mip2_size) {
            free(builder->mip_buf_b);
            builder->mip_buf_b = (uint8_t*)malloc(mip2_size);
            builder->mip_buf_b_size = builder->mip_buf_b ? mip2_size : 0;
        }
    }
    
    /* Local pointers for compatibility with existing code flow */
    uint8_t* mip_buf_a = builder->mip_buf_a;
    uint8_t* mip_buf_b = builder->mip_buf_b;
    
    aodecode_image_t current = builder->tile_image;
    aodecode_image_t next = {0};
    int success = 1;
    int use_buf_a = 1;
    
    for (int32_t mip = 0; mip < mipmap_count && success; mip++) {
        /* Calculate expected compressed size before compression */
        uint32_t blocks_x = (current.width + 3) / 4;
        uint32_t blocks_y = (current.height + 3) / 4;
        uint32_t expected_size = blocks_x * blocks_y * block_size;
        
        if (total_written + expected_size > output_size) {
            success = 0;
            break;
        }
        
        /* Compress directly to output buffer - no temp buffer needed */
        uint32_t compressed_size = aodds_compress(&current, builder->config.format, output_ptr);
        output_ptr += compressed_size;
        total_written += compressed_size;
        
        if (mip < mipmap_count - 1 && current.width > 4) {
            next.width = current.width / 2;
            next.height = current.height / 2;
            next.stride = next.width * 4;
            next.channels = 4;
            
            if (use_buf_a && mip_buf_a) {
                next.data = mip_buf_a;
            } else if (!use_buf_a && mip_buf_b) {
                next.data = mip_buf_b;
            } else {
                next.data = (uint8_t*)malloc(next.width * next.height * 4);
            }
            
            if (next.data) {
                aodds_reduce_half(&current, &next);
                current = next;
                memset(&next, 0, sizeof(next));
                use_buf_a = !use_buf_a;
            } else {
                success = 0;
            }
        }
    }
    
    /* No cleanup needed - buffers are persistent in builder struct */
    
    if (success) {
        *bytes_written = total_written;
    }
    
    return success;
}

AODDS_API int32_t aodds_builder_finalize_to_file(
    aodds_builder_t* builder,
    const char* output_path,
    uint32_t* bytes_written
) {
    if (!builder || !output_path || !bytes_written) {
        return 0;
    }
    
    int32_t chunks_per_side = builder->config.chunks_per_side;
    int32_t tile_size = chunks_per_side * CHUNK_SIZE;
    int32_t mipmap_count = aodds_calc_mipmap_count(tile_size, tile_size);
    
    /* Allocate tile image if not already */
    if (!builder->tile_allocated) {
        builder->tile_image.width = tile_size;
        builder->tile_image.height = tile_size;
        builder->tile_image.stride = tile_size * 4;
        builder->tile_image.channels = 4;
        builder->tile_image.data = (uint8_t*)malloc(tile_size * tile_size * 4);
        if (!builder->tile_image.data) {
            return 0;
        }
        builder->tile_allocated = 1;
    }
    
    /* ═══════════════════════════════════════════════════════════════════════
     * PARALLEL DECODE: Decode all pending JPEGs at once using OpenMP
     * ═══════════════════════════════════════════════════════════════════════ */
    int32_t pending_count = 0;
    int32_t decode_success_count = 0;
    int32_t decode_fail_count = 0;
    
    for (int32_t i = 0; i < builder->chunk_count; i++) {
        if (builder->chunk_status[i] == CHUNK_STATUS_PENDING_DECODE) {
            pending_count++;
        }
    }
    
    if (pending_count > 0) {
        #pragma omp parallel for schedule(dynamic, 4) reduction(+:decode_success_count, decode_fail_count)
        for (int32_t i = 0; i < builder->chunk_count; i++) {
            if (builder->chunk_status[i] == CHUNK_STATUS_PENDING_DECODE) {
                aodecode_image_t decoded = {0};
                int32_t success = aodecode_single(
                    builder->jpeg_buffers[i],
                    builder->jpeg_sizes[i],
                    &decoded,
                    builder->pool
                );
                
                if (success && decoded.data) {
                    builder->chunks[i] = decoded;
                    builder->chunk_status[i] = CHUNK_STATUS_JPEG;
                    decode_success_count++;
                } else {
                    builder->chunk_status[i] = CHUNK_STATUS_MISSING;
                    decode_fail_count++;
                }
                
                free(builder->jpeg_buffers[i]);
                builder->jpeg_buffers[i] = NULL;
                builder->jpeg_sizes[i] = 0;
            }
        }
        
        builder->status.chunks_decoded += decode_success_count;
        builder->status.chunks_failed += decode_fail_count;
        builder->status.chunks_missing += decode_fail_count;
    }
    
    /* Fill missing chunks with color and compose */
    aodds_fill_missing(builder->chunks, chunks_per_side, &builder->tile_image,
                       builder->config.missing_r, builder->config.missing_g, builder->config.missing_b);
    aodds_compose_chunks(builder->chunks, chunks_per_side, &builder->tile_image);
    
    /* Create temp file path */
    char temp_path[4096];
    snprintf(temp_path, sizeof(temp_path), "%s.tmp", output_path);
    
    /* Open temp file */
    FILE* fp = fopen(temp_path, "wb");
    if (!fp) {
        return 0;
    }
    
    /* Write DDS header */
    uint8_t header[DDS_HEADER_SIZE];
    aodds_write_header(header, tile_size, tile_size, mipmap_count, builder->config.format);
    
    if (fwrite(header, 1, DDS_HEADER_SIZE, fp) != DDS_HEADER_SIZE) {
        fclose(fp);
        remove(temp_path);
        return 0;
    }
    
    uint32_t total_written = DDS_HEADER_SIZE;
    
    /* Block size for compression size calculations */
    uint32_t block_size = (builder->config.format == DDS_FORMAT_BC1) ? 8 : 16;
    uint32_t max_blocks_x = (tile_size + 3) / 4;
    uint32_t max_blocks_y = (tile_size + 3) / 4;
    uint32_t max_compressed_size = max_blocks_x * max_blocks_y * block_size;
    
    /* Ensure compression buffer is allocated and large enough (still needed for file writes) */
    if (builder->compress_buffer_size < max_compressed_size) {
        free(builder->compress_buffer);
        builder->compress_buffer = (uint8_t*)malloc(max_compressed_size);
        builder->compress_buffer_size = builder->compress_buffer ? max_compressed_size : 0;
    }
    
    if (!builder->compress_buffer) {
        fclose(fp);
        remove(temp_path);
        return 0;
    }
    
    uint8_t* compress_buffer = builder->compress_buffer;
    
    /* Ensure mipmap buffers are allocated and large enough (persistent in builder) */
    int32_t mip1_size = (tile_size / 2) * (tile_size / 2) * 4;
    int32_t mip2_size = (tile_size / 4) * (tile_size / 4) * 4;
    
    if (mipmap_count > 1) {
        if (builder->mip_buf_a_size < (uint32_t)mip1_size) {
            free(builder->mip_buf_a);
            builder->mip_buf_a = (uint8_t*)malloc(mip1_size);
            builder->mip_buf_a_size = builder->mip_buf_a ? mip1_size : 0;
        }
    }
    if (mipmap_count > 2) {
        if (builder->mip_buf_b_size < (uint32_t)mip2_size) {
            free(builder->mip_buf_b);
            builder->mip_buf_b = (uint8_t*)malloc(mip2_size);
            builder->mip_buf_b_size = builder->mip_buf_b ? mip2_size : 0;
        }
    }
    
    /* Local pointers for compatibility with existing code flow */
    uint8_t* mip_buf_a = builder->mip_buf_a;
    uint8_t* mip_buf_b = builder->mip_buf_b;
    
    aodecode_image_t current = builder->tile_image;
    aodecode_image_t next = {0};
    int success = 1;
    int use_buf_a = 1;
    
    for (int32_t mip = 0; mip < mipmap_count && success; mip++) {
        uint32_t compressed_size = aodds_compress(&current, builder->config.format, compress_buffer);
        
        if (fwrite(compress_buffer, 1, compressed_size, fp) != compressed_size) {
            success = 0;
            break;
        }
        total_written += compressed_size;
        
        if (mip < mipmap_count - 1 && current.width > 4) {
            next.width = current.width / 2;
            next.height = current.height / 2;
            next.stride = next.width * 4;
            next.channels = 4;
            
            if (use_buf_a && mip_buf_a) {
                next.data = mip_buf_a;
            } else if (!use_buf_a && mip_buf_b) {
                next.data = mip_buf_b;
            } else {
                next.data = (uint8_t*)malloc(next.width * next.height * 4);
            }
            
            if (next.data) {
                aodds_reduce_half(&current, &next);
                current = next;
                memset(&next, 0, sizeof(next));
                use_buf_a = !use_buf_a;
            } else {
                success = 0;
            }
        }
    }
    
    /* No cleanup needed - buffers are persistent in builder struct */
    fclose(fp);
    
    if (!success) {
        remove(temp_path);
        return 0;
    }
    
    /* Atomic rename */
#ifdef AOPIPELINE_WINDOWS
    remove(output_path);
#endif
    if (rename(temp_path, output_path) != 0) {
        remove(temp_path);
        return 0;
    }
    
    *bytes_written = total_written;
    return 1;
}

AODDS_API const char* aodds_version(void) {
    aodds_init_ispc();
    
    static char version_buf[256];
    const char* compressor_status;
    if (!ispc_available) {
        compressor_status = "STB (ISPC unavailable)";
    } else if (force_fallback) {
        compressor_status = "STB (forced)";
    } else {
        compressor_status = "ISPC";
    }
    
    snprintf(version_buf, sizeof(version_buf),
             "aodds " AODDS_VERSION
#if AOPIPELINE_HAS_OPENMP
             " (OpenMP enabled)"
#else
             " (OpenMP disabled)"
#endif
             " [%s]"
#ifdef AOPIPELINE_WINDOWS
             " [Windows]"
#elif defined(AOPIPELINE_MACOS)
             " [macOS]"
#else
             " [Linux]"
#endif
             , compressor_status
    );
    
    return version_buf;
}

