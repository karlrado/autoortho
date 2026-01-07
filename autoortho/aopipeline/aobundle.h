/**
 * aobundle.h - Cache Bundle Format for AutoOrtho
 * 
 * Consolidates 256 individual JPEG cache files into a single bundle file
 * for massively reduced I/O overhead.
 * 
 * Bundle Format:
 * ┌──────────────────────────────────────────────────────────────┐
 * │ Header (32 bytes)                                            │
 * ├──────────────────────────────────────────────────────────────┤
 * │  magic[4]        = "AOB1" (0x41 0x4F 0x42 0x31)              │
 * │  version[2]      = 0x0001                                    │
 * │  chunk_count[2]  = number of chunks (e.g., 256)              │
 * │  tile_col[4]     = tile column coordinate                    │
 * │  tile_row[4]     = tile row coordinate                       │
 * │  zoom[2]         = zoom level                                │
 * │  maptype_len[2]  = maptype string length                     │
 * │  reserved[12]    = future use                                │
 * ├──────────────────────────────────────────────────────────────┤
 * │ Maptype String (maptype_len bytes, padded to 8-byte align)   │
 * ├──────────────────────────────────────────────────────────────┤
 * │ Index Table (8 bytes per chunk × chunk_count)                │
 * │  offset[4]       = offset from start of data section         │
 * │  size[4]         = JPEG data size (0 = missing)              │
 * ├──────────────────────────────────────────────────────────────┤
 * │ Data Section (variable, concatenated JPEGs)                  │
 * │  [jpeg_0][jpeg_1]...[jpeg_N]                                 │
 * └──────────────────────────────────────────────────────────────┘
 * 
 * Benefits:
 * - Single file open instead of 256
 * - Sequential read for optimal disk I/O
 * - Memory-mappable for zero-copy access
 * - Atomic writes (temp file + rename)
 * - Fast index lookup (O(1) for any chunk)
 */

#ifndef AOBUNDLE_H
#define AOBUNDLE_H

#include <stdint.h>
#include <stddef.h>  /* For size_t */

#ifdef __cplusplus
extern "C" {
#endif

/* Export/Import macros */
#ifdef _WIN32
  #ifdef AOPIPELINE_EXPORTS
    #define AOBUNDLE_API __declspec(dllexport)
  #else
    #define AOBUNDLE_API __declspec(dllimport)
  #endif
#else
  #define AOBUNDLE_API __attribute__((__visibility__("default")))
#endif

/* Magic number "AOB1" */
#define AOBUNDLE_MAGIC 0x31424F41

/* Current version */
#define AOBUNDLE_VERSION 1

/* Maximum chunks per bundle */
#define AOBUNDLE_MAX_CHUNKS 4096

/* Maximum maptype length */
#define AOBUNDLE_MAX_MAPTYPE 64

/**
 * Bundle header structure (32 bytes).
 */
typedef struct {
    uint32_t magic;           /* Must be AOBUNDLE_MAGIC */
    uint16_t version;         /* Format version */
    uint16_t chunk_count;     /* Number of chunks in bundle */
    int32_t  tile_col;        /* Tile column coordinate */
    int32_t  tile_row;        /* Tile row coordinate */
    uint16_t zoom;            /* Zoom level */
    uint16_t maptype_len;     /* Length of maptype string */
    uint8_t  reserved[12];    /* Future use, must be zero */
} aobundle_header_t;

/**
 * Index entry (8 bytes per chunk).
 */
typedef struct {
    uint32_t offset;          /* Offset from start of data section */
    uint32_t size;            /* JPEG data size (0 = missing) */
} aobundle_index_t;

/**
 * In-memory bundle representation.
 */
typedef struct {
    aobundle_header_t header;
    char maptype[AOBUNDLE_MAX_MAPTYPE];
    aobundle_index_t* index;   /* Array of chunk_count entries */
    uint8_t* data;             /* Pointer to data section */
    uint32_t data_size;        /* Total size of data section */
    
    /* Memory mapping (if used) */
    void* mmap_base;           /* Base address of mmap (NULL if not mapped) */
    size_t mmap_size;          /* Size of mmap region */
} aobundle_t;

/**
 * Create a new bundle from individual JPEG files.
 * 
 * @param cache_dir     Directory containing cached JPEGs
 * @param tile_col      Tile column coordinate
 * @param tile_row      Tile row coordinate
 * @param maptype       Map source identifier
 * @param zoom          Zoom level
 * @param chunks_per_side Number of chunks per side (e.g., 16)
 * @param output_path   Path for output bundle file
 * 
 * @return 1 on success, 0 on failure
 * 
 * This function:
 * 1. Reads all JPEG files from cache_dir
 * 2. Creates bundle in memory
 * 3. Writes atomically (temp file + rename)
 */
AOBUNDLE_API int32_t aobundle_create(
    const char* cache_dir,
    int32_t tile_col,
    int32_t tile_row,
    const char* maptype,
    int32_t zoom,
    int32_t chunks_per_side,
    const char* output_path
);

/**
 * Open an existing bundle file.
 * 
 * @param path      Path to bundle file
 * @param bundle    Output bundle structure (caller allocates)
 * @param use_mmap  If true, memory-map the file; else read into malloc'd buffer
 * 
 * @return 1 on success, 0 on failure
 */
AOBUNDLE_API int32_t aobundle_open(
    const char* path,
    aobundle_t* bundle,
    int32_t use_mmap
);

/**
 * Close a bundle and free resources.
 * 
 * @param bundle    Bundle to close
 */
AOBUNDLE_API void aobundle_close(aobundle_t* bundle);

/**
 * Get JPEG data for a specific chunk.
 * 
 * @param bundle    Open bundle
 * @param index     Chunk index (0 to chunk_count-1)
 * @param data      Output pointer to JPEG data (points into bundle, do not free)
 * @param size      Output size of JPEG data
 * 
 * @return 1 if chunk exists, 0 if missing
 */
AOBUNDLE_API int32_t aobundle_get_chunk(
    const aobundle_t* bundle,
    int32_t index,
    const uint8_t** data,
    uint32_t* size
);

/**
 * Get all JPEG data pointers for building a tile.
 * 
 * This is the fast path for tile building - returns arrays that can be
 * passed directly to aodds_build_from_jpegs().
 * 
 * @param bundle        Open bundle
 * @param jpeg_data     Output array of JPEG pointers (caller allocates)
 * @param jpeg_sizes    Output array of JPEG sizes (caller allocates)
 * @param max_chunks    Size of output arrays
 * 
 * @return Number of valid (non-missing) chunks
 */
AOBUNDLE_API int32_t aobundle_get_all_chunks(
    const aobundle_t* bundle,
    const uint8_t** jpeg_data,
    uint32_t* jpeg_sizes,
    int32_t max_chunks
);

/**
 * Build DDS directly from bundle (optimal single-call path).
 * 
 * Combines bundle reading + JPEG decode + DDS build in one native call.
 * 
 * @param bundle_path   Path to bundle file
 * @param format        DDS format (0=BC1, 1=BC3)
 * @param missing_color RGB fill color for missing chunks
 * @param dds_output    Pre-allocated output buffer
 * @param output_size   Size of output buffer
 * @param bytes_written Actual bytes written (output)
 * 
 * @return 1 on success, 0 on failure
 */
AOBUNDLE_API int32_t aobundle_build_dds(
    const char* bundle_path,
    int32_t format,
    uint8_t missing_color[3],
    uint8_t* dds_output,
    uint32_t output_size,
    uint32_t* bytes_written
);

/**
 * Get version information.
 */
AOBUNDLE_API const char* aobundle_version(void);

#ifdef __cplusplus
}
#endif

#endif /* AOBUNDLE_H */

