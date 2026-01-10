/**
 * aobundle2.h - Multi-Zoom Mutable Cache Bundle Format for AutoOrtho
 * 
 * Version 2 of the cache bundle format with support for:
 * - Multiple zoom levels in a single bundle
 * - Mutable data sections (append-only with garbage tracking)
 * - In-place chunk patching and zoom expansion
 * - Efficient compaction when fragmentation exceeds threshold
 * 
 * Bundle Format (AOB2):
 * ┌──────────────────────────────────────────────────────────────────────────┐
 * │ Header (64 bytes)                                                         │
 * ├──────────────────────────────────────────────────────────────────────────┤
 * │  magic[4]            = "AOB2" (0x41 0x4F 0x42 0x32)                       │
 * │  version[2]          = 0x0002                                             │
 * │  flags[2]            = MUTABLE | MULTI_ZOOM | COMPACTION_NEEDED          │
 * │  tile_row[4]         = tile row coordinate                               │
 * │  tile_col[4]         = tile column coordinate                            │
 * │  maptype_len[2]      = maptype string length                             │
 * │  zoom_count[2]       = number of zoom levels stored                      │
 * │  min_zoom[2]         = minimum zoom level                                │
 * │  max_zoom[2]         = maximum zoom level                                │
 * │  total_chunks[4]     = total chunks across all zoom levels               │
 * │  data_section_offset[4] = offset to data section from file start         │
 * │  garbage_bytes[4]    = bytes marked as garbage (for compaction)          │
 * │  last_modified[8]    = Unix timestamp of last modification               │
 * │  checksum[4]         = CRC32 of header (excluding checksum field)        │
 * │  reserved[12]        = future use, must be zero                          │
 * ├──────────────────────────────────────────────────────────────────────────┤
 * │ Maptype String (padded to 8-byte alignment)                              │
 * ├──────────────────────────────────────────────────────────────────────────┤
 * │ Zoom Level Table (12 bytes × zoom_count)                                 │
 * │  zoom_level[2]       = zoom level value                                  │
 * │  chunks_per_side[2]  = chunks per side (e.g., 16 for 256 chunks)         │
 * │  index_offset[4]     = offset to chunk index for this zoom               │
 * │  chunk_count[4]      = total chunks at this zoom level                   │
 * ├──────────────────────────────────────────────────────────────────────────┤
 * │ Chunk Indices (16 bytes × total_chunks, one section per zoom)            │
 * │  data_offset[4]      = offset from data_section_offset                   │
 * │  size[4]             = JPEG data size (0 = not stored)                   │
 * │  flags[2]            = VALID | MISSING | PLACEHOLDER | UPSCALED | GARBAGE│
 * │  quality[2]          = quality indicator (0-100)                         │
 * │  timestamp[4]        = Unix timestamp when chunk was stored              │
 * ├──────────────────────────────────────────────────────────────────────────┤
 * │ Data Section (variable, append-only)                                     │
 * │  [jpeg_data...] (concatenated, may have gaps from mutations)             │
 * └──────────────────────────────────────────────────────────────────────────┘
 * 
 * Benefits over AOB1:
 * - Multi-zoom: Single file can contain ZL14, ZL15, ZL16, etc.
 * - Mutable: Chunks can be added/replaced without full rewrite
 * - Trackable: Timestamps and quality for cache management
 * - Compactable: Garbage tracking enables efficient defragmentation
 */

#ifndef AOBUNDLE2_H
#define AOBUNDLE2_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Export/Import macros */
#ifdef _WIN32
  #ifdef AOPIPELINE_EXPORTS
    #define AOBUNDLE2_API __declspec(dllexport)
  #else
    #define AOBUNDLE2_API __declspec(dllimport)
  #endif
#else
  #define AOBUNDLE2_API __attribute__((__visibility__("default")))
#endif

/* Magic number "AOB2" (little-endian) */
#define AOBUNDLE2_MAGIC 0x32424F41

/* Current version */
#define AOBUNDLE2_VERSION 2

/* Limits */
#define AOBUNDLE2_MAX_CHUNKS_PER_ZOOM 4096
#define AOBUNDLE2_MAX_ZOOM_LEVELS 8
#define AOBUNDLE2_MAX_MAPTYPE 64
#define AOBUNDLE2_MAX_TOTAL_CHUNKS (AOBUNDLE2_MAX_CHUNKS_PER_ZOOM * AOBUNDLE2_MAX_ZOOM_LEVELS)

/* Header size for alignment calculations */
#define AOBUNDLE2_HEADER_SIZE 64

/* Compaction threshold (fraction of garbage to total data) */
#define AOBUNDLE2_COMPACTION_THRESHOLD 0.30f

/**
 * Bundle flags (stored in header.flags)
 */
typedef enum {
    AOBUNDLE2_FLAG_NONE             = 0x0000,
    AOBUNDLE2_FLAG_MUTABLE          = 0x0001,  /* Bundle supports mutations */
    AOBUNDLE2_FLAG_MULTI_ZOOM       = 0x0002,  /* Bundle has multiple zoom levels */
    AOBUNDLE2_FLAG_COMPACTION_NEEDED= 0x0004,  /* Garbage exceeds threshold */
    AOBUNDLE2_FLAG_LOCKED           = 0x0008,  /* Currently being written */
} aobundle2_flags_t;

/**
 * Chunk flags (stored in chunk index entry)
 */
typedef enum {
    AOBUNDLE2_CHUNK_MISSING     = 0x0000,  /* No data available */
    AOBUNDLE2_CHUNK_VALID       = 0x0001,  /* Valid JPEG data */
    AOBUNDLE2_CHUNK_PLACEHOLDER = 0x0002,  /* Placeholder color (no JPEG) */
    AOBUNDLE2_CHUNK_UPSCALED    = 0x0004,  /* Upscaled from lower zoom */
    AOBUNDLE2_CHUNK_GARBAGE     = 0x0080,  /* Data marked for reclamation */
} aobundle2_chunk_flags_t;

/**
 * Bundle header structure (64 bytes, fits in single cache line).
 */
typedef struct {
    uint32_t magic;               /* Must be AOBUNDLE2_MAGIC */
    uint16_t version;             /* Format version (2) */
    uint16_t flags;               /* aobundle2_flags_t */
    int32_t  tile_row;            /* Tile row coordinate */
    int32_t  tile_col;            /* Tile column coordinate */
    uint16_t maptype_len;         /* Length of maptype string */
    uint16_t zoom_count;          /* Number of zoom levels stored */
    uint16_t min_zoom;            /* Minimum zoom level */
    uint16_t max_zoom;            /* Maximum zoom level */
    uint32_t total_chunks;        /* Total chunks across all zoom levels */
    uint32_t data_section_offset; /* Offset to data section from file start */
    uint32_t garbage_bytes;       /* Bytes marked as garbage */
    uint64_t last_modified;       /* Unix timestamp */
    uint32_t checksum;            /* CRC32 of header (excluding this field) */
    uint8_t  reserved[12];        /* Future use, must be zero */
} aobundle2_header_t;

/**
 * Zoom level table entry (12 bytes per zoom level).
 */
typedef struct {
    uint16_t zoom_level;          /* Zoom level value */
    uint16_t chunks_per_side;     /* Chunks per side (e.g., 16) */
    uint32_t index_offset;        /* Offset to chunk index from file start */
    uint32_t chunk_count;         /* Total chunks at this zoom level */
} aobundle2_zoom_entry_t;

/**
 * Chunk index entry (16 bytes per chunk).
 */
typedef struct {
    uint32_t data_offset;         /* Offset from data_section_offset */
    uint32_t size;                /* JPEG data size (0 = not stored) */
    uint16_t flags;               /* aobundle2_chunk_flags_t */
    uint16_t quality;             /* Quality indicator (0-100, 0=unknown) */
    uint32_t timestamp;           /* Unix timestamp (lower 32 bits) */
} aobundle2_chunk_index_t;

/**
 * In-memory bundle representation.
 */
typedef struct {
    aobundle2_header_t header;
    char maptype[AOBUNDLE2_MAX_MAPTYPE];
    
    /* Zoom level table */
    aobundle2_zoom_entry_t zoom_table[AOBUNDLE2_MAX_ZOOM_LEVELS];
    
    /* Chunk indices for each zoom level (pointers into mapped/read data) */
    aobundle2_chunk_index_t* chunk_indices[AOBUNDLE2_MAX_ZOOM_LEVELS];
    
    /* Data section pointer */
    uint8_t* data;
    uint32_t data_size;
    
    /* Memory mapping (if used) */
    void* mmap_base;
    size_t mmap_size;
    
    /* File access (for mutations) */
    int fd;                       /* File descriptor (-1 if not open for write) */
    char* path;                   /* Bundle file path (for mutations) */
    
    /* State flags */
    int32_t is_writable;          /* Bundle opened for writing */
    int32_t is_dirty;             /* Header/index needs flush */
} aobundle2_t;

/**
 * Create result structure for detailed feedback.
 */
typedef struct {
    int32_t success;              /* 1 = success, 0 = failure */
    int32_t chunks_written;       /* Number of chunks successfully written */
    int32_t chunks_missing;       /* Number of missing chunks */
    uint32_t bytes_written;       /* Total bytes written to bundle */
    char error_msg[256];          /* Error message if failed */
} aobundle2_result_t;

/* ============================================================================
 * Bundle Creation Functions
 * ============================================================================ */

/**
 * Create a new bundle from individual JPEG files.
 * 
 * @param cache_dir     Directory containing cached JPEGs
 * @param tile_row      Tile row coordinate
 * @param tile_col      Tile column coordinate  
 * @param maptype       Map source identifier
 * @param zoom          Zoom level
 * @param chunks_per_side Number of chunks per side (e.g., 16)
 * @param output_path   Path for output bundle file
 * @param result        Optional result structure for detailed feedback
 * 
 * @return 1 on success, 0 on failure
 */
AOBUNDLE2_API int32_t aobundle2_create(
    const char* cache_dir,
    int32_t tile_row,
    int32_t tile_col,
    const char* maptype,
    int32_t zoom,
    int32_t chunks_per_side,
    const char* output_path,
    aobundle2_result_t* result
);

/**
 * Create a new bundle from JPEG data arrays.
 * 
 * @param tile_row      Tile row coordinate
 * @param tile_col      Tile column coordinate
 * @param maptype       Map source identifier
 * @param zoom          Zoom level
 * @param jpeg_data     Array of JPEG data pointers (NULL = missing)
 * @param jpeg_sizes    Array of JPEG data sizes
 * @param chunk_count   Number of chunks
 * @param output_path   Path for output bundle file
 * @param result        Optional result structure
 * 
 * @return 1 on success, 0 on failure
 */
AOBUNDLE2_API int32_t aobundle2_create_from_data(
    int32_t tile_row,
    int32_t tile_col,
    const char* maptype,
    int32_t zoom,
    const uint8_t** jpeg_data,
    const uint32_t* jpeg_sizes,
    int32_t chunk_count,
    const char* output_path,
    aobundle2_result_t* result
);

/**
 * Create an empty bundle structure for incremental population.
 * 
 * @param tile_row      Tile row coordinate
 * @param tile_col      Tile column coordinate
 * @param maptype       Map source identifier
 * @param initial_zoom  Initial zoom level
 * @param chunks_per_side Chunks per side at initial zoom
 * @param output_path   Path for output bundle file
 * 
 * @return 1 on success, 0 on failure
 */
AOBUNDLE2_API int32_t aobundle2_create_empty(
    int32_t tile_row,
    int32_t tile_col,
    const char* maptype,
    int32_t initial_zoom,
    int32_t chunks_per_side,
    const char* output_path
);

/* ============================================================================
 * Bundle Read Functions (Performance Critical)
 * ============================================================================ */

/**
 * Open an existing bundle file.
 * 
 * @param path      Path to bundle file
 * @param bundle    Output bundle structure (caller allocates)
 * @param use_mmap  If true, memory-map the file; else read into buffer
 * 
 * @return 1 on success, 0 on failure
 */
AOBUNDLE2_API int32_t aobundle2_open(
    const char* path,
    aobundle2_t* bundle,
    int32_t use_mmap
);

/**
 * Open bundle for writing (mutations).
 * Uses file locking to prevent concurrent writes.
 * 
 * @param path      Path to bundle file
 * @param bundle    Output bundle structure
 * 
 * @return 1 on success, 0 on failure (e.g., locked by another process)
 */
AOBUNDLE2_API int32_t aobundle2_open_writable(
    const char* path,
    aobundle2_t* bundle
);

/**
 * Close a bundle and free resources.
 * 
 * @param bundle    Bundle to close
 */
AOBUNDLE2_API void aobundle2_close(aobundle2_t* bundle);

/**
 * Get JPEG data for a specific chunk at a zoom level.
 * 
 * @param bundle    Open bundle
 * @param zoom      Zoom level
 * @param index     Chunk index (0 to chunk_count-1 for that zoom)
 * @param data      Output pointer to JPEG data (do not free)
 * @param size      Output size of JPEG data
 * @param flags     Output chunk flags (optional, can be NULL)
 * 
 * @return 1 if chunk exists and is valid, 0 if missing
 */
AOBUNDLE2_API int32_t aobundle2_get_chunk(
    const aobundle2_t* bundle,
    int32_t zoom,
    int32_t index,
    const uint8_t** data,
    uint32_t* size,
    uint16_t* flags
);

/**
 * Get all chunk pointers for a zoom level.
 * This is the fast path for tile building.
 * 
 * @param bundle        Open bundle
 * @param zoom          Zoom level
 * @param jpeg_data     Output array of JPEG pointers (caller allocates)
 * @param jpeg_sizes    Output array of JPEG sizes (caller allocates)
 * @param max_chunks    Size of output arrays
 * 
 * @return Number of chunks at this zoom level, -1 if zoom not found
 */
AOBUNDLE2_API int32_t aobundle2_get_chunks_for_zoom(
    const aobundle2_t* bundle,
    int32_t zoom,
    const uint8_t** jpeg_data,
    uint32_t* jpeg_sizes,
    int32_t max_chunks
);

/**
 * Check if bundle contains a specific zoom level.
 * 
 * @param bundle    Open bundle
 * @param zoom      Zoom level to check
 * 
 * @return 1 if zoom level exists, 0 if not
 */
AOBUNDLE2_API int32_t aobundle2_has_zoom(
    const aobundle2_t* bundle,
    int32_t zoom
);

/**
 * Get the chunk count for a zoom level.
 * 
 * @param bundle    Open bundle
 * @param zoom      Zoom level
 * 
 * @return Chunk count, or 0 if zoom not found
 */
AOBUNDLE2_API int32_t aobundle2_get_chunk_count(
    const aobundle2_t* bundle,
    int32_t zoom
);

/**
 * Build DDS directly from bundle (optimal single-call path).
 * Combines bundle reading + JPEG decode + DDS build.
 * 
 * @param bundle_path   Path to bundle file
 * @param target_zoom   Target zoom level
 * @param format        DDS format (0=BC1, 1=BC3)
 * @param missing_color RGB fill color for missing chunks
 * @param dds_output    Pre-allocated output buffer
 * @param output_size   Size of output buffer
 * @param bytes_written Actual bytes written (output)
 * 
 * @return 1 on success, 0 on failure
 */
AOBUNDLE2_API int32_t aobundle2_build_dds(
    const char* bundle_path,
    int32_t target_zoom,
    int32_t format,
    uint8_t missing_color[3],
    uint8_t* dds_output,
    uint32_t output_size,
    uint32_t* bytes_written
);

/* ============================================================================
 * Bundle Mutation Functions
 * ============================================================================ */

/**
 * Append a single chunk to the bundle.
 * Uses append-only strategy with atomic index update.
 * 
 * @param bundle    Bundle opened with aobundle2_open_writable()
 * @param zoom      Zoom level
 * @param index     Chunk index
 * @param jpeg_data JPEG data to append
 * @param size      Size of JPEG data
 * @param flags     Chunk flags (VALID, PLACEHOLDER, etc.)
 * 
 * @return 1 on success, 0 on failure
 */
AOBUNDLE2_API int32_t aobundle2_append_chunk(
    aobundle2_t* bundle,
    int32_t zoom,
    int32_t index,
    const uint8_t* jpeg_data,
    uint32_t size,
    uint16_t flags
);

/**
 * Append multiple chunks efficiently (batched write).
 * 
 * @param bundle        Bundle opened for writing
 * @param zoom          Zoom level
 * @param indices       Array of chunk indices
 * @param jpeg_data     Array of JPEG data pointers
 * @param sizes         Array of JPEG sizes
 * @param flags         Array of chunk flags
 * @param count         Number of chunks to append
 * 
 * @return Number of chunks successfully appended
 */
AOBUNDLE2_API int32_t aobundle2_append_chunks_batch(
    aobundle2_t* bundle,
    int32_t zoom,
    const int32_t* indices,
    const uint8_t** jpeg_data,
    const uint32_t* sizes,
    const uint16_t* flags,
    int32_t count
);

/**
 * Add a new zoom level to an existing bundle.
 * 
 * @param bundle        Bundle opened for writing
 * @param new_zoom      New zoom level to add
 * @param chunks_per_side Chunks per side at new zoom
 * 
 * @return 1 on success, 0 on failure
 */
AOBUNDLE2_API int32_t aobundle2_expand_zoom(
    aobundle2_t* bundle,
    int32_t new_zoom,
    int32_t chunks_per_side
);

/**
 * Mark a chunk as missing with placeholder.
 * 
 * @param bundle    Bundle opened for writing
 * @param zoom      Zoom level
 * @param index     Chunk index
 * 
 * @return 1 on success, 0 on failure
 */
AOBUNDLE2_API int32_t aobundle2_mark_missing(
    aobundle2_t* bundle,
    int32_t zoom,
    int32_t index
);

/**
 * Flush any pending changes to disk.
 * 
 * @param bundle    Bundle with pending changes
 * 
 * @return 1 on success, 0 on failure
 */
AOBUNDLE2_API int32_t aobundle2_flush(aobundle2_t* bundle);

/* ============================================================================
 * Compaction Functions
 * ============================================================================ */

/**
 * Calculate fragmentation ratio (garbage_bytes / total_data_bytes).
 * 
 * @param path      Path to bundle file
 * 
 * @return Fragmentation ratio (0.0 to 1.0), -1.0 on error
 */
AOBUNDLE2_API float aobundle2_get_fragmentation(const char* path);

/**
 * Check if bundle needs compaction.
 * 
 * @param path      Path to bundle file
 * @param threshold Fragmentation threshold (default 0.30)
 * 
 * @return 1 if compaction needed, 0 if not, -1 on error
 */
AOBUNDLE2_API int32_t aobundle2_needs_compaction(
    const char* path,
    float threshold
);

/**
 * Compact bundle by rewriting without garbage.
 * Creates new bundle atomically (temp file + rename).
 * 
 * @param path      Path to bundle file
 * 
 * @return Bytes reclaimed, 0 if no compaction needed, -1 on error
 */
AOBUNDLE2_API int64_t aobundle2_compact(const char* path);

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/**
 * Validate bundle file integrity.
 * 
 * @param path      Path to bundle file
 * 
 * @return 1 if valid, 0 if invalid/corrupt
 */
AOBUNDLE2_API int32_t aobundle2_validate(const char* path);

/**
 * Get version information.
 */
AOBUNDLE2_API const char* aobundle2_version(void);

/**
 * Calculate CRC32 checksum.
 */
AOBUNDLE2_API uint32_t aobundle2_crc32(const void* data, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* AOBUNDLE2_H */
