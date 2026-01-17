/**
 * aobundle2.c - Multi-Zoom Mutable Cache Bundle Implementation for AutoOrtho
 * 
 * Implements the AOB2 format for multi-zoom, mutable cache bundles.
 */

#include "aobundle2.h"
#include "aodds.h"
#include "internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#ifdef AOPIPELINE_WINDOWS
#include <windows.h>
#include <io.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/file.h>
#endif

#define AOBUNDLE2_VERSION_STR "2.0.0"

/* I/O optimization: 256KB stdio buffer for efficient large writes */
#define AOBUNDLE2_WRITE_BUFFER_SIZE (256 * 1024)

/* ============================================================================
 * CRC32 Implementation (IEEE 802.3 polynomial)
 * ============================================================================ */

static uint32_t crc32_table[256];
static int crc32_table_initialized = 0;

static void init_crc32_table(void) {
    if (crc32_table_initialized) return;
    
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t crc = i;
        for (int j = 0; j < 8; j++) {
            crc = (crc >> 1) ^ (crc & 1 ? 0xEDB88320 : 0);
        }
        crc32_table[i] = crc;
    }
    crc32_table_initialized = 1;
}

AOBUNDLE2_API uint32_t aobundle2_crc32(const void* data, size_t len) {
    init_crc32_table();
    
    const uint8_t* bytes = (const uint8_t*)data;
    uint32_t crc = 0xFFFFFFFF;
    
    for (size_t i = 0; i < len; i++) {
        crc = (crc >> 8) ^ crc32_table[(crc ^ bytes[i]) & 0xFF];
    }
    
    return crc ^ 0xFFFFFFFF;
}

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

/* Align value to boundary */
static inline uint32_t align_to(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

/* Get current Unix timestamp */
static inline uint64_t get_timestamp(void) {
    return (uint64_t)time(NULL);
}

/* Find zoom level index in zoom table */
static int32_t find_zoom_index(const aobundle2_t* bundle, int32_t zoom) {
    for (uint16_t i = 0; i < bundle->header.zoom_count; i++) {
        if (bundle->zoom_table[i].zoom_level == (uint16_t)zoom) {
            return (int32_t)i;
        }
    }
    return -1;
}

/* Calculate header checksum (excluding checksum field) */
static uint32_t calc_header_checksum(const aobundle2_header_t* hdr) {
    /* Create temp copy and zero checksum field */
    aobundle2_header_t temp = *hdr;
    temp.checksum = 0;
    return aobundle2_crc32(&temp, sizeof(temp));
}

/* Pre-size file to expected size for contiguous block allocation.
 * This is a best-effort optimization - failure is non-fatal. */
static int32_t preallocate_file(FILE* f, size_t size) {
    if (!f || size == 0) return 0;
    
#ifdef AOPIPELINE_LINUX
    /* Linux: posix_fallocate is most efficient - actually allocates blocks */
    int fd = fileno(f);
    if (fd >= 0) {
        if (posix_fallocate(fd, 0, (off_t)size) == 0) {
            return 1;
        }
        /* Fall through to ftruncate on failure */
    }
#endif
    
#ifdef AOPIPELINE_WINDOWS
    /* Windows: SetEndOfFile after positioning */
    int fd = _fileno(f);
    if (fd >= 0) {
        HANDLE hFile = (HANDLE)_get_osfhandle(fd);
        if (hFile != INVALID_HANDLE_VALUE) {
            LARGE_INTEGER li;
            li.QuadPart = (LONGLONG)size;
            if (SetFilePointerEx(hFile, li, NULL, FILE_BEGIN) &&
                SetEndOfFile(hFile)) {
                /* Reset position to start for subsequent writes */
                li.QuadPart = 0;
                SetFilePointerEx(hFile, li, NULL, FILE_BEGIN);
                return 1;
            }
        }
    }
#else
    /* POSIX fallback (macOS, others): ftruncate */
    int fd = fileno(f);
    if (fd >= 0) {
        if (ftruncate(fd, (off_t)size) == 0) {
            return 1;
        }
    }
#endif
    
    return 0;  /* Non-fatal: writing will still work, just less efficiently */
}

/* ============================================================================
 * Windows Memory-Mapped File Wrapper
 * ============================================================================
 * Provides mmap-like functionality on Windows using CreateFileMapping/MapViewOfFile.
 * This enables zero-copy file access with OS-managed page caching.
 */

#ifdef AOPIPELINE_WINDOWS
typedef struct {
    HANDLE hFile;
    HANDLE hMapping;
    void* base;
    size_t size;
} win_mmap_t;

static int win_mmap_open(const char* path, win_mmap_t* mm) {
    memset(mm, 0, sizeof(*mm));
    mm->hFile = INVALID_HANDLE_VALUE;
    
    /* Open with sequential scan hint for better prefetching */
    mm->hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ,
                            NULL, OPEN_EXISTING, 
                            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, 
                            NULL);
    if (mm->hFile == INVALID_HANDLE_VALUE) return 0;
    
    LARGE_INTEGER fsize;
    if (!GetFileSizeEx(mm->hFile, &fsize)) {
        CloseHandle(mm->hFile);
        mm->hFile = INVALID_HANDLE_VALUE;
        return 0;
    }
    mm->size = (size_t)fsize.QuadPart;
    
    /* Create file mapping - SEC_READONLY for potential sharing */
    mm->hMapping = CreateFileMappingA(mm->hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!mm->hMapping) {
        CloseHandle(mm->hFile);
        mm->hFile = INVALID_HANDLE_VALUE;
        return 0;
    }
    
    /* Map view of entire file */
    mm->base = MapViewOfFile(mm->hMapping, FILE_MAP_READ, 0, 0, 0);
    if (!mm->base) {
        CloseHandle(mm->hMapping);
        CloseHandle(mm->hFile);
        mm->hMapping = NULL;
        mm->hFile = INVALID_HANDLE_VALUE;
        return 0;
    }
    
    return 1;
}

static void win_mmap_close(win_mmap_t* mm) {
    if (mm->base) {
        UnmapViewOfFile(mm->base);
        mm->base = NULL;
    }
    if (mm->hMapping) {
        CloseHandle(mm->hMapping);
        mm->hMapping = NULL;
    }
    if (mm->hFile != INVALID_HANDLE_VALUE) {
        CloseHandle(mm->hFile);
        mm->hFile = INVALID_HANDLE_VALUE;
    }
    mm->size = 0;
}
#endif

/* Read entire file contents */
static uint8_t* read_file_contents(const char* path, uint32_t* size) {
    *size = 0;
    
#ifdef AOPIPELINE_WINDOWS
    HANDLE hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, 
                               NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) return NULL;
    
    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(hFile, &file_size)) {
        CloseHandle(hFile);
        return NULL;
    }
    
    uint8_t* data = (uint8_t*)malloc((size_t)file_size.QuadPart);
    if (!data) {
        CloseHandle(hFile);
        return NULL;
    }
    
    DWORD bytes_read;
    if (!ReadFile(hFile, data, (DWORD)file_size.QuadPart, &bytes_read, NULL)) {
        free(data);
        CloseHandle(hFile);
        return NULL;
    }
    
    CloseHandle(hFile);
    *size = bytes_read;
    return data;
#else
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    if (fsize <= 0) {
        fclose(f);
        return NULL;
    }
    
    uint8_t* data = (uint8_t*)malloc(fsize);
    if (!data) {
        fclose(f);
        return NULL;
    }
    
    size_t read = fread(data, 1, fsize, f);
    fclose(f);
    
    if (read != (size_t)fsize) {
        free(data);
        return NULL;
    }
    
    *size = (uint32_t)read;
    return data;
#endif
}

/* Atomic file write using temp file + rename.
 * Optimized with 256KB stdio buffer and file pre-allocation. */
static int32_t atomic_write_file(const char* path, const void* data, size_t size) {
    char temp_path[4096];
    snprintf(temp_path, sizeof(temp_path), "%s.tmp.%ld", path, (long)get_timestamp());
    
    FILE* f = fopen(temp_path, "wb");
    if (!f) return 0;
    
    /* Set 256KB stdio buffer for efficient large writes.
     * Thread-local to avoid contention in parallel consolidation. */
    static TLS_VAR char write_buffer[AOBUNDLE2_WRITE_BUFFER_SIZE];
    setvbuf(f, write_buffer, _IOFBF, sizeof(write_buffer));
    
    /* Pre-allocate file size for contiguous blocks (best effort) */
    preallocate_file(f, size);
    
    size_t written = fwrite(data, 1, size, f);
    fclose(f);
    
    if (written != size) {
        remove(temp_path);
        return 0;
    }
    
#ifdef AOPIPELINE_WINDOWS
    if (!MoveFileExA(temp_path, path, MOVEFILE_REPLACE_EXISTING)) {
        remove(temp_path);
        return 0;
    }
#else
    if (rename(temp_path, path) != 0) {
        remove(temp_path);
        return 0;
    }
#endif
    
    return 1;
}

/* Cross-platform file locking */
#ifdef AOPIPELINE_WINDOWS
static int32_t lock_file(HANDLE hFile, int exclusive) {
    OVERLAPPED ov = {0};
    DWORD flags = exclusive ? LOCKFILE_EXCLUSIVE_LOCK : 0;
    return LockFileEx(hFile, flags, 0, 0xFFFFFFFF, 0, &ov) ? 1 : 0;
}

static void unlock_file(HANDLE hFile) {
    OVERLAPPED ov = {0};
    UnlockFileEx(hFile, 0, 0xFFFFFFFF, 0, &ov);
}
#else
static int32_t lock_file(int fd, int exclusive) {
    return flock(fd, exclusive ? LOCK_EX : LOCK_SH) == 0 ? 1 : 0;
}

static void unlock_file(int fd) {
    flock(fd, LOCK_UN);
}
#endif

/* ============================================================================
 * Bundle Creation Functions
 * ============================================================================ */

AOBUNDLE2_API int32_t aobundle2_create(
    const char* cache_dir,
    int32_t tile_row,
    int32_t tile_col,
    const char* maptype,
    int32_t zoom,
    int32_t chunks_per_side,
    const char* output_path,
    aobundle2_result_t* result
) {
    if (result) memset(result, 0, sizeof(aobundle2_result_t));
    
    if (!cache_dir || !maptype || !output_path || chunks_per_side <= 0) {
        if (result) {
            safe_strcpy(result->error_msg, "Invalid arguments", sizeof(result->error_msg));
        }
        return 0;
    }
    
    int32_t chunk_count = chunks_per_side * chunks_per_side;
    if (chunk_count > AOBUNDLE2_MAX_CHUNKS_PER_ZOOM) {
        if (result) {
            safe_strcpy(result->error_msg, "Too many chunks", sizeof(result->error_msg));
        }
        return 0;
    }
    
    /* Read all JPEG files */
    uint8_t** jpeg_data = (uint8_t**)calloc(chunk_count, sizeof(uint8_t*));
    uint32_t* jpeg_sizes = (uint32_t*)calloc(chunk_count, sizeof(uint32_t));
    
    if (!jpeg_data || !jpeg_sizes) {
        free(jpeg_data);
        free(jpeg_sizes);
        if (result) {
            safe_strcpy(result->error_msg, "Memory allocation failed", sizeof(result->error_msg));
        }
        return 0;
    }
    
    uint32_t total_data_size = 0;
    int32_t chunks_found = 0;
    int32_t chunks_missing = 0;
    char path_buf[4096];
    
    for (int32_t i = 0; i < chunk_count; i++) {
        int32_t chunk_row = i / chunks_per_side;
        int32_t chunk_col = i % chunks_per_side;
        int32_t abs_col = tile_col * chunks_per_side + chunk_col;
        int32_t abs_row = tile_row * chunks_per_side + chunk_row;
        
        snprintf(path_buf, sizeof(path_buf), "%s/%d_%d_%d_%s.jpg",
                 cache_dir, abs_col, abs_row, zoom, maptype);
        
        jpeg_data[i] = read_file_contents(path_buf, &jpeg_sizes[i]);
        if (jpeg_data[i]) {
            total_data_size += jpeg_sizes[i];
            chunks_found++;
        } else {
            chunks_missing++;
        }
    }
    
    /* Create bundle from loaded data */
    int32_t success = aobundle2_create_from_data(
        tile_row, tile_col, maptype, zoom,
        (const uint8_t**)jpeg_data, jpeg_sizes, chunk_count,
        output_path, result
    );
    
    /* Cleanup */
    for (int32_t i = 0; i < chunk_count; i++) {
        free(jpeg_data[i]);
    }
    free(jpeg_data);
    free(jpeg_sizes);
    
    if (result) {
        result->chunks_written = chunks_found;
        result->chunks_missing = chunks_missing;
    }
    
    return success;
}

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
) {
    if (result) memset(result, 0, sizeof(aobundle2_result_t));
    
    if (!maptype || !output_path || chunk_count <= 0) {
        if (result) {
            safe_strcpy(result->error_msg, "Invalid arguments", sizeof(result->error_msg));
        }
        return 0;
    }
    
    /* Calculate chunks per side */
    int32_t chunks_per_side = (int32_t)sqrt((double)chunk_count);
    if (chunks_per_side * chunks_per_side != chunk_count) {
        if (result) {
            safe_strcpy(result->error_msg, "Chunk count must be perfect square", sizeof(result->error_msg));
        }
        return 0;
    }
    
    /* Calculate total data size */
    uint32_t total_data_size = 0;
    int32_t valid_chunks = 0;
    for (int32_t i = 0; i < chunk_count; i++) {
        if (jpeg_data && jpeg_data[i] && jpeg_sizes && jpeg_sizes[i] > 0) {
            total_data_size += jpeg_sizes[i];
            valid_chunks++;
        }
    }
    
    /* Calculate sizes */
    size_t maptype_len = strlen(maptype);
    if (maptype_len > AOBUNDLE2_MAX_MAPTYPE - 1) {
        maptype_len = AOBUNDLE2_MAX_MAPTYPE - 1;
    }
    size_t maptype_padded = align_to((uint32_t)maptype_len, 8);
    
    /* Layout:
     * - Header (64 bytes)
     * - Maptype string (padded to 8 bytes)
     * - Zoom table (12 bytes x 1)
     * - Chunk indices (16 bytes x chunk_count)
     * - Data section
     */
    size_t header_size = AOBUNDLE2_HEADER_SIZE;
    size_t zoom_table_size = sizeof(aobundle2_zoom_entry_t);
    size_t index_size = chunk_count * sizeof(aobundle2_chunk_index_t);
    size_t data_offset = header_size + maptype_padded + zoom_table_size + index_size;
    size_t total_size = data_offset + total_data_size;
    
    /* Allocate bundle buffer */
    uint8_t* bundle_buf = (uint8_t*)calloc(1, total_size);
    if (!bundle_buf) {
        if (result) {
            safe_strcpy(result->error_msg, "Memory allocation failed", sizeof(result->error_msg));
        }
        return 0;
    }
    
    /* Write header */
    aobundle2_header_t* header = (aobundle2_header_t*)bundle_buf;
    header->magic = AOBUNDLE2_MAGIC;
    header->version = AOBUNDLE2_VERSION;
    header->flags = AOBUNDLE2_FLAG_MUTABLE;
    header->tile_row = tile_row;
    header->tile_col = tile_col;
    header->maptype_len = (uint16_t)maptype_len;
    header->zoom_count = 1;
    header->min_zoom = (uint16_t)zoom;
    header->max_zoom = (uint16_t)zoom;
    header->total_chunks = (uint32_t)chunk_count;
    header->data_section_offset = (uint32_t)data_offset;
    header->garbage_bytes = 0;
    header->last_modified = get_timestamp();
    memset(header->reserved, 0, sizeof(header->reserved));
    
    /* Write maptype */
    memcpy(bundle_buf + header_size, maptype, maptype_len);
    
    /* Write zoom table entry */
    aobundle2_zoom_entry_t* zoom_entry = (aobundle2_zoom_entry_t*)(bundle_buf + header_size + maptype_padded);
    zoom_entry->zoom_level = (uint16_t)zoom;
    zoom_entry->chunks_per_side = (uint16_t)chunks_per_side;
    zoom_entry->index_offset = (uint32_t)(header_size + maptype_padded + zoom_table_size);
    zoom_entry->chunk_count = (uint32_t)chunk_count;
    
    /* Write chunk indices and data */
    aobundle2_chunk_index_t* indices = (aobundle2_chunk_index_t*)(bundle_buf + zoom_entry->index_offset);
    uint8_t* data_section = bundle_buf + data_offset;
    uint32_t current_offset = 0;
    uint32_t timestamp = (uint32_t)get_timestamp();
    
    for (int32_t i = 0; i < chunk_count; i++) {
        if (jpeg_data && jpeg_data[i] && jpeg_sizes && jpeg_sizes[i] > 0) {
            indices[i].data_offset = current_offset;
            indices[i].size = jpeg_sizes[i];
            indices[i].flags = AOBUNDLE2_CHUNK_VALID;
            indices[i].quality = 0;  /* Unknown quality */
            indices[i].timestamp = timestamp;
            memcpy(data_section + current_offset, jpeg_data[i], jpeg_sizes[i]);
            current_offset += jpeg_sizes[i];
        } else {
            indices[i].data_offset = 0;
            indices[i].size = 0;
            indices[i].flags = AOBUNDLE2_CHUNK_MISSING;
            indices[i].quality = 0;
            indices[i].timestamp = timestamp;
        }
    }
    
    /* Calculate and write header checksum */
    header->checksum = calc_header_checksum(header);
    
    /* Write atomically */
    int32_t success = atomic_write_file(output_path, bundle_buf, total_size);
    free(bundle_buf);
    
    if (result) {
        result->success = success;
        result->chunks_written = valid_chunks;
        result->chunks_missing = chunk_count - valid_chunks;
        result->bytes_written = (uint32_t)total_size;
    }
    
    return success;
}

AOBUNDLE2_API int32_t aobundle2_create_empty(
    int32_t tile_row,
    int32_t tile_col,
    const char* maptype,
    int32_t initial_zoom,
    int32_t chunks_per_side,
    const char* output_path
) {
    int32_t chunk_count = chunks_per_side * chunks_per_side;
    
    /* Create with NULL data arrays */
    return aobundle2_create_from_data(
        tile_row, tile_col, maptype, initial_zoom,
        NULL, NULL, chunk_count,
        output_path, NULL
    );
}

/* ============================================================================
 * Bundle Read Functions
 * ============================================================================ */

AOBUNDLE2_API int32_t aobundle2_open(
    const char* path,
    aobundle2_t* bundle,
    int32_t use_mmap
) {
    if (!path || !bundle) return 0;
    
    memset(bundle, 0, sizeof(aobundle2_t));
    bundle->fd = -1;
#ifdef AOPIPELINE_WINDOWS
    bundle->win_file = INVALID_HANDLE_VALUE;
    bundle->win_mapping = NULL;
#endif
    
#ifdef AOPIPELINE_WINDOWS
    if (use_mmap) {
        /* Windows memory-mapped access */
        win_mmap_t wmm;
        if (win_mmap_open(path, &wmm)) {
            bundle->mmap_base = wmm.base;
            bundle->mmap_size = wmm.size;
            bundle->win_file = wmm.hFile;
            bundle->win_mapping = wmm.hMapping;
            
            /* Parse from mmap - same as POSIX path */
            if (wmm.size < AOBUNDLE2_HEADER_SIZE) {
                win_mmap_close(&wmm);
                memset(bundle, 0, sizeof(*bundle));
                bundle->fd = -1;
                bundle->win_file = INVALID_HANDLE_VALUE;
                bundle->win_mapping = NULL;
                return 0;
            }
            
            aobundle2_header_t* hdr = (aobundle2_header_t*)wmm.base;
            if (hdr->magic != AOBUNDLE2_MAGIC) {
                win_mmap_close(&wmm);
                memset(bundle, 0, sizeof(*bundle));
                bundle->fd = -1;
                bundle->win_file = INVALID_HANDLE_VALUE;
                bundle->win_mapping = NULL;
                return 0;
            }
            
            /* Verify checksum */
            uint32_t expected_checksum = calc_header_checksum(hdr);
            if (hdr->checksum != expected_checksum) {
                win_mmap_close(&wmm);
                memset(bundle, 0, sizeof(*bundle));
                bundle->fd = -1;
                bundle->win_file = INVALID_HANDLE_VALUE;
                bundle->win_mapping = NULL;
                return 0;
            }
            
            bundle->header = *hdr;
            
            /* Copy maptype */
            size_t maptype_offset = AOBUNDLE2_HEADER_SIZE;
            size_t maptype_len = hdr->maptype_len;
            if (maptype_len > AOBUNDLE2_MAX_MAPTYPE - 1) {
                maptype_len = AOBUNDLE2_MAX_MAPTYPE - 1;
            }
            memcpy(bundle->maptype, (uint8_t*)wmm.base + maptype_offset, maptype_len);
            bundle->maptype[maptype_len] = '\0';
            
            /* Parse zoom table */
            size_t maptype_padded = align_to(hdr->maptype_len, 8);
            size_t zoom_table_offset = maptype_offset + maptype_padded;
            aobundle2_zoom_entry_t* zt = (aobundle2_zoom_entry_t*)((uint8_t*)wmm.base + zoom_table_offset);
            
            for (uint16_t i = 0; i < hdr->zoom_count && i < AOBUNDLE2_MAX_ZOOM_LEVELS; i++) {
                bundle->zoom_table[i] = zt[i];
                bundle->chunk_indices[i] = (aobundle2_chunk_index_t*)((uint8_t*)wmm.base + zt[i].index_offset);
            }
            
            /* Point to data section */
            bundle->data = (uint8_t*)wmm.base + hdr->data_section_offset;
            bundle->data_size = (uint32_t)(wmm.size - hdr->data_section_offset);
            
            return 1;
        }
        /* Fall through to standard I/O if mmap fails */
    }
#else
    if (use_mmap) {
        /* Memory-mapped access */
        int fd = open(path, O_RDONLY);
        if (fd < 0) return 0;
        
        struct stat st;
        if (fstat(fd, &st) < 0) {
            close(fd);
            return 0;
        }
        
        void* mapped = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        
        if (mapped == MAP_FAILED) return 0;
        
        /* Advise sequential access */
        madvise(mapped, st.st_size, MADV_SEQUENTIAL);
        
        bundle->mmap_base = mapped;
        bundle->mmap_size = st.st_size;
        
        /* Parse header from mapped memory */
        if (st.st_size < (off_t)AOBUNDLE2_HEADER_SIZE) {
            munmap(mapped, st.st_size);
            return 0;
        }
        
        aobundle2_header_t* hdr = (aobundle2_header_t*)mapped;
        if (hdr->magic != AOBUNDLE2_MAGIC) {
            munmap(mapped, st.st_size);
            return 0;
        }
        
        /* Verify checksum */
        uint32_t expected_checksum = calc_header_checksum(hdr);
        if (hdr->checksum != expected_checksum) {
            munmap(mapped, st.st_size);
            return 0;
        }
        
        bundle->header = *hdr;
        
        /* Copy maptype */
        size_t maptype_offset = AOBUNDLE2_HEADER_SIZE;
        size_t maptype_len = hdr->maptype_len;
        if (maptype_len > AOBUNDLE2_MAX_MAPTYPE - 1) {
            maptype_len = AOBUNDLE2_MAX_MAPTYPE - 1;
        }
        memcpy(bundle->maptype, (uint8_t*)mapped + maptype_offset, maptype_len);
        bundle->maptype[maptype_len] = '\0';
        
        /* Parse zoom table */
        size_t maptype_padded = align_to(hdr->maptype_len, 8);
        size_t zoom_table_offset = maptype_offset + maptype_padded;
        aobundle2_zoom_entry_t* zt = (aobundle2_zoom_entry_t*)((uint8_t*)mapped + zoom_table_offset);
        
        for (uint16_t i = 0; i < hdr->zoom_count && i < AOBUNDLE2_MAX_ZOOM_LEVELS; i++) {
            bundle->zoom_table[i] = zt[i];
            bundle->chunk_indices[i] = (aobundle2_chunk_index_t*)((uint8_t*)mapped + zt[i].index_offset);
        }
        
        /* Point to data section */
        bundle->data = (uint8_t*)mapped + hdr->data_section_offset;
        bundle->data_size = st.st_size - hdr->data_section_offset;
        
        return 1;
    }
#endif
    
    /* Regular file read */
    uint32_t file_size;
    uint8_t* file_data = read_file_contents(path, &file_size);
    if (!file_data) return 0;
    
    if (file_size < AOBUNDLE2_HEADER_SIZE) {
        free(file_data);
        return 0;
    }
    
    aobundle2_header_t* hdr = (aobundle2_header_t*)file_data;
    if (hdr->magic != AOBUNDLE2_MAGIC) {
        free(file_data);
        return 0;
    }
    
    /* Verify checksum */
    uint32_t expected_checksum = calc_header_checksum(hdr);
    if (hdr->checksum != expected_checksum) {
        free(file_data);
        return 0;
    }
    
    bundle->header = *hdr;
    
    /* Copy maptype */
    size_t maptype_offset = AOBUNDLE2_HEADER_SIZE;
    size_t maptype_len = hdr->maptype_len;
    if (maptype_len > AOBUNDLE2_MAX_MAPTYPE - 1) {
        maptype_len = AOBUNDLE2_MAX_MAPTYPE - 1;
    }
    memcpy(bundle->maptype, file_data + maptype_offset, maptype_len);
    bundle->maptype[maptype_len] = '\0';
    
    /* Parse zoom table and allocate index copies */
    size_t maptype_padded = align_to(hdr->maptype_len, 8);
    size_t zoom_table_offset = maptype_offset + maptype_padded;
    aobundle2_zoom_entry_t* zt = (aobundle2_zoom_entry_t*)(file_data + zoom_table_offset);
    
    for (uint16_t i = 0; i < hdr->zoom_count && i < AOBUNDLE2_MAX_ZOOM_LEVELS; i++) {
        bundle->zoom_table[i] = zt[i];
        
        /* Allocate and copy chunk indices for this zoom level */
        size_t idx_size = zt[i].chunk_count * sizeof(aobundle2_chunk_index_t);
        bundle->chunk_indices[i] = (aobundle2_chunk_index_t*)malloc(idx_size);
        if (!bundle->chunk_indices[i]) {
            /* Cleanup on failure */
            for (int j = 0; j < i; j++) {
                free(bundle->chunk_indices[j]);
            }
            free(file_data);
            return 0;
        }
        memcpy(bundle->chunk_indices[i], file_data + zt[i].index_offset, idx_size);
    }
    
    /* Allocate and copy data section */
    bundle->data_size = file_size - hdr->data_section_offset;
    bundle->data = (uint8_t*)malloc(bundle->data_size);
    if (!bundle->data) {
        for (int i = 0; i < hdr->zoom_count; i++) {
            free(bundle->chunk_indices[i]);
        }
        free(file_data);
        return 0;
    }
    memcpy(bundle->data, file_data + hdr->data_section_offset, bundle->data_size);
    
    free(file_data);
    return 1;
}

AOBUNDLE2_API int32_t aobundle2_open_writable(
    const char* path,
    aobundle2_t* bundle
) {
    if (!path || !bundle) return 0;
    
    /* First open read-only to parse structure */
    if (!aobundle2_open(path, bundle, 0)) {
        return 0;
    }
    
    /* Store path for mutations */
    bundle->path = strdup(path);
    if (!bundle->path) {
        aobundle2_close(bundle);
        return 0;
    }
    
    /* Open file for writing with lock */
#ifdef AOPIPELINE_WINDOWS
    HANDLE hFile = CreateFileA(path, GENERIC_READ | GENERIC_WRITE, 0,
                               NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        aobundle2_close(bundle);
        return 0;
    }
    
    if (!lock_file(hFile, 1)) {
        CloseHandle(hFile);
        aobundle2_close(bundle);
        return 0;
    }
    
    bundle->fd = (int)(intptr_t)hFile;  /* Store as int for compatibility */
#else
    int fd = open(path, O_RDWR);
    if (fd < 0) {
        aobundle2_close(bundle);
        return 0;
    }
    
    if (!lock_file(fd, 1)) {
        close(fd);
        aobundle2_close(bundle);
        return 0;
    }
    
    bundle->fd = fd;
#endif
    
    bundle->is_writable = 1;
    return 1;
}

AOBUNDLE2_API void aobundle2_close(aobundle2_t* bundle) {
    if (!bundle) return;
    
    /* Flush pending changes */
    if (bundle->is_dirty) {
        aobundle2_flush(bundle);
    }
    
    /* Unlock and close file descriptor */
    if (bundle->fd >= 0) {
#ifdef AOPIPELINE_WINDOWS
        HANDLE hFile = (HANDLE)(intptr_t)bundle->fd;
        unlock_file(hFile);
        CloseHandle(hFile);
#else
        unlock_file(bundle->fd);
        close(bundle->fd);
#endif
    }
    
    /* Free path */
    free(bundle->path);
    
#ifdef AOPIPELINE_WINDOWS
    /* Windows mmap cleanup */
    if (bundle->mmap_base) {
        UnmapViewOfFile(bundle->mmap_base);
        bundle->mmap_base = NULL;
        bundle->mmap_size = 0;
        /* Indices and data point into mmap */
        for (int i = 0; i < AOBUNDLE2_MAX_ZOOM_LEVELS; i++) {
            bundle->chunk_indices[i] = NULL;
        }
        bundle->data = NULL;
    }
    if (bundle->win_mapping) {
        CloseHandle(bundle->win_mapping);
        bundle->win_mapping = NULL;
    }
    if (bundle->win_file != INVALID_HANDLE_VALUE) {
        CloseHandle(bundle->win_file);
        bundle->win_file = INVALID_HANDLE_VALUE;
    }
    
    /* If not mmap'd, free allocated memory */
    if (!bundle->mmap_base) {
        for (int i = 0; i < bundle->header.zoom_count && i < AOBUNDLE2_MAX_ZOOM_LEVELS; i++) {
            free(bundle->chunk_indices[i]);
        }
        free(bundle->data);
    }
#else
    /* POSIX mmap cleanup */
    if (bundle->mmap_base) {
        munmap(bundle->mmap_base, bundle->mmap_size);
        bundle->mmap_base = NULL;
        bundle->mmap_size = 0;
        /* Indices and data point into mmap */
        for (int i = 0; i < AOBUNDLE2_MAX_ZOOM_LEVELS; i++) {
            bundle->chunk_indices[i] = NULL;
        }
        bundle->data = NULL;
    } else {
        /* Free allocated memory */
        for (int i = 0; i < bundle->header.zoom_count && i < AOBUNDLE2_MAX_ZOOM_LEVELS; i++) {
            free(bundle->chunk_indices[i]);
        }
        free(bundle->data);
    }
#endif
    
    memset(bundle, 0, sizeof(aobundle2_t));
    bundle->fd = -1;
#ifdef AOPIPELINE_WINDOWS
    bundle->win_file = INVALID_HANDLE_VALUE;
    bundle->win_mapping = NULL;
#endif
}

AOBUNDLE2_API int32_t aobundle2_get_chunk(
    const aobundle2_t* bundle,
    int32_t zoom,
    int32_t index,
    const uint8_t** data,
    uint32_t* size,
    uint16_t* flags
) {
    if (!bundle || !data || !size) return 0;
    
    int32_t zi = find_zoom_index(bundle, zoom);
    if (zi < 0) return 0;
    
    if (index < 0 || (uint32_t)index >= bundle->zoom_table[zi].chunk_count) return 0;
    
    const aobundle2_chunk_index_t* idx = &bundle->chunk_indices[zi][index];
    
    if (flags) *flags = idx->flags;
    
    if (idx->size == 0 || (idx->flags & AOBUNDLE2_CHUNK_GARBAGE)) {
        *data = NULL;
        *size = 0;
        return 0;  /* Missing or garbage chunk */
    }
    
    *data = bundle->data + idx->data_offset;
    *size = idx->size;
    return 1;
}

AOBUNDLE2_API int32_t aobundle2_get_chunks_for_zoom(
    const aobundle2_t* bundle,
    int32_t zoom,
    const uint8_t** jpeg_data,
    uint32_t* jpeg_sizes,
    int32_t max_chunks
) {
    if (!bundle || !jpeg_data || !jpeg_sizes) return -1;
    
    int32_t zi = find_zoom_index(bundle, zoom);
    if (zi < 0) return -1;
    
    int32_t count = (int32_t)bundle->zoom_table[zi].chunk_count;
    if (count > max_chunks) count = max_chunks;
    
    for (int32_t i = 0; i < count; i++) {
        const aobundle2_chunk_index_t* idx = &bundle->chunk_indices[zi][i];
        
        if (idx->size > 0 && !(idx->flags & AOBUNDLE2_CHUNK_GARBAGE)) {
            jpeg_data[i] = bundle->data + idx->data_offset;
            jpeg_sizes[i] = idx->size;
        } else {
            jpeg_data[i] = NULL;
            jpeg_sizes[i] = 0;
        }
    }
    
    return count;
}

AOBUNDLE2_API int32_t aobundle2_has_zoom(const aobundle2_t* bundle, int32_t zoom) {
    if (!bundle) return 0;
    return find_zoom_index(bundle, zoom) >= 0 ? 1 : 0;
}

AOBUNDLE2_API int32_t aobundle2_get_chunk_count(const aobundle2_t* bundle, int32_t zoom) {
    if (!bundle) return 0;
    int32_t zi = find_zoom_index(bundle, zoom);
    if (zi < 0) return 0;
    return (int32_t)bundle->zoom_table[zi].chunk_count;
}

AOBUNDLE2_API int32_t aobundle2_build_dds(
    const char* bundle_path,
    int32_t target_zoom,
    int32_t format,
    uint8_t missing_color[3],
    uint8_t* dds_output,
    uint32_t output_size,
    uint32_t* bytes_written
) {
    if (!bundle_path || !dds_output || !bytes_written) return 0;
    
    /* Open bundle with mmap for fastest access (all platforms now support mmap) */
    aobundle2_t bundle;
    if (!aobundle2_open(bundle_path, &bundle, 1)) return 0;
    
    /* Check if target zoom exists */
    int32_t zi = find_zoom_index(&bundle, target_zoom);
    if (zi < 0) {
        aobundle2_close(&bundle);
        return 0;
    }
    
    int32_t chunk_count = (int32_t)bundle.zoom_table[zi].chunk_count;
    
    /* Allocate arrays for jpeg pointers */
    const uint8_t** jpeg_ptrs = (const uint8_t**)malloc(chunk_count * sizeof(uint8_t*));
    uint32_t* jpeg_sizes = (uint32_t*)malloc(chunk_count * sizeof(uint32_t));
    
    if (!jpeg_ptrs || !jpeg_sizes) {
        free(jpeg_ptrs);
        free(jpeg_sizes);
        aobundle2_close(&bundle);
        return 0;
    }
    
    /* Get all chunk pointers */
    aobundle2_get_chunks_for_zoom(&bundle, target_zoom, jpeg_ptrs, jpeg_sizes, chunk_count);
    
    /* Build DDS using existing aodds function */
    int32_t result = aodds_build_from_jpegs(
        jpeg_ptrs,
        jpeg_sizes,
        chunk_count,
        (dds_format_t)format,
        missing_color[0],
        missing_color[1],
        missing_color[2],
        dds_output,
        output_size,
        bytes_written,
        NULL
    );
    
    free(jpeg_ptrs);
    free(jpeg_sizes);
    aobundle2_close(&bundle);
    
    return result;
}

/* ============================================================================
 * Bundle Mutation Functions
 * ============================================================================ */

AOBUNDLE2_API int32_t aobundle2_append_chunk(
    aobundle2_t* bundle,
    int32_t zoom,
    int32_t index,
    const uint8_t* jpeg_data,
    uint32_t size,
    uint16_t flags
) {
    if (!bundle || !bundle->is_writable || !jpeg_data || size == 0) return 0;
    
    int32_t zi = find_zoom_index(bundle, zoom);
    if (zi < 0) return 0;
    
    if (index < 0 || (uint32_t)index >= bundle->zoom_table[zi].chunk_count) return 0;
    
    aobundle2_chunk_index_t* idx = &bundle->chunk_indices[zi][index];
    
    /* Mark old data as garbage if exists */
    if (idx->size > 0 && !(idx->flags & AOBUNDLE2_CHUNK_GARBAGE)) {
        bundle->header.garbage_bytes += idx->size;
        idx->flags |= AOBUNDLE2_CHUNK_GARBAGE;
    }
    
    /* Append new data to end of file */
#ifdef AOPIPELINE_WINDOWS
    HANDLE hFile = (HANDLE)(intptr_t)bundle->fd;
    LARGE_INTEGER liDistanceToMove = {0};
    liDistanceToMove.QuadPart = 0;
    LARGE_INTEGER liNewPos;
    SetFilePointerEx(hFile, liDistanceToMove, &liNewPos, FILE_END);
    
    uint32_t new_offset = (uint32_t)(liNewPos.QuadPart - bundle->header.data_section_offset);
    
    DWORD bytes_written;
    if (!WriteFile(hFile, jpeg_data, size, &bytes_written, NULL) || bytes_written != size) {
        return 0;
    }
#else
    /* Seek to end */
    off_t end_pos = lseek(bundle->fd, 0, SEEK_END);
    if (end_pos < 0) return 0;
    
    uint32_t new_offset = (uint32_t)(end_pos - bundle->header.data_section_offset);
    
    ssize_t written = write(bundle->fd, jpeg_data, size);
    if (written != (ssize_t)size) return 0;
#endif
    
    /* Update index entry */
    idx->data_offset = new_offset;
    idx->size = size;
    idx->flags = flags;
    idx->timestamp = (uint32_t)get_timestamp();
    
    bundle->header.last_modified = get_timestamp();
    bundle->is_dirty = 1;
    
    return 1;
}

AOBUNDLE2_API int32_t aobundle2_append_chunks_batch(
    aobundle2_t* bundle,
    int32_t zoom,
    const int32_t* indices,
    const uint8_t** jpeg_data,
    const uint32_t* sizes,
    const uint16_t* flags,
    int32_t count
) {
    if (!bundle || !bundle->is_writable || !indices || !jpeg_data || !sizes || count <= 0) {
        return 0;
    }
    
    int32_t success_count = 0;
    for (int32_t i = 0; i < count; i++) {
        uint16_t chunk_flags = flags ? flags[i] : AOBUNDLE2_CHUNK_VALID;
        if (aobundle2_append_chunk(bundle, zoom, indices[i], jpeg_data[i], sizes[i], chunk_flags)) {
            success_count++;
        }
    }
    
    return success_count;
}

AOBUNDLE2_API int32_t aobundle2_expand_zoom(
    aobundle2_t* bundle,
    int32_t new_zoom,
    int32_t chunks_per_side
) {
    if (!bundle || !bundle->is_writable) return 0;
    
    /* Check if zoom already exists */
    if (find_zoom_index(bundle, new_zoom) >= 0) return 0;
    
    /* Check limits */
    if (bundle->header.zoom_count >= AOBUNDLE2_MAX_ZOOM_LEVELS) return 0;
    
    int32_t chunk_count = chunks_per_side * chunks_per_side;
    if (chunk_count > AOBUNDLE2_MAX_CHUNKS_PER_ZOOM) return 0;
    
    /* This is a complex operation - requires rewriting the file with new zoom table */
    /* For now, mark as needing compaction which will reorganize the file */
    bundle->header.flags |= AOBUNDLE2_FLAG_COMPACTION_NEEDED | AOBUNDLE2_FLAG_MULTI_ZOOM;
    bundle->is_dirty = 1;
    
    /* TODO: Implement full zoom expansion by rewriting file */
    /* This would involve:
     * 1. Calculate new layout with additional zoom table entry and index section
     * 2. Create new file with expanded structure
     * 3. Copy existing data
     * 4. Atomic rename
     */
    
    return 0;  /* Not fully implemented yet */
}

AOBUNDLE2_API int32_t aobundle2_mark_missing(
    aobundle2_t* bundle,
    int32_t zoom,
    int32_t index
) {
    if (!bundle || !bundle->is_writable) return 0;
    
    int32_t zi = find_zoom_index(bundle, zoom);
    if (zi < 0) return 0;
    
    if (index < 0 || (uint32_t)index >= bundle->zoom_table[zi].chunk_count) return 0;
    
    aobundle2_chunk_index_t* idx = &bundle->chunk_indices[zi][index];
    
    /* Mark existing data as garbage */
    if (idx->size > 0) {
        bundle->header.garbage_bytes += idx->size;
    }
    
    idx->flags = AOBUNDLE2_CHUNK_MISSING;
    idx->size = 0;
    idx->timestamp = (uint32_t)get_timestamp();
    
    bundle->header.last_modified = get_timestamp();
    bundle->is_dirty = 1;
    
    return 1;
}

AOBUNDLE2_API int32_t aobundle2_flush(aobundle2_t* bundle) {
    if (!bundle || !bundle->is_writable || !bundle->is_dirty) return 1;
    
    /* Recalculate checksum */
    bundle->header.checksum = calc_header_checksum(&bundle->header);
    
    /* Write header back to file */
#ifdef AOPIPELINE_WINDOWS
    HANDLE hFile = (HANDLE)(intptr_t)bundle->fd;
    LARGE_INTEGER liDistanceToMove = {0};
    SetFilePointerEx(hFile, liDistanceToMove, NULL, FILE_BEGIN);
    
    DWORD bytes_written;
    if (!WriteFile(hFile, &bundle->header, sizeof(bundle->header), &bytes_written, NULL)) {
        return 0;
    }
    
    /* Write chunk indices for each zoom level */
    for (uint16_t i = 0; i < bundle->header.zoom_count; i++) {
        liDistanceToMove.QuadPart = bundle->zoom_table[i].index_offset;
        SetFilePointerEx(hFile, liDistanceToMove, NULL, FILE_BEGIN);
        
        size_t idx_size = bundle->zoom_table[i].chunk_count * sizeof(aobundle2_chunk_index_t);
        if (!WriteFile(hFile, bundle->chunk_indices[i], (DWORD)idx_size, &bytes_written, NULL)) {
            return 0;
        }
    }
    
    FlushFileBuffers(hFile);
#else
    lseek(bundle->fd, 0, SEEK_SET);
    if (write(bundle->fd, &bundle->header, sizeof(bundle->header)) != sizeof(bundle->header)) {
        return 0;
    }
    
    /* Write chunk indices for each zoom level */
    for (uint16_t i = 0; i < bundle->header.zoom_count; i++) {
        lseek(bundle->fd, bundle->zoom_table[i].index_offset, SEEK_SET);
        
        size_t idx_size = bundle->zoom_table[i].chunk_count * sizeof(aobundle2_chunk_index_t);
        if (write(bundle->fd, bundle->chunk_indices[i], idx_size) != (ssize_t)idx_size) {
            return 0;
        }
    }
    
    fsync(bundle->fd);
#endif
    
    bundle->is_dirty = 0;
    return 1;
}

/* ============================================================================
 * Compaction Functions
 * ============================================================================ */

AOBUNDLE2_API float aobundle2_get_fragmentation(const char* path) {
    aobundle2_t bundle;
    if (!aobundle2_open(path, &bundle, 0)) return -1.0f;
    
    /* Calculate total valid data size */
    uint32_t valid_bytes = 0;
    for (uint16_t zi = 0; zi < bundle.header.zoom_count; zi++) {
        for (uint32_t i = 0; i < bundle.zoom_table[zi].chunk_count; i++) {
            const aobundle2_chunk_index_t* idx = &bundle.chunk_indices[zi][i];
            if (idx->size > 0 && !(idx->flags & AOBUNDLE2_CHUNK_GARBAGE)) {
                valid_bytes += idx->size;
            }
        }
    }
    
    float fragmentation = 0.0f;
    if (valid_bytes + bundle.header.garbage_bytes > 0) {
        fragmentation = (float)bundle.header.garbage_bytes / 
                       (float)(valid_bytes + bundle.header.garbage_bytes);
    }
    
    aobundle2_close(&bundle);
    return fragmentation;
}

AOBUNDLE2_API int32_t aobundle2_needs_compaction(const char* path, float threshold) {
    float frag = aobundle2_get_fragmentation(path);
    if (frag < 0) return -1;
    return frag >= threshold ? 1 : 0;
}

AOBUNDLE2_API int64_t aobundle2_compact(const char* path) {
    aobundle2_t bundle;
    if (!aobundle2_open(path, &bundle, 0)) return -1;
    
    if (bundle.header.garbage_bytes == 0) {
        aobundle2_close(&bundle);
        return 0;  /* No compaction needed */
    }
    
    /* Collect valid chunks for each zoom level */
    uint32_t total_valid_data = 0;
    for (uint16_t zi = 0; zi < bundle.header.zoom_count; zi++) {
        for (uint32_t i = 0; i < bundle.zoom_table[zi].chunk_count; i++) {
            const aobundle2_chunk_index_t* idx = &bundle.chunk_indices[zi][i];
            if (idx->size > 0 && !(idx->flags & AOBUNDLE2_CHUNK_GARBAGE)) {
                total_valid_data += idx->size;
            }
        }
    }
    
    /* Calculate new file layout */
    size_t maptype_padded = align_to(bundle.header.maptype_len, 8);
    size_t zoom_table_size = bundle.header.zoom_count * sizeof(aobundle2_zoom_entry_t);
    size_t index_size = bundle.header.total_chunks * sizeof(aobundle2_chunk_index_t);
    size_t data_offset = AOBUNDLE2_HEADER_SIZE + maptype_padded + zoom_table_size + index_size;
    size_t new_total_size = data_offset + total_valid_data;
    
    /* Allocate new buffer */
    uint8_t* new_buf = (uint8_t*)calloc(1, new_total_size);
    if (!new_buf) {
        aobundle2_close(&bundle);
        return -1;
    }
    
    /* Copy header with updated values */
    aobundle2_header_t* new_header = (aobundle2_header_t*)new_buf;
    *new_header = bundle.header;
    new_header->garbage_bytes = 0;
    new_header->flags &= ~AOBUNDLE2_FLAG_COMPACTION_NEEDED;
    new_header->data_section_offset = (uint32_t)data_offset;
    new_header->last_modified = get_timestamp();
    
    /* Copy maptype */
    memcpy(new_buf + AOBUNDLE2_HEADER_SIZE, bundle.maptype, bundle.header.maptype_len);
    
    /* Copy zoom table and rebuild indices with new offsets */
    aobundle2_zoom_entry_t* new_zt = (aobundle2_zoom_entry_t*)(new_buf + AOBUNDLE2_HEADER_SIZE + maptype_padded);
    uint32_t current_data_offset = 0;
    uint32_t current_index_offset = (uint32_t)(AOBUNDLE2_HEADER_SIZE + maptype_padded + zoom_table_size);
    
    for (uint16_t zi = 0; zi < bundle.header.zoom_count; zi++) {
        new_zt[zi] = bundle.zoom_table[zi];
        new_zt[zi].index_offset = current_index_offset;
        
        aobundle2_chunk_index_t* new_indices = (aobundle2_chunk_index_t*)(new_buf + current_index_offset);
        
        for (uint32_t i = 0; i < bundle.zoom_table[zi].chunk_count; i++) {
            const aobundle2_chunk_index_t* old_idx = &bundle.chunk_indices[zi][i];
            aobundle2_chunk_index_t* new_idx = &new_indices[i];
            
            if (old_idx->size > 0 && !(old_idx->flags & AOBUNDLE2_CHUNK_GARBAGE)) {
                /* Copy valid chunk data */
                memcpy(new_buf + data_offset + current_data_offset,
                       bundle.data + old_idx->data_offset,
                       old_idx->size);
                
                new_idx->data_offset = current_data_offset;
                new_idx->size = old_idx->size;
                new_idx->flags = old_idx->flags & ~AOBUNDLE2_CHUNK_GARBAGE;
                new_idx->quality = old_idx->quality;
                new_idx->timestamp = old_idx->timestamp;
                
                current_data_offset += old_idx->size;
            } else {
                /* Mark as missing */
                new_idx->data_offset = 0;
                new_idx->size = 0;
                new_idx->flags = AOBUNDLE2_CHUNK_MISSING;
                new_idx->quality = 0;
                new_idx->timestamp = old_idx->timestamp;
            }
        }
        
        current_index_offset += bundle.zoom_table[zi].chunk_count * sizeof(aobundle2_chunk_index_t);
    }
    
    /* Calculate header checksum */
    new_header->checksum = calc_header_checksum(new_header);
    
    int64_t bytes_reclaimed = bundle.header.garbage_bytes;
    
    aobundle2_close(&bundle);
    
    /* Write new file atomically */
    if (!atomic_write_file(path, new_buf, new_total_size)) {
        free(new_buf);
        return -1;
    }
    
    free(new_buf);
    return bytes_reclaimed;
}

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

AOBUNDLE2_API int32_t aobundle2_validate(const char* path) {
    aobundle2_t bundle;
    if (!aobundle2_open(path, &bundle, 0)) return 0;
    
    /* Basic validation passed if we got here */
    /* Could add more checks: data bounds, JPEG signature validation, etc. */
    
    aobundle2_close(&bundle);
    return 1;
}

AOBUNDLE2_API const char* aobundle2_version(void) {
    return "aobundle2 " AOBUNDLE2_VERSION_STR
#ifdef AOPIPELINE_WINDOWS
           " [Windows]"
#elif defined(AOPIPELINE_MACOS)
           " [macOS]"
#else
           " [Linux]"
#endif
    ;
}

/* ============================================================================
 * Atomic Consolidation (with file locking and optional source deletion)
 * ============================================================================ */

AOBUNDLE2_API int32_t aobundle2_consolidate_atomic(
    const char* bundle_path,
    int32_t tile_row,
    int32_t tile_col,
    const char* maptype,
    int32_t zoom,
    const uint8_t** jpeg_data,
    const uint32_t* jpeg_sizes,
    int32_t chunk_count,
    const char** source_paths,
    int32_t delete_sources,
    aobundle2_result_t* result
) {
    if (result) memset(result, 0, sizeof(aobundle2_result_t));
    
    if (!bundle_path || !maptype || chunk_count <= 0) {
        if (result) {
            safe_strcpy(result->error_msg, "Invalid arguments", sizeof(result->error_msg));
        }
        return 0;
    }
    
    /*
     * Strategy:
     * 1. If bundle exists, open it with file locking for update
     * 2. If bundle doesn't exist, create new bundle
     * 3. Write to temp file
     * 4. Atomic rename temp -> bundle
     * 5. Only then delete source files
     *
     * This ensures:
     * - No partial writes (atomic rename)
     * - No data loss (sources deleted only after commit)
     * - No race conditions (file locking)
     */
    
    int32_t success = 0;
    char lock_path[1024];
#ifdef AOPIPELINE_WINDOWS
    HANDLE hLockFile = INVALID_HANDLE_VALUE;
#else
    int32_t fd_lock = -1;
#endif
    
    /* Create lock file path (same as bundle but with .lock extension) */
    snprintf(lock_path, sizeof(lock_path), "%s.lock", bundle_path);
    
#ifdef AOPIPELINE_WINDOWS
    /* Create/open lock file with exclusive access */
    hLockFile = CreateFileA(
        lock_path,
        GENERIC_WRITE,
        0,  /* No sharing - exclusive access */
        NULL,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_DELETE_ON_CLOSE,
        NULL
    );
    
    if (hLockFile == INVALID_HANDLE_VALUE) {
        /* Could not acquire lock - another process is consolidating */
        if (result) {
            snprintf(result->error_msg, sizeof(result->error_msg),
                     "Could not acquire consolidation lock: %lu", GetLastError());
        }
        return 0;
    }
#else
    /* Unix: use flock on lock file */
    fd_lock = open(lock_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd_lock < 0) {
        if (result) {
            snprintf(result->error_msg, sizeof(result->error_msg),
                     "Could not create lock file: %s", lock_path);
        }
        return 0;
    }
    
    /* Try to acquire exclusive lock (non-blocking) */
    if (flock(fd_lock, LOCK_EX | LOCK_NB) != 0) {
        close(fd_lock);
        if (result) {
            safe_strcpy(result->error_msg, "Could not acquire consolidation lock (in use)", 
                       sizeof(result->error_msg));
        }
        return 0;
    }
#endif
    
    /* Check if bundle exists */
    FILE* test_file = fopen(bundle_path, "rb");
    int32_t bundle_exists = (test_file != NULL);
    if (test_file) fclose(test_file);
    
    if (bundle_exists) {
        /* Update existing bundle - merge with new data at zoom level */
        /* For simplicity, we read existing bundle, add/update zoom level, and rewrite */
        aobundle2_t existing;
        if (!aobundle2_open(bundle_path, &existing, 0)) {
            if (result) {
                safe_strcpy(result->error_msg, "Could not open existing bundle", 
                           sizeof(result->error_msg));
            }
            goto cleanup;
        }
        
        /* Check if zoom level exists */
        int32_t zoom_idx = find_zoom_index(&existing, zoom);
        
        if (zoom_idx >= 0) {
            /* Zoom exists - need to merge/update chunks */
            /* For now, just replace (could implement smarter merging later) */
        }
        
        /* Close existing bundle - we'll rewrite it */
        aobundle2_close(&existing);
        
        /* Create new bundle with updated data */
        /* This is a simplified approach - production code might want to 
         * preserve other zoom levels and just update the target zoom */
        success = aobundle2_create_from_data(
            tile_row, tile_col, maptype, zoom,
            jpeg_data, jpeg_sizes, chunk_count,
            bundle_path, result
        );
    } else {
        /* Create new bundle */
        success = aobundle2_create_from_data(
            tile_row, tile_col, maptype, zoom,
            jpeg_data, jpeg_sizes, chunk_count,
            bundle_path, result
        );
    }
    
    /* Delete source files only after successful bundle write */
    if (success && delete_sources && source_paths) {
        int32_t deleted = 0;
        for (int32_t i = 0; i < chunk_count; i++) {
            if (source_paths[i] && source_paths[i][0] != '\0') {
                /* Try to delete source file */
#ifdef AOPIPELINE_WINDOWS
                if (DeleteFileA(source_paths[i])) {
                    deleted++;
                }
#else
                if (unlink(source_paths[i]) == 0) {
                    deleted++;
                }
#endif
                /* Ignore deletion failures - file may have been already deleted
                 * or is still in use. Cleanup will happen later. */
            }
        }
        
        /* Update result with deletion count (reusing chunks_missing as deleted count) */
        if (result) {
            result->chunks_missing = deleted;  /* Repurpose for deleted count */
        }
    }
    
cleanup:
#ifdef AOPIPELINE_WINDOWS
    if (hLockFile != INVALID_HANDLE_VALUE) {
        CloseHandle(hLockFile);  /* FILE_FLAG_DELETE_ON_CLOSE removes lock file */
    }
#else
    if (fd_lock >= 0) {
        flock(fd_lock, LOCK_UN);  /* Release lock */
        close(fd_lock);
        unlink(lock_path);  /* Remove lock file */
    }
#endif
    
    return success;
}