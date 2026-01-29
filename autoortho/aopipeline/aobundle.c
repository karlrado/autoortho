/**
 * aobundle.c - Cache Bundle Implementation for AutoOrtho
 * 
 * Implements the bundle format for consolidating cache files.
 */

#include "aobundle.h"
#include "aodds.h"
#include "internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef AOPIPELINE_WINDOWS
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif

#define AOBUNDLE_VERSION_STR "1.0.0"

/* Helper: Align value to boundary */
static inline uint32_t align_to(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

/* Helper: Read entire file */
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

AOBUNDLE_API int32_t aobundle_create(
    const char* cache_dir,
    int32_t tile_col,
    int32_t tile_row,
    const char* maptype,
    int32_t zoom,
    int32_t chunks_per_side,
    const char* output_path
) {
    if (!cache_dir || !maptype || !output_path || chunks_per_side <= 0) {
        return 0;
    }
    
    int32_t chunk_count = chunks_per_side * chunks_per_side;
    if (chunk_count > AOBUNDLE_MAX_CHUNKS) {
        return 0;
    }
    
    /* Read all JPEG files */
    uint8_t** jpeg_data = (uint8_t**)calloc(chunk_count, sizeof(uint8_t*));
    uint32_t* jpeg_sizes = (uint32_t*)calloc(chunk_count, sizeof(uint32_t));
    
    if (!jpeg_data || !jpeg_sizes) {
        free(jpeg_data);
        free(jpeg_sizes);
        return 0;
    }
    
    uint32_t total_data_size = 0;
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
        }
    }
    
    /* Calculate sizes */
    size_t maptype_len = strlen(maptype);
    if (maptype_len > AOBUNDLE_MAX_MAPTYPE) {
        maptype_len = AOBUNDLE_MAX_MAPTYPE;
    }
    size_t maptype_padded = align_to(maptype_len, 8);
    
    size_t header_size = sizeof(aobundle_header_t);
    size_t index_size = chunk_count * sizeof(aobundle_index_t);
    size_t total_size = header_size + maptype_padded + index_size + total_data_size;
    
    /* Allocate bundle buffer */
    uint8_t* bundle_buf = (uint8_t*)calloc(1, total_size);
    if (!bundle_buf) {
        for (int32_t i = 0; i < chunk_count; i++) free(jpeg_data[i]);
        free(jpeg_data);
        free(jpeg_sizes);
        return 0;
    }
    
    /* Write header */
    aobundle_header_t* header = (aobundle_header_t*)bundle_buf;
    header->magic = AOBUNDLE_MAGIC;
    header->version = AOBUNDLE_VERSION;
    header->chunk_count = (uint16_t)chunk_count;
    header->tile_col = tile_col;
    header->tile_row = tile_row;
    header->zoom = (uint16_t)zoom;
    header->maptype_len = (uint16_t)maptype_len;
    memset(header->reserved, 0, sizeof(header->reserved));
    
    /* Write maptype */
    memcpy(bundle_buf + header_size, maptype, maptype_len);
    
    /* Write index and data */
    aobundle_index_t* index = (aobundle_index_t*)(bundle_buf + header_size + maptype_padded);
    uint8_t* data_section = bundle_buf + header_size + maptype_padded + index_size;
    uint32_t data_offset = 0;
    
    for (int32_t i = 0; i < chunk_count; i++) {
        if (jpeg_data[i] && jpeg_sizes[i] > 0) {
            index[i].offset = data_offset;
            index[i].size = jpeg_sizes[i];
            memcpy(data_section + data_offset, jpeg_data[i], jpeg_sizes[i]);
            data_offset += jpeg_sizes[i];
        } else {
            index[i].offset = 0;
            index[i].size = 0;
        }
        free(jpeg_data[i]);
    }
    free(jpeg_data);
    free(jpeg_sizes);
    
    /* Write atomically: temp file + rename */
    char temp_path[4096];
    snprintf(temp_path, sizeof(temp_path), "%s.tmp", output_path);
    
    FILE* f = fopen(temp_path, "wb");
    if (!f) {
        free(bundle_buf);
        return 0;
    }
    
    size_t written = fwrite(bundle_buf, 1, total_size, f);
    fclose(f);
    free(bundle_buf);
    
    if (written != total_size) {
        remove(temp_path);
        return 0;
    }
    
    /* Atomic rename */
#ifdef AOPIPELINE_WINDOWS
    if (!MoveFileExA(temp_path, output_path, MOVEFILE_REPLACE_EXISTING)) {
        remove(temp_path);
        return 0;
    }
#else
    if (rename(temp_path, output_path) != 0) {
        remove(temp_path);
        return 0;
    }
#endif
    
    return 1;
}

AOBUNDLE_API int32_t aobundle_open(
    const char* path,
    aobundle_t* bundle,
    int32_t use_mmap
) {
    if (!path || !bundle) return 0;
    
    memset(bundle, 0, sizeof(aobundle_t));
    
#ifndef AOPIPELINE_WINDOWS
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
        aobundle_header_t* hdr = (aobundle_header_t*)mapped;
        if (hdr->magic != AOBUNDLE_MAGIC) {
            munmap(mapped, st.st_size);
            return 0;
        }
        
        bundle->header = *hdr;
        
        /* Point to maptype */
        size_t maptype_offset = sizeof(aobundle_header_t);
        size_t maptype_len = hdr->maptype_len;
        if (maptype_len > AOBUNDLE_MAX_MAPTYPE - 1) {
            maptype_len = AOBUNDLE_MAX_MAPTYPE - 1;
        }
        memcpy(bundle->maptype, (uint8_t*)mapped + maptype_offset, maptype_len);
        bundle->maptype[maptype_len] = '\0';
        
        /* Point to index */
        size_t maptype_padded = align_to(hdr->maptype_len, 8);
        size_t index_offset = maptype_offset + maptype_padded;
        bundle->index = (aobundle_index_t*)((uint8_t*)mapped + index_offset);
        
        /* Point to data */
        size_t index_size = hdr->chunk_count * sizeof(aobundle_index_t);
        size_t data_offset = index_offset + index_size;
        bundle->data = (uint8_t*)mapped + data_offset;
        bundle->data_size = st.st_size - data_offset;
        
        return 1;
    }
#endif
    
    /* Regular file read */
    uint32_t file_size;
    uint8_t* file_data = read_file_contents(path, &file_size);
    if (!file_data) return 0;
    
    if (file_size < sizeof(aobundle_header_t)) {
        free(file_data);
        return 0;
    }
    
    aobundle_header_t* hdr = (aobundle_header_t*)file_data;
    if (hdr->magic != AOBUNDLE_MAGIC) {
        free(file_data);
        return 0;
    }
    
    bundle->header = *hdr;
    
    /* Copy maptype */
    size_t maptype_offset = sizeof(aobundle_header_t);
    size_t maptype_len = hdr->maptype_len;
    if (maptype_len > AOBUNDLE_MAX_MAPTYPE - 1) {
        maptype_len = AOBUNDLE_MAX_MAPTYPE - 1;
    }
    memcpy(bundle->maptype, file_data + maptype_offset, maptype_len);
    bundle->maptype[maptype_len] = '\0';
    
    /* Allocate and copy index */
    size_t maptype_padded = align_to(hdr->maptype_len, 8);
    size_t index_offset = maptype_offset + maptype_padded;
    size_t index_size = hdr->chunk_count * sizeof(aobundle_index_t);
    
    bundle->index = (aobundle_index_t*)malloc(index_size);
    if (!bundle->index) {
        free(file_data);
        return 0;
    }
    memcpy(bundle->index, file_data + index_offset, index_size);
    
    /* Allocate and copy data */
    size_t data_offset = index_offset + index_size;
    bundle->data_size = file_size - data_offset;
    
    bundle->data = (uint8_t*)malloc(bundle->data_size);
    if (!bundle->data) {
        free(bundle->index);
        free(file_data);
        return 0;
    }
    memcpy(bundle->data, file_data + data_offset, bundle->data_size);
    
    free(file_data);
    return 1;
}

AOBUNDLE_API void aobundle_close(aobundle_t* bundle) {
    if (!bundle) return;
    
#ifndef AOPIPELINE_WINDOWS
    if (bundle->mmap_base) {
        munmap(bundle->mmap_base, bundle->mmap_size);
        bundle->mmap_base = NULL;
        bundle->mmap_size = 0;
        /* Index and data point into mmap, so don't free */
        bundle->index = NULL;
        bundle->data = NULL;
    } else
#endif
    {
        free(bundle->index);
        free(bundle->data);
    }
    
    memset(bundle, 0, sizeof(aobundle_t));
}

AOBUNDLE_API int32_t aobundle_get_chunk(
    const aobundle_t* bundle,
    int32_t index,
    const uint8_t** data,
    uint32_t* size
) {
    if (!bundle || !data || !size) return 0;
    if (index < 0 || index >= bundle->header.chunk_count) return 0;
    
    *size = bundle->index[index].size;
    if (*size == 0) {
        *data = NULL;
        return 0;  /* Missing chunk */
    }
    
    *data = bundle->data + bundle->index[index].offset;
    return 1;
}

AOBUNDLE_API int32_t aobundle_get_all_chunks(
    const aobundle_t* bundle,
    const uint8_t** jpeg_data,
    uint32_t* jpeg_sizes,
    int32_t max_chunks
) {
    if (!bundle || !jpeg_data || !jpeg_sizes) return 0;
    
    int32_t valid_count = 0;
    int32_t count = bundle->header.chunk_count;
    if (count > max_chunks) count = max_chunks;
    
    for (int32_t i = 0; i < count; i++) {
        jpeg_sizes[i] = bundle->index[i].size;
        if (jpeg_sizes[i] > 0) {
            jpeg_data[i] = bundle->data + bundle->index[i].offset;
            valid_count++;
        } else {
            jpeg_data[i] = NULL;
        }
    }
    
    return valid_count;
}

AOBUNDLE_API int32_t aobundle_build_dds(
    const char* bundle_path,
    int32_t format,
    uint8_t missing_color[3],
    uint8_t* dds_output,
    uint32_t output_size,
    uint32_t* bytes_written
) {
    if (!bundle_path || !dds_output || !bytes_written) return 0;
    
    /* Open bundle with mmap for fastest access */
    aobundle_t bundle;
#ifdef AOPIPELINE_WINDOWS
    if (!aobundle_open(bundle_path, &bundle, 0)) return 0;
#else
    if (!aobundle_open(bundle_path, &bundle, 1)) return 0;
#endif
    
    int32_t chunk_count = bundle.header.chunk_count;
    
    /* Allocate arrays for jpeg pointers */
    const uint8_t** jpeg_ptrs = (const uint8_t**)malloc(chunk_count * sizeof(uint8_t*));
    uint32_t* jpeg_sizes = (uint32_t*)malloc(chunk_count * sizeof(uint32_t));
    
    if (!jpeg_ptrs || !jpeg_sizes) {
        free(jpeg_ptrs);
        free(jpeg_sizes);
        aobundle_close(&bundle);
        return 0;
    }
    
    /* Get all chunk pointers */
    aobundle_get_all_chunks(&bundle, jpeg_ptrs, jpeg_sizes, chunk_count);
    
    /* Build DDS */
    int32_t result = aodds_build_from_jpegs(
        jpeg_ptrs,
        jpeg_sizes,
        chunk_count,
        format,
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
    aobundle_close(&bundle);
    
    return result;
}

AOBUNDLE_API const char* aobundle_version(void) {
    return "aobundle " AOBUNDLE_VERSION_STR
#ifdef AOPIPELINE_WINDOWS
           " [Windows]"
#elif defined(AOPIPELINE_MACOS)
           " [macOS]"
#else
           " [Linux]"
#endif
    ;
}

