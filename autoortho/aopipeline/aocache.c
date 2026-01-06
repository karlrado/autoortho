/**
 * aocache.c - Native Parallel Cache I/O Implementation
 * 
 * Provides high-performance batch file reading with OpenMP parallelism.
 * Supports both memory-mapped I/O for large files and standard reads
 * for smaller files.
 */

#include "aocache.h"
#include "internal.h"
#include <stdio.h>

/* Version string */
#define AOCACHE_VERSION "1.0.0"

/* Threshold for using mmap vs standard read (64KB) */
#define MMAP_THRESHOLD (64 * 1024)

/* Maximum reasonable file size for cache files (100MB) */
#define MAX_CACHE_FILE_SIZE (100 * 1024 * 1024)

/*============================================================================
 * Platform-specific file operations
 *============================================================================*/

#ifdef AOPIPELINE_WINDOWS

/**
 * Windows implementation of file reading using CreateFile/ReadFile
 */
static int read_file_win32(const char* path, uint8_t** out_data, 
                           uint32_t* out_len, char* error_buf) {
    HANDLE hFile = CreateFileA(
        path,
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
        NULL
    );
    
    if (hFile == INVALID_HANDLE_VALUE) {
        DWORD err = GetLastError();
        if (err == ERROR_FILE_NOT_FOUND || err == ERROR_PATH_NOT_FOUND) {
            safe_strcpy(error_buf, "File not found", 64);
        } else if (err == ERROR_SHARING_VIOLATION) {
            safe_strcpy(error_buf, "File locked", 64);
        } else {
            snprintf(error_buf, 64, "Open failed: %lu", err);
        }
        return 0;
    }
    
    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(hFile, &file_size)) {
        CloseHandle(hFile);
        safe_strcpy(error_buf, "GetFileSizeEx failed", 64);
        return 0;
    }
    
    if (file_size.QuadPart > MAX_CACHE_FILE_SIZE) {
        CloseHandle(hFile);
        safe_strcpy(error_buf, "File too large", 64);
        return 0;
    }
    
    if (file_size.QuadPart == 0) {
        CloseHandle(hFile);
        safe_strcpy(error_buf, "File is empty", 64);
        return 0;
    }
    
    uint32_t size = (uint32_t)file_size.QuadPart;
    uint8_t* buffer = (uint8_t*)malloc(size);
    if (!buffer) {
        CloseHandle(hFile);
        safe_strcpy(error_buf, "Out of memory", 64);
        return 0;
    }
    
    DWORD bytes_read;
    if (!ReadFile(hFile, buffer, size, &bytes_read, NULL) || bytes_read != size) {
        free(buffer);
        CloseHandle(hFile);
        safe_strcpy(error_buf, "ReadFile failed", 64);
        return 0;
    }
    
    CloseHandle(hFile);
    *out_data = buffer;
    *out_len = size;
    return 1;
}

static int file_exists_win32(const char* path) {
    DWORD attrs = GetFileAttributesA(path);
    return (attrs != INVALID_FILE_ATTRIBUTES && 
            !(attrs & FILE_ATTRIBUTE_DIRECTORY));
}

static int64_t file_size_win32(const char* path) {
    WIN32_FILE_ATTRIBUTE_DATA fad;
    if (!GetFileAttributesExA(path, GetFileExInfoStandard, &fad)) {
        return -1;
    }
    LARGE_INTEGER size;
    size.HighPart = fad.nFileSizeHigh;
    size.LowPart = fad.nFileSizeLow;
    return size.QuadPart;
}

static int write_file_atomic_win32(const char* path, const uint8_t* data, 
                                   uint32_t length) {
    /* Generate temp file path */
    char temp_path[MAX_PATH];
    snprintf(temp_path, MAX_PATH, "%s.tmp.%lu", path, GetCurrentThreadId());
    
    HANDLE hFile = CreateFileA(
        temp_path,
        GENERIC_WRITE,
        0,
        NULL,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );
    
    if (hFile == INVALID_HANDLE_VALUE) {
        return 0;
    }
    
    DWORD written;
    if (!WriteFile(hFile, data, length, &written, NULL) || written != length) {
        CloseHandle(hFile);
        DeleteFileA(temp_path);
        return 0;
    }
    
    CloseHandle(hFile);
    
    /* Atomic rename (MoveFileEx with REPLACE_EXISTING) */
    if (!MoveFileExA(temp_path, path, MOVEFILE_REPLACE_EXISTING)) {
        DeleteFileA(temp_path);
        return 0;
    }
    
    return 1;
}

#else /* POSIX */

/**
 * POSIX implementation using mmap for large files, read() for small files
 */
static int read_file_posix(const char* path, uint8_t** out_data, 
                           uint32_t* out_len, char* error_buf) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        if (errno == ENOENT) {
            safe_strcpy(error_buf, "File not found", 64);
        } else if (errno == EACCES) {
            safe_strcpy(error_buf, "Permission denied", 64);
        } else {
            snprintf(error_buf, 64, "Open failed: %d", errno);
        }
        return 0;
    }
    
    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        safe_strcpy(error_buf, "fstat failed", 64);
        return 0;
    }
    
    if (!S_ISREG(st.st_mode)) {
        close(fd);
        safe_strcpy(error_buf, "Not a regular file", 64);
        return 0;
    }
    
    if (st.st_size > MAX_CACHE_FILE_SIZE) {
        close(fd);
        safe_strcpy(error_buf, "File too large", 64);
        return 0;
    }
    
    if (st.st_size == 0) {
        close(fd);
        safe_strcpy(error_buf, "File is empty", 64);
        return 0;
    }
    
    uint32_t size = (uint32_t)st.st_size;
    uint8_t* buffer = (uint8_t*)malloc(size);
    if (!buffer) {
        close(fd);
        safe_strcpy(error_buf, "Out of memory", 64);
        return 0;
    }
    
    /* Use mmap for large files, direct read for small files */
    if (size >= MMAP_THRESHOLD) {
        void* mapped = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped == MAP_FAILED) {
            /* Fallback to read() if mmap fails */
            goto do_read;
        }
        
        /* Copy from mmap to our buffer */
        memcpy(buffer, mapped, size);
        munmap(mapped, size);
    } else {
do_read:
        /* Standard read for small files */
        size_t total_read = 0;
        while (total_read < size) {
            ssize_t n = read(fd, buffer + total_read, size - total_read);
            if (n <= 0) {
                if (n < 0 && errno == EINTR) continue;
                free(buffer);
                close(fd);
                safe_strcpy(error_buf, "Read error", 64);
                return 0;
            }
            total_read += n;
        }
    }
    
    close(fd);
    *out_data = buffer;
    *out_len = size;
    return 1;
}

static int file_exists_posix(const char* path) {
    struct stat st;
    return (stat(path, &st) == 0 && S_ISREG(st.st_mode));
}

static int64_t file_size_posix(const char* path) {
    struct stat st;
    if (stat(path, &st) != 0) {
        return -1;
    }
    return st.st_size;
}

static int write_file_atomic_posix(const char* path, const uint8_t* data, 
                                   uint32_t length) {
    /* Generate temp file path with thread ID for uniqueness */
    char temp_path[4096];
#ifdef _OPENMP
    int tid = omp_get_thread_num();
#else
    int tid = 0;
#endif
    snprintf(temp_path, sizeof(temp_path), "%s.tmp.%d", path, tid);
    
    int fd = open(temp_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        return 0;
    }
    
    size_t written = 0;
    while (written < length) {
        ssize_t n = write(fd, data + written, length - written);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            close(fd);
            unlink(temp_path);
            return 0;
        }
        written += n;
    }
    
    /* Ensure data is on disk before rename */
    fsync(fd);
    close(fd);
    
    /* Atomic rename */
    if (rename(temp_path, path) != 0) {
        unlink(temp_path);
        return 0;
    }
    
    return 1;
}

#endif /* Platform selection */

/*============================================================================
 * Cross-platform wrapper functions
 *============================================================================*/

static int read_file_impl(const char* path, uint8_t** out_data, 
                          uint32_t* out_len, char* error_buf) {
#ifdef AOPIPELINE_WINDOWS
    return read_file_win32(path, out_data, out_len, error_buf);
#else
    return read_file_posix(path, out_data, out_len, error_buf);
#endif
}

/*============================================================================
 * Public API Implementation
 *============================================================================*/

AOCACHE_API int32_t aocache_batch_read(
    const char** paths,
    int32_t count,
    aocache_result_t* results,
    int32_t max_threads
) {
    if (!paths || !results || count <= 0) {
        return 0;
    }
    
    int32_t success_count = 0;
    
#if AOPIPELINE_HAS_OPENMP
    if (max_threads > 0) {
        omp_set_num_threads(max_threads);
    }
#endif
    
#pragma omp parallel for reduction(+:success_count) schedule(dynamic, 4)
    for (int32_t i = 0; i < count; i++) {
        results[i].data = NULL;
        results[i].length = 0;
        results[i].success = 0;
        results[i].error[0] = '\0';
        
        if (!paths[i] || paths[i][0] == '\0') {
            safe_strcpy(results[i].error, "Empty path", 64);
            continue;
        }
        
        uint8_t* data = NULL;
        uint32_t len = 0;
        
        if (!read_file_impl(paths[i], &data, &len, results[i].error)) {
            continue;
        }
        
        /* Validate JPEG signature */
        if (!JPEG_SIGNATURE_VALID(data, len)) {
            safe_strcpy(results[i].error, "Invalid JPEG signature", 64);
            free(data);
            continue;
        }
        
        results[i].data = data;
        results[i].length = len;
        results[i].success = 1;
        success_count++;
    }
    
    return success_count;
}

AOCACHE_API int32_t aocache_batch_read_raw(
    const char** paths,
    int32_t count,
    aocache_result_t* results,
    int32_t max_threads
) {
    if (!paths || !results || count <= 0) {
        return 0;
    }
    
    int32_t success_count = 0;
    
#if AOPIPELINE_HAS_OPENMP
    if (max_threads > 0) {
        omp_set_num_threads(max_threads);
    }
#endif
    
#pragma omp parallel for reduction(+:success_count) schedule(dynamic, 4)
    for (int32_t i = 0; i < count; i++) {
        results[i].data = NULL;
        results[i].length = 0;
        results[i].success = 0;
        results[i].error[0] = '\0';
        
        if (!paths[i] || paths[i][0] == '\0') {
            safe_strcpy(results[i].error, "Empty path", 64);
            continue;
        }
        
        uint8_t* data = NULL;
        uint32_t len = 0;
        
        if (!read_file_impl(paths[i], &data, &len, results[i].error)) {
            continue;
        }
        
        results[i].data = data;
        results[i].length = len;
        results[i].success = 1;
        success_count++;
    }
    
    return success_count;
}

AOCACHE_API void aocache_batch_free(aocache_result_t* results, int32_t count) {
    if (!results) return;
    
    for (int32_t i = 0; i < count; i++) {
        if (results[i].data) {
            free(results[i].data);
            results[i].data = NULL;
        }
        results[i].length = 0;
        results[i].success = 0;
    }
}

AOCACHE_API int32_t aocache_validate_jpegs(
    const char** paths,
    int32_t count,
    int32_t* valid_flags,
    int32_t max_threads
) {
    if (!paths || !valid_flags || count <= 0) {
        return 0;
    }
    
    int32_t valid_count = 0;
    
#if AOPIPELINE_HAS_OPENMP
    if (max_threads > 0) {
        omp_set_num_threads(max_threads);
    }
#endif
    
#pragma omp parallel for reduction(+:valid_count) schedule(dynamic, 8)
    for (int32_t i = 0; i < count; i++) {
        valid_flags[i] = 0;
        
        if (!paths[i] || paths[i][0] == '\0') {
            continue;
        }
        
#ifdef AOPIPELINE_WINDOWS
        HANDLE hFile = CreateFileA(paths[i], GENERIC_READ, FILE_SHARE_READ,
                                   NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE) continue;
        
        uint8_t header[3];
        DWORD read;
        if (ReadFile(hFile, header, 3, &read, NULL) && read == 3) {
            if (JPEG_SIGNATURE_VALID(header, 3)) {
                valid_flags[i] = 1;
                valid_count++;
            }
        }
        CloseHandle(hFile);
#else
        int fd = open(paths[i], O_RDONLY);
        if (fd < 0) continue;
        
        uint8_t header[3];
        ssize_t n = read(fd, header, 3);
        close(fd);
        
        if (n == 3 && JPEG_SIGNATURE_VALID(header, 3)) {
            valid_flags[i] = 1;
            valid_count++;
        }
#endif
    }
    
    return valid_count;
}

AOCACHE_API int32_t aocache_file_exists(const char* path) {
    if (!path) return 0;
#ifdef AOPIPELINE_WINDOWS
    return file_exists_win32(path);
#else
    return file_exists_posix(path);
#endif
}

AOCACHE_API int64_t aocache_file_size(const char* path) {
    if (!path) return -1;
#ifdef AOPIPELINE_WINDOWS
    return file_size_win32(path);
#else
    return file_size_posix(path);
#endif
}

AOCACHE_API int32_t aocache_read_file(
    const char* path,
    uint8_t** out_data,
    uint32_t* out_len
) {
    if (!path || !out_data || !out_len) return 0;
    
    char error[64];
    return read_file_impl(path, out_data, out_len, error);
}

AOCACHE_API int32_t aocache_write_file_atomic(
    const char* path,
    const uint8_t* data,
    uint32_t length
) {
    if (!path || !data || length == 0) return 0;
    
#ifdef AOPIPELINE_WINDOWS
    return write_file_atomic_win32(path, data, length);
#else
    return write_file_atomic_posix(path, data, length);
#endif
}

AOCACHE_API const char* aocache_version(void) {
    return "aocache " AOCACHE_VERSION 
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

