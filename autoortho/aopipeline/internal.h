/**
 * internal.h - Shared internal definitions for aopipeline
 * 
 * This header is NOT part of the public API and should only be
 * included by aopipeline source files.
 */

#ifndef AOPIPELINE_INTERNAL_H
#define AOPIPELINE_INTERNAL_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Platform detection */
#ifdef _WIN32
  #define AOPIPELINE_WINDOWS 1
  #include <windows.h>
#else
  #define AOPIPELINE_POSIX 1
  #include <sys/stat.h>
  #include <fcntl.h>
  #include <unistd.h>
  #include <errno.h>
  #include <pthread.h>
  #ifdef __APPLE__
    #define AOPIPELINE_MACOS 1
    #include <sys/mman.h>
  #else
    #define AOPIPELINE_LINUX 1
    #include <sys/mman.h>
  #endif
#endif

/* OpenMP availability */
#ifdef _OPENMP
  #include <omp.h>
  #define AOPIPELINE_HAS_OPENMP 1
#else
  #define AOPIPELINE_HAS_OPENMP 0
  /* Stub macros for when OpenMP is not available */
  #define omp_get_thread_num() 0
  #define omp_get_num_threads() 1
  #define omp_set_num_threads(n) ((void)0)
#endif

/* Export macros */
#ifdef _WIN32
  #ifdef AOPIPELINE_EXPORTS
    #define AOPIPELINE_EXPORT __declspec(dllexport)
  #else
    #define AOPIPELINE_EXPORT __declspec(dllimport)
  #endif
#else
  #define AOPIPELINE_EXPORT __attribute__((__visibility__("default")))
#endif

/* Utility macros */
#define AOPIPELINE_MIN(a, b) ((a) < (b) ? (a) : (b))
#define AOPIPELINE_MAX(a, b) ((a) > (b) ? (a) : (b))

/* JPEG signature check */
#define JPEG_SIGNATURE_VALID(data, len) \
    ((len) >= 3 && (data)[0] == 0xFF && (data)[1] == 0xD8 && (data)[2] == 0xFF)

/* Default chunk size for tile images */
#define CHUNK_WIDTH  256
#define CHUNK_HEIGHT 256
#define CHUNK_CHANNELS 4
#define CHUNK_BUFFER_SIZE (CHUNK_WIDTH * CHUNK_HEIGHT * CHUNK_CHANNELS)

/* Safe string copy with null termination */
static inline void safe_strcpy(char* dest, const char* src, size_t dest_size) {
    if (dest_size == 0) return;
    size_t src_len = strlen(src);
    size_t copy_len = (src_len < dest_size - 1) ? src_len : dest_size - 1;
    memcpy(dest, src, copy_len);
    dest[copy_len] = '\0';
}

/* Thread-local error message buffer size */
#define ERROR_MSG_SIZE 256

/* ============================================================================
 * Thread-Local Storage (TLS) for Persistent Resources
 * ============================================================================
 * Used for maintaining per-thread resources like TurboJPEG handles that are
 * expensive to create/destroy. TLS ensures each OpenMP thread has its own
 * instance without synchronization overhead.
 */
#if defined(__GNUC__)
  /* GCC/Clang/MinGW - use __thread */
  #define TLS_VAR __thread
#elif defined(_MSC_VER)
  /* MSVC - use __declspec(thread) */
  #define TLS_VAR __declspec(thread)
#else
  #define TLS_VAR  /* Fallback: no TLS, will use array indexing */
#endif

/* Maximum OpenMP threads we support for persistent resource pools */
#define MAX_OMP_THREADS 64

/* Note: aopipeline_stats_t is defined in aopipeline.h */

#endif /* AOPIPELINE_INTERNAL_H */

