/**
 * aohttp.h - Native HTTP Client Pool for AutoOrtho
 * 
 * Provides high-performance parallel HTTP downloads using libcurl
 * multi-interface:
 * - True concurrent downloads without Python GIL
 * - Connection pooling and keep-alive
 * - HTTP/2 multiplexing when available
 * - Callback-based completion notification
 * 
 * This module enables downloading many chunks in parallel while
 * other native processing (decode, compress) continues.
 */

#ifndef AOHTTP_H
#define AOHTTP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Export/Import macros */
#ifdef _WIN32
  #ifdef AOPIPELINE_EXPORTS
    #define AOHTTP_API __declspec(dllexport)
  #else
    #define AOHTTP_API __declspec(dllimport)
  #endif
#else
  #define AOHTTP_API __attribute__((__visibility__("default")))
#endif

/**
 * Completion callback function type.
 * 
 * Called when an HTTP request completes (success or failure).
 * 
 * @param request_id    Unique request identifier
 * @param status_code   HTTP status code (200 = success, 0 = connection error)
 * @param data          Response body data (NULL on error)
 * @param length        Length of response data
 * @param user_data     User-provided context pointer
 */
typedef void (*aohttp_callback_fn)(
    int32_t request_id,
    int32_t status_code,
    const uint8_t* data,
    uint32_t length,
    void* user_data
);

/**
 * HTTP request structure.
 */
typedef struct {
    int32_t request_id;         /**< Unique identifier for this request */
    const char* url;            /**< URL to fetch */
    void* user_data;            /**< User context passed to callback */
} aohttp_request_t;

/**
 * HTTP response structure (for synchronous API).
 */
typedef struct {
    int32_t status_code;        /**< HTTP status code (0 = error) */
    uint8_t* data;              /**< Response body (caller must free) */
    uint32_t length;            /**< Length of data */
    char error[128];            /**< Error message if status_code <= 0 */
} aohttp_response_t;

/**
 * Initialize the HTTP client pool.
 * 
 * Must be called before any other aohttp functions.
 * 
 * @param max_connections   Maximum concurrent connections (default: 32)
 * @return 1 on success, 0 if libcurl not available
 */
AOHTTP_API int32_t aohttp_init(int32_t max_connections);

/**
 * Shutdown the HTTP client pool.
 * 
 * Cancels any pending requests and frees resources.
 */
AOHTTP_API void aohttp_shutdown(void);

/**
 * Check if HTTP client is initialized and available.
 * 
 * @return 1 if available, 0 if not
 */
AOHTTP_API int32_t aohttp_is_available(void);

/**
 * Submit a batch of HTTP requests asynchronously.
 * 
 * Requests are processed in parallel using libcurl multi-interface.
 * The callback is invoked for each completed request.
 * 
 * @param requests      Array of request structures
 * @param count         Number of requests
 * @param callback      Completion callback function
 * 
 * @return Number of requests successfully submitted
 */
AOHTTP_API int32_t aohttp_submit_batch(
    const aohttp_request_t* requests,
    int32_t count,
    aohttp_callback_fn callback
);

/**
 * Process pending I/O (non-blocking).
 * 
 * Drives the libcurl multi-interface, processing any completed
 * transfers and invoking callbacks.
 * 
 * @param timeout_ms    Maximum time to wait for I/O (0 = non-blocking)
 * @return Number of completed transfers processed
 */
AOHTTP_API int32_t aohttp_poll(int32_t timeout_ms);

/**
 * Get number of active/pending requests.
 * 
 * @return Number of requests currently in flight
 */
AOHTTP_API int32_t aohttp_pending_count(void);

/**
 * Fetch a single URL synchronously.
 * 
 * Convenience function for simple single requests.
 * Blocks until complete.
 * 
 * @param url       URL to fetch
 * @param response  Output response structure
 * @param timeout_ms Maximum time to wait (0 = default)
 * 
 * @return 1 on success (status_code 200-299), 0 on failure
 */
AOHTTP_API int32_t aohttp_get_sync(
    const char* url,
    aohttp_response_t* response,
    int32_t timeout_ms
);

/**
 * Fetch multiple URLs synchronously.
 * 
 * Downloads all URLs in parallel and waits for completion.
 * 
 * @param urls          Array of URL strings
 * @param count         Number of URLs
 * @param responses     Pre-allocated array of response structures
 * @param timeout_ms    Maximum time per request
 * 
 * @return Number of successful downloads
 */
AOHTTP_API int32_t aohttp_get_batch_sync(
    const char** urls,
    int32_t count,
    aohttp_response_t* responses,
    int32_t timeout_ms
);

/**
 * Free a response structure.
 * 
 * Frees the data buffer allocated by aohttp_get_sync.
 * 
 * @param response  Response to free
 */
AOHTTP_API void aohttp_response_free(aohttp_response_t* response);

/**
 * Free multiple response structures.
 * 
 * @param responses Array of responses
 * @param count     Number of responses
 */
AOHTTP_API void aohttp_response_batch_free(
    aohttp_response_t* responses,
    int32_t count
);

/**
 * Set a custom User-Agent header.
 * 
 * @param user_agent    User-Agent string (copied internally)
 */
AOHTTP_API void aohttp_set_user_agent(const char* user_agent);

/**
 * Enable/disable SSL certificate verification.
 * 
 * @param verify    1 = verify certificates (default), 0 = skip verification
 */
AOHTTP_API void aohttp_set_ssl_verify(int32_t verify);

/**
 * Get version information.
 * 
 * @return Static string with version info
 */
AOHTTP_API const char* aohttp_version(void);

#ifdef __cplusplus
}
#endif

#endif /* AOHTTP_H */

