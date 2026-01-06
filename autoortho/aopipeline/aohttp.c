/**
 * aohttp.c - Native HTTP Client Pool Implementation
 * 
 * Uses libcurl multi-interface for high-performance parallel downloads.
 */

#include "aohttp.h"
#include "internal.h"
#include <stdio.h>
#include <string.h>

/* Version string */
#define AOHTTP_VERSION "1.0.0"

/* Check for libcurl availability */
#ifdef AOPIPELINE_HAS_CURL
#include <curl/curl.h>
#else
/* Stub implementation when libcurl is not available */
#endif

/* Configuration */
#define DEFAULT_MAX_CONNECTIONS 32
#define DEFAULT_TIMEOUT_MS 30000
#define MAX_PENDING_REQUESTS 1024
#define INITIAL_BUFFER_SIZE 65536

/* Global settings (shared between curl and non-curl builds) */
static char user_agent[256] = "AutoOrtho/1.0";
static int32_t ssl_verify = 1;

/*============================================================================
 * Internal Structures
 *============================================================================*/

#ifdef AOPIPELINE_HAS_CURL

/* Buffer for receiving response data */
typedef struct {
    uint8_t* data;
    size_t size;
    size_t capacity;
} response_buffer_t;

/* Active transfer tracking */
typedef struct {
    CURL* easy;
    int32_t request_id;
    void* user_data;
    aohttp_callback_fn callback;
    response_buffer_t buffer;
    int32_t in_use;
} transfer_t;

/* Global state */
static CURLM* curl_multi = NULL;
static transfer_t* transfers = NULL;
static int32_t max_transfers = 0;
static int32_t active_count = 0;
static int32_t initialized = 0;

#ifdef AOPIPELINE_WINDOWS
static CRITICAL_SECTION http_lock;
#else
static pthread_mutex_t http_lock = PTHREAD_MUTEX_INITIALIZER;
#endif

#endif /* AOPIPELINE_HAS_CURL */

/*============================================================================
 * Helper Functions
 *============================================================================*/

#ifdef AOPIPELINE_HAS_CURL

static void lock_http(void) {
#ifdef AOPIPELINE_WINDOWS
    EnterCriticalSection(&http_lock);
#else
    pthread_mutex_lock(&http_lock);
#endif
}

static void unlock_http(void) {
#ifdef AOPIPELINE_WINDOWS
    LeaveCriticalSection(&http_lock);
#else
    pthread_mutex_unlock(&http_lock);
#endif
}

/* libcurl write callback */
static size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t realsize = size * nmemb;
    response_buffer_t* buf = (response_buffer_t*)userp;
    
    /* Grow buffer if needed */
    if (buf->size + realsize > buf->capacity) {
        size_t new_capacity = buf->capacity * 2;
        if (new_capacity < buf->size + realsize) {
            new_capacity = buf->size + realsize + INITIAL_BUFFER_SIZE;
        }
        uint8_t* new_data = realloc(buf->data, new_capacity);
        if (!new_data) {
            return 0;  /* Signal error to curl */
        }
        buf->data = new_data;
        buf->capacity = new_capacity;
    }
    
    memcpy(buf->data + buf->size, contents, realsize);
    buf->size += realsize;
    
    return realsize;
}

static transfer_t* get_free_transfer(void) {
    for (int32_t i = 0; i < max_transfers; i++) {
        if (!transfers[i].in_use) {
            return &transfers[i];
        }
    }
    return NULL;
}

static transfer_t* find_transfer_by_easy(CURL* easy) {
    for (int32_t i = 0; i < max_transfers; i++) {
        if (transfers[i].in_use && transfers[i].easy == easy) {
            return &transfers[i];
        }
    }
    return NULL;
}

static void reset_transfer(transfer_t* t) {
    if (t->buffer.data) {
        free(t->buffer.data);
    }
    memset(t, 0, sizeof(transfer_t));
}

#endif /* AOPIPELINE_HAS_CURL */

/*============================================================================
 * Public API Implementation
 *============================================================================*/

AOHTTP_API int32_t aohttp_init(int32_t max_connections) {
#ifdef AOPIPELINE_HAS_CURL
    if (initialized) {
        return 1;
    }
    
    if (max_connections <= 0) {
        max_connections = DEFAULT_MAX_CONNECTIONS;
    }
    if (max_connections > MAX_PENDING_REQUESTS) {
        max_connections = MAX_PENDING_REQUESTS;
    }
    
    /* Initialize curl globally */
    if (curl_global_init(CURL_GLOBAL_DEFAULT) != 0) {
        return 0;
    }
    
    /* Create multi handle */
    curl_multi = curl_multi_init();
    if (!curl_multi) {
        curl_global_cleanup();
        return 0;
    }
    
    /* Configure multi handle */
    curl_multi_setopt(curl_multi, CURLMOPT_MAX_TOTAL_CONNECTIONS, max_connections);
    curl_multi_setopt(curl_multi, CURLMOPT_PIPELINING, CURLPIPE_MULTIPLEX);
    
    /* Allocate transfer pool */
    transfers = calloc(max_connections, sizeof(transfer_t));
    if (!transfers) {
        curl_multi_cleanup(curl_multi);
        curl_global_cleanup();
        return 0;
    }
    max_transfers = max_connections;
    
#ifdef AOPIPELINE_WINDOWS
    InitializeCriticalSection(&http_lock);
#endif
    
    initialized = 1;
    return 1;
#else
    return 0;  /* libcurl not available */
#endif
}

AOHTTP_API void aohttp_shutdown(void) {
#ifdef AOPIPELINE_HAS_CURL
    if (!initialized) return;
    
    lock_http();
    
    /* Clean up all transfers */
    for (int32_t i = 0; i < max_transfers; i++) {
        if (transfers[i].in_use && transfers[i].easy) {
            curl_multi_remove_handle(curl_multi, transfers[i].easy);
            curl_easy_cleanup(transfers[i].easy);
        }
        reset_transfer(&transfers[i]);
    }
    
    free(transfers);
    transfers = NULL;
    max_transfers = 0;
    active_count = 0;
    
    curl_multi_cleanup(curl_multi);
    curl_multi = NULL;
    
    initialized = 0;
    
    unlock_http();
    
#ifdef AOPIPELINE_WINDOWS
    DeleteCriticalSection(&http_lock);
#endif
    
    curl_global_cleanup();
#endif
}

AOHTTP_API int32_t aohttp_is_available(void) {
#ifdef AOPIPELINE_HAS_CURL
    return initialized;
#else
    return 0;
#endif
}

AOHTTP_API int32_t aohttp_submit_batch(
    const aohttp_request_t* requests,
    int32_t count,
    aohttp_callback_fn callback
) {
#ifdef AOPIPELINE_HAS_CURL
    if (!initialized || !requests || count <= 0) {
        return 0;
    }
    
    int32_t submitted = 0;
    
    lock_http();
    
    for (int32_t i = 0; i < count; i++) {
        transfer_t* t = get_free_transfer();
        if (!t) {
            break;  /* Pool exhausted */
        }
        
        CURL* easy = curl_easy_init();
        if (!easy) {
            continue;
        }
        
        /* Configure easy handle */
        curl_easy_setopt(easy, CURLOPT_URL, requests[i].url);
        curl_easy_setopt(easy, CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(easy, CURLOPT_WRITEDATA, &t->buffer);
        curl_easy_setopt(easy, CURLOPT_USERAGENT, user_agent);
        curl_easy_setopt(easy, CURLOPT_TIMEOUT_MS, DEFAULT_TIMEOUT_MS);
        curl_easy_setopt(easy, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(easy, CURLOPT_PRIVATE, t);
        
        if (!ssl_verify) {
            curl_easy_setopt(easy, CURLOPT_SSL_VERIFYPEER, 0L);
            curl_easy_setopt(easy, CURLOPT_SSL_VERIFYHOST, 0L);
        }
        
        /* Initialize buffer */
        t->buffer.data = malloc(INITIAL_BUFFER_SIZE);
        if (!t->buffer.data) {
            curl_easy_cleanup(easy);
            continue;
        }
        t->buffer.size = 0;
        t->buffer.capacity = INITIAL_BUFFER_SIZE;
        
        /* Set up transfer */
        t->easy = easy;
        t->request_id = requests[i].request_id;
        t->user_data = requests[i].user_data;
        t->callback = callback;
        t->in_use = 1;
        
        /* Add to multi handle */
        if (curl_multi_add_handle(curl_multi, easy) == CURLM_OK) {
            active_count++;
            submitted++;
        } else {
            reset_transfer(t);
            curl_easy_cleanup(easy);
        }
    }
    
    unlock_http();
    
    return submitted;
#else
    return 0;
#endif
}

AOHTTP_API int32_t aohttp_poll(int32_t timeout_ms) {
#ifdef AOPIPELINE_HAS_CURL
    if (!initialized || active_count == 0) {
        return 0;
    }
    
    lock_http();
    
    int still_running;
    curl_multi_perform(curl_multi, &still_running);
    
    /* Wait for activity if requested */
    if (timeout_ms > 0 && still_running > 0) {
        int numfds;
        curl_multi_wait(curl_multi, NULL, 0, timeout_ms, &numfds);
        curl_multi_perform(curl_multi, &still_running);
    }
    
    /* Process completed transfers */
    int32_t completed = 0;
    CURLMsg* msg;
    int msgs_left;
    
    while ((msg = curl_multi_info_read(curl_multi, &msgs_left))) {
        if (msg->msg == CURLMSG_DONE) {
            CURL* easy = msg->easy_handle;
            transfer_t* t = NULL;
            curl_easy_getinfo(easy, CURLINFO_PRIVATE, &t);
            
            if (t && t->callback) {
                long status_code = 0;
                curl_easy_getinfo(easy, CURLINFO_RESPONSE_CODE, &status_code);
                
                if (msg->data.result == CURLE_OK && status_code >= 200 && status_code < 300) {
                    t->callback(t->request_id, (int32_t)status_code,
                               t->buffer.data, (uint32_t)t->buffer.size,
                               t->user_data);
                } else {
                    t->callback(t->request_id, 
                               (msg->data.result != CURLE_OK) ? 0 : (int32_t)status_code,
                               NULL, 0, t->user_data);
                }
            }
            
            /* Clean up */
            curl_multi_remove_handle(curl_multi, easy);
            curl_easy_cleanup(easy);
            
            if (t) {
                reset_transfer(t);
            }
            
            active_count--;
            completed++;
        }
    }
    
    unlock_http();
    
    return completed;
#else
    return 0;
#endif
}

AOHTTP_API int32_t aohttp_pending_count(void) {
#ifdef AOPIPELINE_HAS_CURL
    return active_count;
#else
    return 0;
#endif
}

AOHTTP_API int32_t aohttp_get_sync(
    const char* url,
    aohttp_response_t* response,
    int32_t timeout_ms
) {
#ifdef AOPIPELINE_HAS_CURL
    if (!url || !response) return 0;
    
    memset(response, 0, sizeof(aohttp_response_t));
    
    CURL* easy = curl_easy_init();
    if (!easy) {
        safe_strcpy(response->error, "Failed to init curl", 128);
        return 0;
    }
    
    response_buffer_t buffer = {0};
    buffer.data = malloc(INITIAL_BUFFER_SIZE);
    if (!buffer.data) {
        curl_easy_cleanup(easy);
        safe_strcpy(response->error, "Memory allocation failed", 128);
        return 0;
    }
    buffer.capacity = INITIAL_BUFFER_SIZE;
    
    curl_easy_setopt(easy, CURLOPT_URL, url);
    curl_easy_setopt(easy, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(easy, CURLOPT_WRITEDATA, &buffer);
    curl_easy_setopt(easy, CURLOPT_USERAGENT, user_agent);
    curl_easy_setopt(easy, CURLOPT_TIMEOUT_MS, timeout_ms > 0 ? timeout_ms : DEFAULT_TIMEOUT_MS);
    curl_easy_setopt(easy, CURLOPT_FOLLOWLOCATION, 1L);
    
    if (!ssl_verify) {
        curl_easy_setopt(easy, CURLOPT_SSL_VERIFYPEER, 0L);
        curl_easy_setopt(easy, CURLOPT_SSL_VERIFYHOST, 0L);
    }
    
    CURLcode res = curl_easy_perform(easy);
    
    if (res == CURLE_OK) {
        long status_code;
        curl_easy_getinfo(easy, CURLINFO_RESPONSE_CODE, &status_code);
        response->status_code = (int32_t)status_code;
        response->data = buffer.data;
        response->length = (uint32_t)buffer.size;
        buffer.data = NULL;  /* Ownership transferred */
    } else {
        response->status_code = 0;
        safe_strcpy(response->error, curl_easy_strerror(res), 128);
    }
    
    if (buffer.data) free(buffer.data);
    curl_easy_cleanup(easy);
    
    return (response->status_code >= 200 && response->status_code < 300);
#else
    if (response) {
        memset(response, 0, sizeof(aohttp_response_t));
        safe_strcpy(response->error, "libcurl not available", 128);
    }
    return 0;
#endif
}

AOHTTP_API int32_t aohttp_get_batch_sync(
    const char** urls,
    int32_t count,
    aohttp_response_t* responses,
    int32_t timeout_ms
) {
#ifdef AOPIPELINE_HAS_CURL
    if (!urls || !responses || count <= 0) return 0;
    
    int32_t success_count = 0;
    
    /* Create multi handle for this batch */
    CURLM* multi = curl_multi_init();
    if (!multi) return 0;
    
    /* Set up transfers */
    CURL** handles = malloc(count * sizeof(CURL*));
    response_buffer_t* buffers = calloc(count, sizeof(response_buffer_t));
    
    if (!handles || !buffers) {
        free(handles);
        free(buffers);
        curl_multi_cleanup(multi);
        return 0;
    }
    
    for (int32_t i = 0; i < count; i++) {
        memset(&responses[i], 0, sizeof(aohttp_response_t));
        handles[i] = curl_easy_init();
        if (!handles[i]) continue;
        
        buffers[i].data = malloc(INITIAL_BUFFER_SIZE);
        if (!buffers[i].data) {
            curl_easy_cleanup(handles[i]);
            handles[i] = NULL;
            continue;
        }
        buffers[i].capacity = INITIAL_BUFFER_SIZE;
        
        curl_easy_setopt(handles[i], CURLOPT_URL, urls[i]);
        curl_easy_setopt(handles[i], CURLOPT_WRITEFUNCTION, write_callback);
        curl_easy_setopt(handles[i], CURLOPT_WRITEDATA, &buffers[i]);
        curl_easy_setopt(handles[i], CURLOPT_USERAGENT, user_agent);
        curl_easy_setopt(handles[i], CURLOPT_TIMEOUT_MS, 
                        timeout_ms > 0 ? timeout_ms : DEFAULT_TIMEOUT_MS);
        curl_easy_setopt(handles[i], CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(handles[i], CURLOPT_PRIVATE, (void*)(intptr_t)i);
        
        if (!ssl_verify) {
            curl_easy_setopt(handles[i], CURLOPT_SSL_VERIFYPEER, 0L);
            curl_easy_setopt(handles[i], CURLOPT_SSL_VERIFYHOST, 0L);
        }
        
        curl_multi_add_handle(multi, handles[i]);
    }
    
    /* Perform all transfers */
    int still_running;
    do {
        curl_multi_perform(multi, &still_running);
        if (still_running) {
            curl_multi_wait(multi, NULL, 0, 1000, NULL);
        }
    } while (still_running);
    
    /* Process results */
    CURLMsg* msg;
    int msgs_left;
    while ((msg = curl_multi_info_read(multi, &msgs_left))) {
        if (msg->msg == CURLMSG_DONE) {
            intptr_t idx;
            curl_easy_getinfo(msg->easy_handle, CURLINFO_PRIVATE, &idx);
            int32_t i = (int32_t)idx;
            
            if (i >= 0 && i < count) {
                if (msg->data.result == CURLE_OK) {
                    long status_code;
                    curl_easy_getinfo(msg->easy_handle, CURLINFO_RESPONSE_CODE, &status_code);
                    responses[i].status_code = (int32_t)status_code;
                    responses[i].data = buffers[i].data;
                    responses[i].length = (uint32_t)buffers[i].size;
                    buffers[i].data = NULL;
                    
                    if (status_code >= 200 && status_code < 300) {
                        success_count++;
                    }
                } else {
                    safe_strcpy(responses[i].error, curl_easy_strerror(msg->data.result), 128);
                }
            }
        }
    }
    
    /* Cleanup */
    for (int32_t i = 0; i < count; i++) {
        if (handles[i]) {
            curl_multi_remove_handle(multi, handles[i]);
            curl_easy_cleanup(handles[i]);
        }
        if (buffers[i].data) {
            free(buffers[i].data);
        }
    }
    
    free(handles);
    free(buffers);
    curl_multi_cleanup(multi);
    
    return success_count;
#else
    if (responses) {
        for (int32_t i = 0; i < count; i++) {
            memset(&responses[i], 0, sizeof(aohttp_response_t));
            safe_strcpy(responses[i].error, "libcurl not available", 128);
        }
    }
    return 0;
#endif
}

AOHTTP_API void aohttp_response_free(aohttp_response_t* response) {
    if (response && response->data) {
        free(response->data);
        response->data = NULL;
        response->length = 0;
    }
}

AOHTTP_API void aohttp_response_batch_free(aohttp_response_t* responses, int32_t count) {
    if (!responses) return;
    for (int32_t i = 0; i < count; i++) {
        aohttp_response_free(&responses[i]);
    }
}

AOHTTP_API void aohttp_set_user_agent(const char* ua) {
    if (ua) {
        safe_strcpy(user_agent, ua, sizeof(user_agent));
    }
}

AOHTTP_API void aohttp_set_ssl_verify(int32_t verify) {
    ssl_verify = verify;
}

AOHTTP_API const char* aohttp_version(void) {
#ifdef AOPIPELINE_HAS_CURL
    static char version_buf[256];
    curl_version_info_data* info = curl_version_info(CURLVERSION_NOW);
    snprintf(version_buf, sizeof(version_buf),
             "aohttp " AOHTTP_VERSION " (libcurl %s)"
#if AOPIPELINE_HAS_OPENMP
             " (OpenMP enabled)"
#endif
#ifdef AOPIPELINE_WINDOWS
             " [Windows]"
#elif defined(AOPIPELINE_MACOS)
             " [macOS]"
#else
             " [Linux]"
#endif
             , info ? info->version : "unknown"
    );
    return version_buf;
#else
    return "aohttp " AOHTTP_VERSION " (libcurl not available)";
#endif
}

