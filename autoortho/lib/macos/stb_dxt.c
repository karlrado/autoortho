#include <stdint.h>
#include <stdbool.h>
#include <string.h>

// Minimal DXT1 compression implementation
// This is a simplified version for compatibility

typedef struct {
    uint8_t r, g, b, a;
} rgba_pixel;

static uint16_t pack_rgb565(uint8_t r, uint8_t g, uint8_t b) {
    return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
}

static void compress_block_dxt1(const rgba_pixel* block, uint8_t* output) {
    // Very simplified DXT1 compression - just pick two representative colors
    rgba_pixel color0 = block[0];
    rgba_pixel color1 = block[15];
    
    uint16_t c0 = pack_rgb565(color0.r, color0.g, color0.b);
    uint16_t c1 = pack_rgb565(color1.r, color1.g, color1.b);
    
    // Ensure c0 > c1 for 4-color mode
    if (c0 < c1) {
        uint16_t temp = c0;
        c0 = c1;
        c1 = temp;
    }
    
    // Store colors
    *(uint16_t*)output = c0;
    *(uint16_t*)(output + 2) = c1;
    
    // Simple index selection (just use pattern)
    *(uint32_t*)(output + 4) = 0x00000000; // All pixels use color0
}

__attribute__((visibility("default")))
bool compress_pixels(uint8_t* output, const uint8_t* input, 
                    uint64_t width, uint64_t height, bool is_rgba) {
    if (!output || !input || width % 4 != 0 || height % 4 != 0) {
        return false;
    }
    
    size_t blocks_x = width / 4;
    size_t blocks_y = height / 4;
    size_t stride = is_rgba ? 4 : 3;
    
    for (size_t by = 0; by < blocks_y; by++) {
        for (size_t bx = 0; bx < blocks_x; bx++) {
            rgba_pixel block[16];
            
            // Extract 4x4 block
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 4; x++) {
                    size_t src_idx = ((by * 4 + y) * width + (bx * 4 + x)) * stride;
                    block[y * 4 + x].r = input[src_idx];
                    block[y * 4 + x].g = input[src_idx + 1];
                    block[y * 4 + x].b = input[src_idx + 2];
                    block[y * 4 + x].a = is_rgba ? input[src_idx + 3] : 255;
                }
            }
            
            // Compress block
            size_t output_idx = (by * blocks_x + bx) * 8;
            compress_block_dxt1(block, output + output_idx);
        }
    }
    
    return true;
}
