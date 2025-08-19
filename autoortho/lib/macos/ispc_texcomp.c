#include <stdint.h>
#include <string.h>

// RGBA surface structure matching the Python definition
typedef struct {
    uint8_t* data;
    uint32_t width;
    uint32_t height;
    uint32_t stride;
} rgba_surface;

// DXT5/BC3 block structure
typedef struct {
    uint8_t alpha[8];    // Alpha block (2 alphas + 6 bytes indices)
    uint8_t color[8];    // Color block (same as DXT1)
} dxt5_block;

// DXT1/BC1 block structure  
typedef struct {
    uint16_t color0;
    uint16_t color1;
    uint32_t indices;
} dxt1_block;

static uint16_t pack_rgb565(uint8_t r, uint8_t g, uint8_t b) {
    return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
}

static void compress_bc1_block(const uint8_t* rgba_block, dxt1_block* output) {
    // Simple implementation - pick first and last pixels as representative colors
    uint8_t r0 = rgba_block[0], g0 = rgba_block[1], b0 = rgba_block[2];
    uint8_t r1 = rgba_block[60], g1 = rgba_block[61], b1 = rgba_block[62]; // Last pixel
    
    output->color0 = pack_rgb565(r0, g0, b0);
    output->color1 = pack_rgb565(r1, g1, b1);
    
    // Ensure color0 > color1 for 4-color mode
    if (output->color0 < output->color1) {
        uint16_t temp = output->color0;
        output->color0 = output->color1;
        output->color1 = temp;
    }
    
    // Simple index pattern (all pixels use color0)
    output->indices = 0x00000000;
}

static void compress_bc3_block(const uint8_t* rgba_block, dxt5_block* output) {
    // Alpha block - simple implementation
    output->alpha[0] = rgba_block[3];     // alpha0
    output->alpha[1] = rgba_block[63];    // alpha1 (last pixel)
    
    // Ensure alpha0 > alpha1 for 8-alpha mode
    if (output->alpha[0] < output->alpha[1]) {
        uint8_t temp = output->alpha[0];
        output->alpha[0] = output->alpha[1];
        output->alpha[1] = temp;
    }
    
    // Alpha indices (all use alpha0)
    memset(&output->alpha[2], 0, 6);
    
    // Color block (same as BC1)
    dxt1_block* color_block = (dxt1_block*)output->color;
    compress_bc1_block(rgba_block, color_block);
}

__attribute__((visibility("default")))
void CompressBlocksBC1(rgba_surface* surface, uint8_t* output) {
    if (!surface || !surface->data || !output) return;
    
    uint32_t blocks_x = surface->width / 4;
    uint32_t blocks_y = surface->height / 4;
    
    for (uint32_t by = 0; by < blocks_y; by++) {
        for (uint32_t bx = 0; bx < blocks_x; bx++) {
            uint8_t block[64]; // 4x4 RGBA block
            
            // Extract 4x4 block
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 4; x++) {
                    uint32_t src_idx = ((by * 4 + y) * surface->width + (bx * 4 + x)) * 4;
                    uint32_t dst_idx = (y * 4 + x) * 4;
                    
                    block[dst_idx + 0] = surface->data[src_idx + 0]; // R
                    block[dst_idx + 1] = surface->data[src_idx + 1]; // G
                    block[dst_idx + 2] = surface->data[src_idx + 2]; // B
                    block[dst_idx + 3] = surface->data[src_idx + 3]; // A
                }
            }
            
            // Compress block
            uint32_t output_idx = (by * blocks_x + bx) * 8;
            dxt1_block* out_block = (dxt1_block*)(output + output_idx);
            compress_bc1_block(block, out_block);
        }
    }
}

__attribute__((visibility("default")))
void CompressBlocksBC3(rgba_surface* surface, uint8_t* output) {
    if (!surface || !surface->data || !output) return;
    
    uint32_t blocks_x = surface->width / 4;
    uint32_t blocks_y = surface->height / 4;
    
    for (uint32_t by = 0; by < blocks_y; by++) {
        for (uint32_t bx = 0; bx < blocks_x; bx++) {
            uint8_t block[64]; // 4x4 RGBA block
            
            // Extract 4x4 block
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 4; x++) {
                    uint32_t src_idx = ((by * 4 + y) * surface->width + (bx * 4 + x)) * 4;
                    uint32_t dst_idx = (y * 4 + x) * 4;
                    
                    block[dst_idx + 0] = surface->data[src_idx + 0]; // R
                    block[dst_idx + 1] = surface->data[src_idx + 1]; // G  
                    block[dst_idx + 2] = surface->data[src_idx + 2]; // B
                    block[dst_idx + 3] = surface->data[src_idx + 3]; // A
                }
            }
            
            // Compress block
            uint32_t output_idx = (by * blocks_x + bx) * 16; // BC3 blocks are 16 bytes
            dxt5_block* out_block = (dxt5_block*)(output + output_idx);
            compress_bc3_block(block, out_block);
        }
    }
}
