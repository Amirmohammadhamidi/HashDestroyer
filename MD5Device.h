#ifndef MD5DEVICE_H
#define MD5DEVICE_H

#include <stdint.h>

__device__ inline char nibble_to_hex(uint8_t nibble) {
    return (nibble < 10) ? ('0' + nibble) : ('a' + (nibble - 10));
}

// __device__ helper: left rotate a 32-bit value.
__device__ uint32_t leftRotate(uint32_t x, uint32_t c) {
    return (x << c) | (x >> (32 - c));
}

// A simplified __device__ MD5 implementation for fixed 5-byte input.
__device__ void deviceMD5(const char* input, char* output) {
    // Assume input length = 5.
    const int inLen = 5;
    uint32_t a0 = 0x67452301;
    uint32_t b0 = 0xefcdab89;
    uint32_t c0 = 0x98badcfe;
    uint32_t d0 = 0x10325476;
    
    // Prepare one 64-byte block.
    unsigned char block[64] = {0};
    for (int i = 0; i < inLen; i++) {
        block[i] = input[i];
    }
    block[inLen] = 0x80;  // append bit '1'
    // Zero padding until index 56 is implicit.
    // Append message length in bits (5*8 = 40) as 64-bit little endian.
    block[56] = 40; // 40, then rest zero.
    
    // Prepare sixteen 32-bit words in little endian.
    uint32_t w[16];
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i*4]) |
               (((uint32_t)block[i*4+1]) << 8) |
               (((uint32_t)block[i*4+2]) << 16) |
               (((uint32_t)block[i*4+3]) << 24);
    }
    
    const uint32_t k[64] = {
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
        0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
        0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
        0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
        0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
        0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
        0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
        0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
        0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
        0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
        0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
        0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
        0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
        0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
    };
    const uint32_t s[64] = {
         7,12,17,22, 7,12,17,22, 7,12,17,22, 7,12,17,22,
         5,9,14,20, 5,9,14,20, 5,9,14,20, 5,9,14,20,
         4,11,16,23, 4,11,16,23, 4,11,16,23, 4,11,16,23,
         6,10,15,21, 6,10,15,21, 6,10,15,21, 6,10,15,21
    };
    
    uint32_t A = a0, B = b0, C = c0, D = d0;
    for (int i = 0; i < 64; i++) {
        uint32_t F, g;
        if (i < 16) {
            F = (B & C) | ((~B) & D);
            g = i;
        } else if (i < 32) {
            F = (D & B) | ((~D) & C);
            g = (5 * i + 1) % 16;
        } else if (i < 48) {
            F = B ^ C ^ D;
            g = (3 * i + 5) % 16;
        } else {
            F = C ^ (B | (~D));
            g = (7 * i) % 16;
        }
        uint32_t temp = D;
        D = C;
        C = B;
        B = B + leftRotate(A + F + k[i] + w[g], s[i]);
        A = temp;
    }
    A += a0; B += b0; C += c0; D += d0;
    
    // Convert the 4 32-bit words (A, B, C, D) into a 32-character hexadecimal string.
    // We process each byte manually.
    uint32_t vals[4] = {A, B, C, D};
    int pos = 0;
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
            uint8_t byte = (vals[j] >> (8 * i)) & 0xFF;
            uint8_t hi = byte >> 4;
            uint8_t lo = byte & 0x0F;
            output[pos++] = nibble_to_hex(hi);
            output[pos++] = nibble_to_hex(lo);
        }
    }
    output[32] = '\0';
}

#endif