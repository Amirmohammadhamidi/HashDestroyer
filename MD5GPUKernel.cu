#include <cuda_runtime.h>
#include <stdio.h>
#include <string>  // Needed for std::string conversions on host
#include "HashCrackerEngine.hpp"   // Provides MD5 and swapEndian
#include "MD5GPUKernel.h"
#include "MD5Device.h"             // Uses our optimized deviceMD5

// Candidate charset stored in constant memory.
__constant__ char d_charset[73] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()";

// Optimized kernel: uses __restrict__ pointers and unroll directives.
__global__ void md5BruteForceKernel(const uint32_t* __restrict__ d_targetHash,
                                    char* __restrict__ d_foundCandidate,
                                    bool* __restrict__ d_found,
                                    int numCandidates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numCandidates || *d_found)
        return;
    
    // Generate a 5-character candidate.
    char candidate[6];
    int temp = idx;
    #pragma unroll
    for (int i = 4; i >= 0; i--) {
        candidate[i] = d_charset[temp % 72];
        temp /= 72;
    }
    candidate[5] = '\0';
    
    // Compute MD5 (raw output: 4 uint32_t words).
    uint32_t computedHash[4];
    deviceMD5(candidate, computedHash);
    
    // Compare computed hash with target hash.
    bool match = true;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        if (computedHash[i] != d_targetHash[i]) {
            match = false;
            break;
        }
    }
    
    // If a match is found, save the candidate.
    if (match) {
        *d_found = true;
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            d_foundCandidate[i] = candidate[i];
        }
        d_foundCandidate[5] = '\0';
    }
}

extern "C" void runMD5BruteForceKernel(const char* targetHash, char* foundCandidate, bool* found, int numCandidates) {
    // Convert the target hash (hex string) into four 32-bit words.
    uint32_t a = std::stoul(std::string(targetHash).substr(0, 8), nullptr, 16);
    uint32_t b = std::stoul(std::string(targetHash).substr(8, 8), nullptr, 16);
    uint32_t c = std::stoul(std::string(targetHash).substr(16, 8), nullptr, 16);
    uint32_t d = std::stoul(std::string(targetHash).substr(24, 8), nullptr, 16);
    
    a = MD5::swapEndian(a);
    b = MD5::swapEndian(b);
    c = MD5::swapEndian(c);
    d = MD5::swapEndian(d);
    
    // Adjust the target state.
    a = a - 0x67452301;
    b = b - 0xefcdab89;
    c = c - 0x98badcfe;
    d = d - 0x10325476;
    
    uint32_t h_target[4] = { a, b, c, d };
    
    // Allocate device memory.
    uint32_t* d_targetHash;
    cudaMalloc(&d_targetHash, 4 * sizeof(uint32_t));
    cudaMemcpy(d_targetHash, h_target, 4 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    char* d_foundCandidate;
    bool* d_found;
    cudaMalloc(&d_foundCandidate, 6 * sizeof(char));
    cudaMalloc(&d_found, sizeof(bool));
    
    bool initFound = false;
    cudaMemcpy(d_found, &initFound, sizeof(bool), cudaMemcpyHostToDevice);
    
    int blockSize = 1024;
    int gridSize = (numCandidates + blockSize - 1) / blockSize;
    
    // Time the kernel execution.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    md5BruteForceKernel<<<gridSize, blockSize>>>(d_targetHash, d_foundCandidate, d_found, numCandidates);
    cudaDeviceSynchronize();
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel time: %.2f ms\n", ms);
    
    // Copy back result if found.
    bool h_found;
    cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);
    if (h_found) {
        cudaMemcpy(foundCandidate, d_foundCandidate, 6 * sizeof(char), cudaMemcpyDeviceToHost);
        *found = true;
    }
    
    cudaFree(d_targetHash);
    cudaFree(d_foundCandidate);
    cudaFree(d_found);
}