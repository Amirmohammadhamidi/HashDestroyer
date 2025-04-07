#include <cuda_runtime.h>
#include <stdio.h>
#include "MD5GPUKernel.h"
#include "MD5Device.h"  // Include our __device__ MD5 function

// Place the candidate charset in constant memory (cached for fast access)
__constant__ char d_charset[37] = "abcdefghijklmnopqrstuvwxyz0123456789";

// Kernel: generate candidate from idx (using base-36 conversion) and compute its MD5.
__global__ void md5BruteForceKernel(const char* d_targetHash, char* d_foundCandidate, bool* d_found, int numCandidates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numCandidates || *d_found)
        return;
    
    // Optimized candidate generation using a reverse loop.
    char candidate[6];
    int temp = idx;
    #pragma unroll
    for (int i = 4; i >= 0; i--) {
        candidate[i] = d_charset[temp % 36];
        temp /= 36;
    }
    candidate[5] = '\0';
    
    // Compute MD5 for this candidate.
    char computedHash[33];
    deviceMD5(candidate, computedHash);
    
    // Optimized hash comparison: process 32 bytes as 8 uint32_t words.
    bool match = true;
    const uint32_t* targetInt = reinterpret_cast<const uint32_t*>(d_targetHash);
    const uint32_t* computedInt = reinterpret_cast<const uint32_t*>(computedHash);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (computedInt[i] != targetInt[i]) {
            match = false;
            break;
        }
    }
    
    // If a match is found, copy the candidate to global memory.
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
    size_t hashSize = 33 * sizeof(char);
    char* d_targetHash;
    char* d_foundCandidate;
    bool* d_found;
    cudaMalloc(&d_targetHash, hashSize);
    cudaMalloc(&d_foundCandidate, 6 * sizeof(char));
    cudaMalloc(&d_found, sizeof(bool));
    
    cudaMemcpy(d_targetHash, targetHash, hashSize, cudaMemcpyHostToDevice);
    bool foundInit = false;
    cudaMemcpy(d_found, &foundInit, sizeof(bool), cudaMemcpyHostToDevice);
    
    int blockSize = 1024;
    int gridSize = (numCandidates + blockSize - 1) / blockSize;
    
    // Create events to time the kernel execution.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // Launch the kernel.
    md5BruteForceKernel<<<gridSize, blockSize>>>(d_targetHash, d_foundCandidate, d_found, numCandidates);
    cudaDeviceSynchronize();
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel time: %.2f ms\n", ms);
    
    // Copy back the result.
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