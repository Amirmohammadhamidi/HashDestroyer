#include <cuda_runtime.h>
#include <stdio.h>
#include "MD5GPUKernel.h"
#include "MD5Device.h"  // Include our __device__ MD5 function

// Kernel: generate candidate from idx (as a base-36 number) and compute its MD5.
__global__ void md5BruteForceKernel(const char* d_targetHash, char* d_foundCandidate, bool* d_found, int numCandidates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numCandidates || *d_found)
        return;
    
    // Convert idx to a 5-character candidate using Base36.
    char candidate[6];
    const char charset[37] = "abcdefghijklmnopqrstuvwxyz0123456789";
    int base = 36;
    int temp = idx;
    candidate[5] = '\0';
    for (int i = 4; i >= 0; i--) {
        candidate[i] = charset[temp % base];
        temp /= base;
    }
    
    // Compute MD5 on candidate.
    char computedHash[33];
    deviceMD5(candidate, computedHash);
    
    // Compare computed hash with target hash.
    bool match = true;
    for (int i = 0; i < 32; i++) {
        if (computedHash[i] != d_targetHash[i]) {
            match = false;
            break;
        }
    }
    
    if (match) {
        // Copy candidate to global memory.
        for (int i = 0; i < 5; i++)
            d_foundCandidate[i] = candidate[i];
        d_foundCandidate[5] = '\0';
        *d_found = true;
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
    
    int blockSize = 256;
    int gridSize = (numCandidates + blockSize - 1) / blockSize;
    md5BruteForceKernel<<<gridSize, blockSize>>>(d_targetHash, d_foundCandidate, d_found, numCandidates);
    cudaDeviceSynchronize();
    
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