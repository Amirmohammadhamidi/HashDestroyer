#pragma once
#ifdef __cplusplus
extern "C" {
#endif

void runMD5BruteForceKernel(const char* targetHash, char* foundCandidate, bool* found, int numCandidates);

#ifdef __cplusplus
}
#endif