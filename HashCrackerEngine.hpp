#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <thread>
#include <atomic>
#include <array>
#include <sstream>
#include <iomanip>
#include <cstdint>
#include <mutex>
#include <future>

// ==================== REAL MD5 IMPLEMENTATION ====================
class MD5
{
private:
    static constexpr std::array<uint32_t, 64> k = {
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
        0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391};
    static constexpr std::array<uint32_t, 64> s = {
        7, 12, 17, 22, 7, 12, 17, 22,
        7, 12, 17, 22, 7, 12, 17, 22,
        5, 9, 14, 20, 5, 9, 14, 20,
        5, 9, 14, 20, 5, 9, 14, 20,
        4, 11, 16, 23, 4, 11, 16, 23,
        4, 11, 16, 23, 4, 11, 16, 23,
        6, 10, 15, 21, 6, 10, 15, 21,
        6, 10, 15, 21, 6, 10, 15, 21};

    static uint32_t leftRotate(uint32_t x, uint32_t c)
    {
        return (x << c) | (x >> (32 - c));
    }
    static void processBlock(const uint8_t *block, uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d)
    {
        uint32_t w[16];
        for (int i = 0; i < 16; ++i)
        {
            w[i] = (block[i * 4 + 0]) | (block[i * 4 + 1] << 8) | (block[i * 4 + 2] << 16) | (block[i * 4 + 3] << 24);
        }
        uint32_t aa = a, bb = b, cc = c, dd = d;
        for (int i = 0; i < 64; ++i)
        {
            uint32_t f = 0, g = 0;
            if (i < 16)
            {
                f = (b & c) | ((~b) & d);
                g = i;
            }
            else if (i < 32)
            {
                f = (d & b) | ((~d) & c);
                g = (5 * i + 1) % 16;
            }
            else if (i < 48)
            {
                f = b ^ c ^ d;
                g = (3 * i + 5) % 16;
            }
            else
            {
                f = c ^ (b | (~d));
                g = (7 * i) % 16;
            }
            uint32_t temp = d;
            d = c;
            c = b;
            b = b + leftRotate(a + f + k[i] + w[g], s[i]);
            a = temp;
        }
        a += aa;
        b += bb;
        c += cc;
        d += dd;
    }

public:
    static uint32_t swapEndian(uint32_t n)
    {
        return ((n & 0xFF000000) >> 24) |
               ((n & 0x00FF0000) >> 8) |
               ((n & 0x0000FF00) << 8) |
               ((n & 0x000000FF) << 24);
    }
    static std::string hash(const std::string &input)
    {
        uint32_t a = 0x67452301;
        uint32_t b = 0xefcdab89;
        uint32_t c = 0x98badcfe;
        uint32_t d = 0x10325476;
        uint64_t bitLength = input.size() * 8;
        std::vector<uint8_t> padded(input.begin(), input.end());
        padded.push_back(0x80);
        while ((padded.size() % 64) != 56)
            padded.push_back(0);
        for (int i = 0; i < 8; ++i)
            padded.push_back((bitLength >> (i * 8)) & 0xff);
        for (size_t i = 0; i < padded.size(); i += 64)
            processBlock(&padded[i], a, b, c, d);
        std::ostringstream result;
        result << std::hex << std::setfill('0')
               << std::setw(8) << swapEndian(a)
               << std::setw(8) << swapEndian(b)
               << std::setw(8) << swapEndian(c)
               << std::setw(8) << swapEndian(d);
        return result.str();
    }
};
constexpr std::array<uint32_t, 64> MD5::k;
constexpr std::array<uint32_t, 64> MD5::s;

// ==================== CORE ENGINE ====================
class HashCrackerEngine
{
protected:
    std::atomic<bool> isRunning{false};
    std::atomic<bool> passwordFound{false};
    std::string foundPassword;

public:
    virtual ~HashCrackerEngine() = default;
    virtual std::string crack(const std::string &targetHash,
                              const std::vector<std::string> &wordlist) = 0;
    void stop() { isRunning = false; }
    bool running() const { return isRunning; }
    std::string result() const { return passwordFound ? foundPassword : ""; }
};

// ==================== ALGORITHM IMPLEMENTATIONS ====================

// Dictionary (wordlist) based MD5 cracker.
class MD5Cracker : public HashCrackerEngine
{
public:
    std::string crack(const std::string &targetHash,
                      const std::vector<std::string> &wordlist) override
    {
        isRunning = true;
        passwordFound = false;
        foundPassword.clear();
        for (const auto &word : wordlist)
        {
            if (!isRunning)
                break;
            std::string hashed = MD5::hash(word);
            if (hashed == targetHash)
            {
                foundPassword = word;
                passwordFound = true;
                break;
            }
        }
        isRunning = false;
        return foundPassword;
    }
};

class SHACracker : public HashCrackerEngine
{
public:
    std::string crack(const std::string &targetHash,
                      const std::vector<std::string> &wordlist) override
    {
        isRunning = true;
        passwordFound = false;
        foundPassword.clear();
        for (const auto &word : wordlist)
        {
            if (!isRunning)
                break;
            std::string hashed = "dummy_sha_hash";
            if (hashed == targetHash)
            {
                foundPassword = word;
                passwordFound = true;
                break;
            }
        }
        isRunning = false;
        return foundPassword;
    }
};

//
// GPU-enabled MD5 cracker (dictionary-based)
// (This remains unchanged as it uses CPU code.)
//
class MD5GPUCracker : public HashCrackerEngine
{
public:
    std::string crack(const std::string &targetHash,
                      const std::vector<std::string> &wordlist) override
    {
        isRunning = true;
        passwordFound = false;
        foundPassword.clear();
        std::string concatenated;
        std::vector<int> offsets;
        std::vector<int> lengths;
        int currentOffset = 0;
        for (const auto &word : wordlist)
        {
            offsets.push_back(currentOffset);
            lengths.push_back(static_cast<int>(word.size()));
            concatenated += word;
            currentOffset += static_cast<int>(word.size());
        }
        uint32_t targetParts[4];
        for (int i = 0; i < 4; ++i)
        {
            targetParts[i] = static_cast<uint32_t>(std::stoul(targetHash.substr(i * 8, 8), nullptr, 16));
        }
        int foundIndex = -1;
        launchGPUMD5Kernel(concatenated, offsets, lengths, targetParts, foundIndex);
        if (foundIndex >= 0 && foundIndex < static_cast<int>(wordlist.size()))
        {
            foundPassword = wordlist[foundIndex];
            passwordFound = true;
        }
        isRunning = false;
        return foundPassword;
    }

private:
    // In our dictionary-based method we simulate GPU work on CPU.
    void launchGPUMD5Kernel(const std::string &concatenated,
                            const std::vector<int> &offsets,
                            const std::vector<int> &lengths,
                            const uint32_t targetParts[4],
                            int &foundIndex)
    {
        std::string target;
        {
            std::ostringstream oss;
            oss << std::hex << std::setfill('0')
                << std::setw(8) << targetParts[0]
                << std::setw(8) << targetParts[1]
                << std::setw(8) << targetParts[2]
                << std::setw(8) << targetParts[3];
            target = oss.str();
        }
        for (size_t i = 0; i < offsets.size(); ++i)
        {
            std::string word = concatenated.substr(offsets[i], lengths[i]);
            std::string hashed = MD5::hash(word);
            if (hashed == target)
            {
                foundIndex = static_cast<int>(i);
                break;
            }
        }
    }
};

//
// High-performance CPU brute-force MD5 cracker using std::async.
//
class MD5BruteforceCPUCracker : public HashCrackerEngine
{
public:
    std::string crack(const std::string &targetHash,
                      const std::vector<std::string> & /*unused*/) override
    {
        isRunning = true;
        passwordFound = false;
        foundPassword.clear();
        found.store(false);

        // Total candidate count for 5-character strings (36^5)
        const int totalCandidates = 60466176;
        // Number of threads to launch
        const int numThreads = 1000;
        // Candidate range per thread
        int segment = totalCandidates / numThreads;
        std::string charset = "abcdefghijklmnopqrstuvwxyz0123456789";

        std::vector<std::future<void>> futures;
        for (int i = 0; i < numThreads; i++)
        {
            int startIdx = i * segment;
            int endIdx = (i == numThreads - 1) ? totalCandidates : (i + 1) * segment;
            futures.push_back(std::async(std::launch::async, [&, startIdx, endIdx, targetHash, charset]()
                                         {
                for (int idx = startIdx; idx < endIdx; idx++) {
                    if(found.load())
                        break;
                    // Convert idx into a 5-character candidate (base-36 conversion)
                    std::string candidate(5, ' ');
                    int temp = idx;
                    for (int pos = 4; pos >= 0; pos--) {
                        candidate[pos] = charset[temp % 36];
                        temp /= 36;
                    }
                    if(MD5::hash(candidate) == targetHash) {
                        std::lock_guard<std::mutex> lock(mtx);
                        if(!found.load()){
                            found = true;
                            foundPassword = candidate;
                            passwordFound = true;
                        }
                        break;
                    }
                } }));
        }
        for (auto &fut : futures)
            fut.wait();
        isRunning = false;
        return foundPassword;
    }

private:
    std::atomic<bool> found{false};
    std::mutex mtx;
};

//
// GPU brute-force MD5 cracker (real GPU implementation)
// This class calls our CUDA wrapper declared in MD5GPUKernel.h.
//
#ifdef USE_CUDA
#include "MD5GPUKernel.h"
class MD5BruteforceGPUCracker : public HashCrackerEngine
{
public:
    std::string crack(const std::string &targetHash,
                      const std::vector<std::string> & /*unused*/) override
    {
        isRunning = true;
        passwordFound = false;
        foundPassword.clear();
        // We assume a candidate length of 5 and plan to search (example value)
        int numCandidates = 60466176;
        char foundCandidate[6] = {0};
        bool gpuFound = false;
        // Call the CUDA wrapper function.
        runMD5BruteForceKernel(targetHash.c_str(), foundCandidate, &gpuFound, numCandidates);
        if (gpuFound)
        {
            foundPassword = std::string(foundCandidate);
            passwordFound = true;
        }
        isRunning = false;
        return foundPassword;
    }
};
#else
// If CUDA is not enabled, fallback to simulated GPU code.
class MD5BruteforceGPUCracker : public HashCrackerEngine
{
public:
    std::string crack(const std::string &targetHash,
                      const std::vector<std::string> & /*unused*/) override
    {
        return "CUDA not enabled";
    }
};
#endif

// ==================== CRACKER MANAGER ====================
class HashCrackerManager
{
public:
    std::unordered_map<std::string, std::unique_ptr<HashCrackerEngine>> crackers;
    HashCrackerManager()
    {
        crackers["md5_cpu"] = std::make_unique<MD5Cracker>();
        crackers["md5_gpu"] = std::make_unique<MD5GPUCracker>();
        crackers["md5_bruteforce_cpu"] = std::make_unique<MD5BruteforceCPUCracker>();
        crackers["md5_bruteforce_gpu"] = std::make_unique<MD5BruteforceGPUCracker>();
        crackers["sha1_cpu"] = std::make_unique<SHACracker>();
    }

    std::string crackHash(const std::string &hash,
                          const std::vector<std::string> &wordlist,
                          const std::string &method,
                          const std::string &processor)
    {
        std::string key = method;
        if (processor == "gpu")
            key += "_gpu";
        else if (processor == "cpu")
            key += "_cpu";
        else
            throw std::runtime_error("Unsupported processor type");

        if (crackers.find(key) == crackers.end())
            throw std::runtime_error("Unsupported cracking method for the given processor");

        return crackers[key]->crack(hash, wordlist);
    }

    std::string crackHash(const std::string &hash,
                          const std::vector<std::string> &wordlist,
                          const std::string &hashType = "auto")
    {
        std::string type = (hashType == "auto") ? identifyHash(hash) : hashType;
        if (crackers.find(type) == crackers.end())
            throw std::runtime_error("Unsupported hash type");
        return crackers[type]->crack(hash, wordlist);
    }

private:
    std::string identifyHash(const std::string &hash)
    {
        switch (hash.length())
        {
        case 32:
            return "md5_cpu";
        case 40:
            return "sha1_cpu";
        case 64:
            return "sha256";
        default:
            return "unknown";
        }
    }
};
