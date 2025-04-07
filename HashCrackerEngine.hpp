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

// ==================== REAL MD5 IMPLEMENTATION ====================
class MD5 {
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
            0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
        };
    
        static constexpr std::array<uint32_t, 64> s = {
            7, 12, 17, 22,   7, 12, 17, 22,
            7, 12, 17, 22,   7, 12, 17, 22,
            5,  9, 14, 20,   5,  9, 14, 20,
            5,  9, 14, 20,   5,  9, 14, 20,
            4, 11, 16, 23,   4, 11, 16, 23,
            4, 11, 16, 23,   4, 11, 16, 23,
            6, 10, 15, 21,   6, 10, 15, 21,
            6, 10, 15, 21,   6, 10, 15, 21
        };
    
        static uint32_t leftRotate(uint32_t x, uint32_t c) {
            return (x << c) | (x >> (32 - c));
        }
    
        static uint32_t swapEndian(uint32_t n) {
            return ((n & 0xFF000000) >> 24) |
                   ((n & 0x00FF0000) >> 8)  |
                   ((n & 0x0000FF00) << 8)  |
                   ((n & 0x000000FF) << 24);
        }
    
      
    
        static void processBlock(const uint8_t* block, uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
            uint32_t w[16];
            for (int i = 0; i < 16; ++i) {
                w[i] =   (block[i * 4 + 0])
                       | (block[i * 4 + 1] << 8)
                       | (block[i * 4 + 2] << 16)
                       | (block[i * 4 + 3] << 24);
            }
          
    
            uint32_t aa = a, bb = b, cc = c, dd = d;
    
            for (int i = 0; i < 64; ++i) {
                uint32_t f = 0, g = 0;
                if (i < 16) {
                    f = (b & c) | ((~b) & d);
                    g = i;
                } else if (i < 32) {
                    f = (d & b) | ((~d) & c);
                    g = (5 * i + 1) % 16;
                } else if (i < 48) {
                    f = b ^ c ^ d;
                    g = (3 * i + 5) % 16;
                } else {
                    f = c ^ (b | (~d));
                    g = (7 * i) % 16;
                }
    
                uint32_t temp = d;
                d = c;
                c = b;
                b = b + leftRotate(a + f + k[i] + w[g], s[i]);
                a = temp;
    
               
            }
    
            a += aa; b += bb; c += cc; d += dd;
            
        }
    
    public:
        static std::string hash(const std::string& input) {
            uint32_t a = 0x67452301;
            uint32_t b = 0xefcdab89;
            uint32_t c = 0x98badcfe;
            uint32_t d = 0x10325476;
    
            uint64_t bitLength = input.size() * 8;
            std::vector<uint8_t> padded(input.begin(), input.end());
            // Append 0x80
            padded.push_back(0x80);
    
            // Pad with zeros until the size in bytes mod 64 is 56
            while ((padded.size() % 64) != 56)
                padded.push_back(0);
    
            // Append bit length in little-endian format (8 bytes)
            for (int i = 0; i < 8; ++i)
                padded.push_back((bitLength >> (i * 8)) & 0xff);
    
            // Process each 64-byte block.
            for (size_t i = 0; i < padded.size(); i += 64) {
               
                processBlock(&padded[i], a, b, c, d);
            }
    
            std::ostringstream result;
            result << std::hex << std::setfill('0')
                   << std::setw(8) << swapEndian(a)
                   << std::setw(8) << swapEndian(b)
                   << std::setw(8) << swapEndian(c)
                   << std::setw(8) << swapEndian(d);
            return result.str();
        }
    };
    
    // Define static constexpr members outside the class
    constexpr std::array<uint32_t, 64> MD5::k;
    constexpr std::array<uint32_t, 64> MD5::s;
    

// ==================== CORE ENGINE ====================
class HashCrackerEngine {
protected:
    std::atomic<bool> isRunning{false};
    std::atomic<bool> passwordFound{false};
    std::string foundPassword;

public:
    virtual ~HashCrackerEngine() = default;

    virtual std::string crack(const std::string& targetHash,
                              const std::vector<std::string>& wordlist) = 0;

    void stop() { isRunning = false; }
    bool running() const { return isRunning; }
    std::string result() const { return passwordFound ? foundPassword : ""; }
};

// ==================== ALGORITHM IMPLEMENTATIONS ====================
class MD5Cracker : public HashCrackerEngine {
public:
    std::string crack(const std::string& targetHash,
                      const std::vector<std::string>& wordlist) override {
        isRunning = true;
        passwordFound = false;
        foundPassword.clear();

        for (const auto& word : wordlist) {
            if (!isRunning) break;

            std::string hashed = MD5::hash(word);
            if (hashed == targetHash) {
                foundPassword = word;
                passwordFound = true;
                break;
            }
        }

        isRunning = false;
        return foundPassword;
    }
};

class SHACracker : public HashCrackerEngine {
public:
    std::string crack(const std::string& targetHash,
                      const std::vector<std::string>& wordlist) override {
        // Placeholder implementation for SHA cracking
        isRunning = true;
        passwordFound = false;
        foundPassword.clear();

        for (const auto& word : wordlist) {
            if (!isRunning) break;

            // Simulate SHA hash comparison (replace with actual SHA logic)
            std::string hashed = "dummy_sha_hash"; // Replace with actual SHA hash function
            if (hashed == targetHash) {
                foundPassword = word;
                passwordFound = true;
                break;
            }
        }

        isRunning = false;
        return foundPassword;
    }
};

// ==================== CRACKER MANAGER ====================
class HashCrackerManager {
   
    
public:
    std::unordered_map<std::string, std::unique_ptr<HashCrackerEngine>> crackers;
    HashCrackerManager() {
        // Register available algorithms
        crackers["md5"] = std::make_unique<MD5Cracker>();
        crackers["sha1"] = std::make_unique<SHACracker>();
        // Add more algorithms...
    }

    std::string crackHash(const std::string& hash,
                         const std::vector<std::string>& wordlist,
                         const std::string& hashType = "auto") {
        std::string type = (hashType == "auto") ? identifyHash(hash) : hashType;
        
        if (crackers.find(type) == crackers.end()) {
            throw std::runtime_error("Unsupported hash type");
        }

        return crackers[type]->crack(hash, wordlist);
    }

private:
    std::string identifyHash(const std::string& hash) {
        // Simple hash identification
        switch(hash.length()) {
            case 32:  return "md5";
            case 40:  return "sha1";
            case 64:  return "sha256";
            default:  return "unknown";
        }
    }
};

// ==================== USAGE EXAMPLE ====================
