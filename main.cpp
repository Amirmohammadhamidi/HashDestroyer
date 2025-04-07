#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include "HashCrackerEngine.hpp"

int main() {
    std::string input = "password";
    std::string expectedHash = "5f4dcc3b5aa765d61d8327deb882cf99"; // MD5("password")
    std::string computedHash = MD5::hash(input);
    
    std::cout << "Dictionary Test:" << std::endl;
    std::cout << "Input: \"" << input << "\"" << std::endl;
    std::cout << "Expected MD5: " << expectedHash << std::endl;
    std::cout << "Computed MD5: " << computedHash << std::endl;
    if (computedHash == expectedHash)
        std::cout << "MD5 implementation is correct!" << std::endl;
    else
        std::cout << "MD5 implementation is incorrect!" << std::endl;
    
    std::vector<std::string> wordlist = {"123456", "admin", "letmein", "password", "qwerty"};
    HashCrackerManager manager;
    
    {
        auto start = std::chrono::high_resolution_clock::now();
        std::string cracked = manager.crackHash(expectedHash, wordlist, "md5", "cpu");
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "\n[MD5 (CPU)] Cracked password: " << cracked << std::endl;
        std::cout << "[MD5 (CPU)] Time taken: " << elapsed.count() << " seconds" << std::endl;
    }
    
    {
        auto start = std::chrono::high_resolution_clock::now();
        std::string cracked = manager.crackHash(expectedHash, wordlist, "md5", "gpu");
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "\n[MD5 (GPU)] Cracked password: " << cracked << std::endl;
        std::cout << "[MD5 (GPU)] Time taken: " << elapsed.count() << " seconds" << std::endl;
    }
    
    std::string bfTestPassword = "abc12";
    std::string bfExpectedHash = MD5::hash(bfTestPassword);
    std::cout << "\nBrute-force Test:" << std::endl;
    std::cout << "Brute-force target password (5 letters): \"" << bfTestPassword << "\"" << std::endl;
    std::cout << "Expected MD5: " << bfExpectedHash << std::endl;
    
    {
        auto start = std::chrono::high_resolution_clock::now();
        std::string cracked = manager.crackHash(bfExpectedHash, {}, "md5_bruteforce", "gpu");
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        if (cracked.empty())
            std::cout << "[MD5 Brute-force (GPU)] Password not found!" << std::endl;
        else
            std::cout << "[MD5 Brute-force (GPU)] Cracked password: " << cracked << std::endl;
        std::cout << "[MD5 Brute-force (GPU)] Time taken: " << elapsed.count() << " seconds" << std::endl;
    }
    
    {
        auto start = std::chrono::high_resolution_clock::now();
        std::string cracked = manager.crackHash(bfExpectedHash, {}, "md5_bruteforce", "cpu");
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        if (cracked.empty())
            std::cout << "[MD5 Brute-force (CPU)] Password not found!" << std::endl;
        else
            std::cout << "[MD5 Brute-force (CPU)] Cracked password: " << cracked << std::endl;
        std::cout << "[MD5 Brute-force (CPU)] Time taken: " << elapsed.count() << " seconds" << std::endl;
    }
    
    return 0;
}