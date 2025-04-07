#include <iostream>
#include <array>
#include <vector>
#include <sstream>
#include <iomanip>
#include <cstdint>

#include "HashCrackerEngine.hpp"


int main() {
    std::string input = "password";
    std::string expectedHash = "5f4dcc3b5aa765d61d8327deb882cf99"; // MD5 hash of "password"
    std::string computedHash = MD5::hash(input);

    std::cout << "Input: \"" << input << "\"" << std::endl;
    std::cout << "Expected Hash: " << expectedHash << std::endl;
    std::cout << "Computed Hash: " << computedHash << std::endl;

    if (computedHash == expectedHash)
        std::cout << "MD5 implementation is correct!" << std::endl;
    else
        std::cout << "MD5 implementation is incorrect!" << std::endl;

    return 0;
}