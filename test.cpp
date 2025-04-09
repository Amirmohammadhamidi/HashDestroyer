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

using namespace std;
class MD5 {
    public:
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
        static void processBlock(const uint8_t* block, uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d) {
            uint32_t w[16];
            for (int i = 0; i < 16; ++i) {
                w[i] = (block[i * 4 + 0])
                     | (block[i * 4 + 1] << 8)
                     | (block[i * 4 + 2] << 16)
                     | (block[i * 4 + 3] << 24);
            }
            
            // Debugging output
            // ptint w
            std::cout << "this is w " << std::hex;
            for (int i = 0; i < 16; ++i) {
                std::cout << std::hex << w[i] << " ";
            }
            cout<<endl;


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
                cout << "this is f " << std::hex << f << " " << g << std::endl;
                uint32_t temp = d;
                d = c;
                c = b;
                b = b + leftRotate(a + f + k[i] + w[g], s[i]);
                a = temp;
                // Debugging output
                std::cout <<" this is it " << i << "  :" << std::hex<< a << " "<< b << " " << c << " "<< d << std::endl;   
                
                
            }
            a += aa; b += bb; c += cc; d += dd;
        }
        
    public:
        static uint32_t swapEndian(uint32_t n) {
            return ((n & 0xFF000000) >> 24) |
                   ((n & 0x00FF0000) >> 8)  |
                   ((n & 0x0000FF00) << 8)  |
                   ((n & 0x000000FF) << 24);
        }
        static std::string hash(const std::string& input) {
            uint32_t a = 0x67452301;
            uint32_t b = 0xefcdab89;
            uint32_t c = 0x98badcfe;
            uint32_t d = 0x10325476;

            uint32_t aa = 0x67452301;
            uint32_t bb = 0xefcdab89;
            uint32_t cc = 0x98badcfe;
            uint32_t dd = 0x10325476;

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
            std::cout << "this is final" << " "<< std::hex << a-aa << " " << b-bb << " " << c-cc << " " << d-dd << std::endl;
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

static uint32_t leftRotate(uint32_t x, uint32_t c) {
    return (x << c) | (x >> (32 - c));
}

static uint32_t rightRotate(uint32_t x, uint32_t c) {
    return (x >> c) | (x << (32 - c));
}

void Crack(uint32_t a ,uint32_t b, uint32_t c, uint32_t d , int index)
{
    uint32_t a2 ,b2, c2, d2;
    a2 = a;
    b2 = b;
    c2 = c;
    d2 = d;

    uint32_t a1 ,b1, c1, d1;
    d1 = a2;
    c1 = d2;
    b1 = c2;
    a1 = b2 - b1;
    a1 = rightRotate(a1, MD5::s[index]);

    // cout << "unrotated a1 foutned  " << std::hex << a1 << endl;
    // cout << "unrotated a1 orginal  " << std::hex << 0x8844e77b + 0xe6e1d9fd + MD5::k[index] + 0 << endl;

    uint32_t f = 0;
    uint32_t w ;
    int g;
    if ( index < 16) {
        f = (b1 & c1) | ((~b1) & d1);
        g = index;
    } else if (index < 32) {
        f = (d1 & b1) | ((~d1) & c1);
        g = (5 * index + 1) % 16;
    } else if (index < 48) {
        f = b1 ^ c1 ^ d1;
        g = (3 * index + 5) % 16;
    } else {
        f = c1 ^ (b1 | (~d1));
        g = (7 * index) % 16;
    }
    if (false)
    {
        a1 = a1 - f - MD5::k[index] ;
        // a1 is actuall a1 + w[i] 
        cout << "before fuck " << " index is  " << index << std::hex << a1 << " " << b1 << " " << c1 << " " << d1 << endl  ;
    }
    else {
        uint32_t decoded = a1 - f - MD5::k[index];
        cout << "Base decoded value for index " << index << " is: " << std::hex << decoded << endl;
        
        // For index 62, assume decoded = candidate1 + candidate2.
        bool found = false;
        uint32_t cand1_final = 0, cand2_final = 0;
        // Adjust the search range as needed:
        for (int i = 0; i < decoded; i ++)
        {
            cout << " cand1 is " << decoded + i << " cand2 " << decoded;
        }
    }


    

    

}

void Crack8(uint32_t a ,uint32_t b, uint32_t c, uint32_t d , int index)
{
    uint32_t a2 ,b2, c2, d2;
    a2 = a;
    b2 = b;
    c2 = c;
    d2 = d;

    uint32_t a1 ,b1, c1, d1;
    d1 = a2;
    c1 = d2;
    b1 = c2;
    a1 = b2 - b1;
    a1 = rightRotate(a1, MD5::s[index]);

    // cout << "unrotated a1 foutned  " << std::hex << a1 << endl;
    // cout << "unrotated a1 orginal  " << std::hex << 0x8844e77b + 0xe6e1d9fd + MD5::k[index] + 0 << endl;

    uint32_t f = 0;
    uint32_t w ;
    int g;
    if ( index < 16) {
        f = (b1 & c1) | ((~b1) & d1);
        g = index;
    } else if (index < 32) {
        f = (d1 & b1) | ((~d1) & c1);
        g = (5 * index + 1) % 16;
    } else if (index < 48) {
        f = b1 ^ c1 ^ d1;
        g = (3 * index + 5) % 16;
    } else {
        f = c1 ^ (b1 | (~d1));
        g = (7 * index) % 16;
    }
    
    a1 = a1 - f - MD5::k[index] ;
    // a1 is actuall a1 + w[i] 
    cout << "before fuck " << " index is  " << index << std::hex << a1 << " " << b1 << " " << c1 << " " << d1 << endl  ;

    
    
    


    

    

}

void MD5_reverce(std::string hash )
{
    cout << "reverce function " << hash << endl;
    uint32_t a ,b, c, d;
    a = stoul(hash.substr(0, 8), nullptr, 16);
    b = stoul(hash.substr(8, 8), nullptr, 16);
    c = stoul(hash.substr(16, 8), nullptr, 16);
    d = stoul(hash.substr(24, 8), nullptr, 16);
    
    cout << "from rev  " << std::hex << a <<" "<< b << " " << c << " " << d << std::endl;
    a = MD5::swapEndian(a);
    b = MD5::swapEndian(b);
    c = MD5::swapEndian(c);
    d = MD5::swapEndian(d);

    cout << "from rev  swapEndian " << std::hex << a <<" "<< b << " " << c << " " << d << std::endl;

    a = a - 0x67452301;
    b = b - 0xefcdab89;
    c = c - 0x98badcfe;
    d = d - 0x10325476;

    cout << "from rev  pure " << std::hex << a <<" "<< b << " " << c << " " << d << std::endl;

    Crack(a, b, c, d, 63);
    // Crack(a, b, c, d, 62);

    
}



int main() {
   
    std::string result;
    result = MD5::hash("password");
    std::cout << "MD5 Hash: " << result << std::endl;
    
    MD5_reverce(result);

    uint32_t f = 0; // Define 'f' with an appropriate value
    cout << "alr " << 0x456ca61f + leftRotate(0x8844e77b + 0xe6e1d9fd + MD5::k[63] + 0, MD5::s[63]);

    return 0;

}