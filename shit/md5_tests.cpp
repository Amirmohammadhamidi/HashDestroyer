#include <iostream>
#include <vector>
#include <string>
#include <iomanip>

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
    

struct TestCase {
    std::string input;
    std::string expected;
};

int main() {
    std::vector<TestCase> tests = {
        // Known test vectors from RFC 1321:
        {"", "d41d8cd98f00b204e9800998ecf8427e"},
        {"a", "0cc175b9c0f1b6a831c399e269772661"},
        {"abc", "900150983cd24fb0d6963f7d28e17f72"},
        {"message digest", "f96b697d7cb7938d525a2f31aaf161d0"},
        {"abcdefghijklmnopqrstuvwxyz", "c3fcd3d76192e4007dfb496cca67e13b"},
        {"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", "d174ab98d277d9f5a5611c2c9f419d9f"},
        {"12345678901234567890123456789012345678901234567890123456789012345678901234567890", "57edf4a22be3c955ac49da2e2107b67a"},
        
        // Additional 143 test cases with their expected MD5 hashes:
        {"test_case_7", "18b049cc8d8535787929df716f9f4e68"},
        {"test_case_8", "cd8927eef9c7853e5ab814d0091c9de0"},
        {"test_case_9", "2863f01f5f016ce0fe855b2c45c77fec"},
        {"test_case_10", "22ea1c649c82946aa6e479e1ffd321e4"},
        {"test_case_11", "c72b9698fa1927e1dd9cabe10a245aaa"},
        {"test_case_12", "6f825aa2a6b729b774b58a138971d65d"},
        {"test_case_13", "295077c264025a9215ea9ef5ae6437ed"},
        {"test_case_14", "d2d363cb4800c8a50a48927e271a6872"},
        {"test_case_15", "8e03a2e6546ce0829f4f5fa3c6430964"},
        {"test_case_16", "5509ee2abdf0ebec754b25409dc5b0a6"},
        {"test_case_17", "92c2ba4950708e46e81bc2fe11287892"},
        {"test_case_18", "2559bcef72f3969484ac66f7242eb456"},
        {"test_case_19", "51e96eada6ec7f2887fe8eaadc7a9e74"},
        {"test_case_20", "0b37fa840c1d8f5818941311be854a2c"},
        {"test_case_21", "df56703f71cfa76e17b2055d7cab3e89"},
        {"test_case_22", "bdac809d3d1d4e3fa3e43f33ea37260c"},
        {"test_case_23", "09c527ee62ef3de0c85897e338fe1bf5"},
        {"test_case_24", "1090c2ffaaf65ec86e8e7f4bb598f1d2"},
        {"test_case_25", "f0c5fc39714c89ecb8a60989d5c118fc"},
        {"test_case_26", "306c59c3e0f31c3037977644280da586"},
        {"test_case_27", "05214e40b99422dc3036f35b953d1e97"},
        {"test_case_28", "a83d566c05f6e4ab9fe4642dd92233ba"},
        {"test_case_29", "bd33747f44900e3c479b223eff959083"},
        {"test_case_30", "1ef9ea1e663adb75de1c78cae8951f30"},
        {"test_case_31", "0fe4dce2299fabec1507d1a5c1b932b9"},
        {"test_case_32", "d41385aba355f6f58d665fb9025a6fe0"},
        {"test_case_33", "e85098ec2c5dbc2f96a0945865a03254"},
        {"test_case_34", "99b1837548cd9354f0c9a81c58c95b70"},
        {"test_case_35", "6a7db4a08030bab5351f0f1a33b05a9b"},
        {"test_case_36", "43c401d32282e32cb515f8c0071cca78"},
        {"test_case_37", "89a4d518bd7f6cb6e23c6879d2c4937c"},
        {"test_case_38", "074c0fe87120ff1681e9eccdb9c7d275"},
        {"test_case_39", "5d1eb83ad370a03c51bf10d26f706e52"},
        {"test_case_40", "900c1d8f7c9992608c2f8d338f9a5630"},
        {"test_case_41", "9fc3f9063d1b89e9261d84d0dab4b1ac"},
        {"test_case_42", "2b8f5327e21c7337c6b261e7d2220941"},
        {"test_case_43", "0c4e2f071bc42849ee1f12f947ea8598"},
        {"test_case_44", "c940c95da06f4d6038b866289ef029fc"},
        {"test_case_45", "e83478a7c431e077412f85994fc28c74"},
        {"test_case_46", "d6454eaa4175c9635ccb7dab76407dee"},
        {"test_case_47", "625ab10003156ca65231be9c28299f9d"},
        {"test_case_48", "5efb954d9dda3cb3e0e12c0b571c5d92"},
        {"test_case_49", "26f7c024498d39ad214967fbf0921ad5"},
        {"test_case_50", "20ade84fb272d4e20eaa087d13a70e64"},
        {"test_case_51", "9ad53387144a7ddf9ba09d7d7da28e17"},
        {"test_case_52", "26483dd481bbc38bbb14ecf35f77357f"},
        {"test_case_53", "2aadf97bf734a64da3a25dd308985e18"},
        {"test_case_54", "d180a9024e811d8b3778e1d34c7073be"},
        {"test_case_55", "ff4bb5a4931708bed98e3bb42aabf1c3"},
        {"test_case_56", "86e9799cb0c6e259eb51aa818c9ee7b1"},
        {"test_case_57", "e043de8822b1d011b85efcf3d317ca3b"},
        {"test_case_58", "8e5a5dd4d24eae4f481d2454d7205524"},
        {"test_case_59", "fdd7136ca790b6dfa90d1ce26cbb4a42"},
        {"test_case_60", "104be12361c4d7cf4e42ffaa0d233a69"},
        {"test_case_61", "6a0d83351fa18e69be993c3259d1f124"},
        {"test_case_62", "eb4adf83c42cb1b1cd1635a6c59fd732"},
        {"test_case_63", "3f7574dac82606f294f0efa13c1cdb09"},
        {"test_case_64", "9c579e75f4b7a6238c0e3c46428bb453"},
        {"test_case_65", "0366d2f005e6da919369b49f298d10ab"},
        {"test_case_66", "d0a4c1a312ad861b7b5caae64bd18862"},
        {"test_case_67", "3fd88be813af62abb97f4ab9496ef75e"},
        {"test_case_68", "ea55e0a955c7656876063f24383bd110"},
        {"test_case_69", "793a463d32c454e59cbb14bd5a01b075"},
        {"test_case_70", "185378e760c5d26d2219f8364a25dd4c"},
        {"test_case_71", "1d82d8e823023425b1d850f44cfb4161"},
        {"test_case_72", "fd0a3f33541b180f6c642f895c4e9f82"},
        {"test_case_73", "de2d9532ce7c2a1b09b6a05253d5301a"},
        {"test_case_74", "610d2d1e44c909fcb3e9f070b46dfcf6"},
        {"test_case_75", "ca1c5ad70eb53af8365ea9888b41fc1f"},
        {"test_case_76", "8fb5229cef63a7790935c07b1849c255"},
        {"test_case_77", "24735cbdf6a578927b4ae9eaeaea8ef1"},
        {"test_case_78", "468d14c0007ca053a74f0cd995aadb54"},
        {"test_case_79", "e513966c04617735a1824c31b62c5062"},
        {"test_case_80", "bd9b32493543e4d217d216a6fe96d174"},
        {"test_case_81", "912e735cad6ef7c461b5da9cb778bd2c"},
        {"test_case_82", "b72a6000d8b7e03682f13a808a245599"},
        {"test_case_83", "c234272c23cc5e684eab9e666f888edf"},
        {"test_case_84", "fa716a4b75ad3962373cccf5fc9dc1e7"},
        {"test_case_85", "0d2e910d5332a13c8b3fa9a56e0b5baa"},
        {"test_case_86", "4c0402a357dfd1ba45336c9688a10097"},
        {"test_case_87", "3c436f2e1ff399025e36bf48d6fa0770"},
        {"test_case_88", "63f9acfb924992bbf478b4b395880405"},
        {"test_case_89", "58bb09fa9508957c9a4b0484e4390eba"},
        {"test_case_90", "7a0e26a6611d147a277335e63e51b6f0"},
        {"test_case_91", "5c0e30ba9b8a6443b42ef8655725f9eb"},
        {"test_case_92", "8fa500eb6482af02cbc55bc346c1d8ce"},
        {"test_case_93", "a1af8c74dd08febb1b96ac562da10415"},
        {"test_case_94", "47572214e13df86805b4c5beea0ca017"},
        {"test_case_95", "473706c8c3aa95700c7d5b7799cb8766"},
        {"test_case_96", "6291e3395ad72327a91acea496c1c17c"},
        {"test_case_97", "e4ffafc9398f618853cc9870f0c31d5b"},
        {"test_case_98", "1b2b475a9d1ea5bf8c1a9ec0691a6b83"},
        {"test_case_99", "00e27b6191ad9c13677720fa7c228e3d"},
        {"test_case_100", "49af199a7a59c7b3f67a5766652c074f"},
        {"test_case_101", "24173c3fd27dc551c0db530403e0ff9e"},
        {"test_case_102", "cb6c8480cc978e528a9cf20f8f7ab443"},
        {"test_case_103", "4e16c2c4cdb3249c575697a8fb2077b9"},
        {"test_case_104", "ff16538a88f2f4be16aa149963e9da76"},
        {"test_case_105", "180d310e025ca0c64562fb131de9beff"},
        {"test_case_106", "53e8eec81e841be6e4585a2ee444e7b7"},
        {"test_case_107", "4a6b77a0617fd73f46af1dea2c961701"},
        {"test_case_108", "af5d4f0a1acdf831fd3989938070978f"},
        {"test_case_109", "c3c8fb6a5159cbf9eb96d8379e850f4e"},
        {"test_case_110", "e3d1b7586e36dcd6a9b6ff3758e69f16"},
        {"test_case_111", "80f25a869dbb97bb9403a1198ad3b88f"},
        {"test_case_112", "4362d76aa5a9659ed106657c4e206ca7"},
        {"test_case_113", "fc734ce49ac9621385ad06179ac11d59"},
        {"test_case_114", "4fc2137b07d8744267ea5dab6c30b62d"},
        {"test_case_115", "a9abf3e7211640bb0ec4ce827f5d58ee"},
        {"test_case_116", "b5b4c8eeae5db40f5a1f39bcf295fbfc"},
        {"test_case_117", "4290b3c8423e6125eb556dca15d277fa"},
        {"test_case_118", "e0b4a8c9e8775744cf7bee34cd9b7840"},
        {"test_case_119", "c2ae4c4323bca9256b8306d252c716ec"},
        {"test_case_120", "afc56369e3f0c5a95f292e90357ec156"},
        {"test_case_121", "856dee6ca67aedc8c55d18bfe686d7fc"},
        {"test_case_122", "c1c449287c3e9e3a0c8a944a637d6d9c"},
        {"test_case_123", "a9fc638693a5189a4ac2bfa71db153c6"},
        {"test_case_124", "1adc0f7b31381c21e40eef951f6489e0"},
        {"test_case_125", "76f1a0cbff0112d2728d1ba63a9124f7"},
        {"test_case_126", "2761ecd6fbb206243edc04e5e478aa18"},
        {"test_case_127", "4c23453e3520a9c3da9bbd72bd9b05dd"},
        {"test_case_128", "c9194dd8fc9ea5b5edbfcd875ac94553"},
        {"test_case_129", "8499b2d15b6b1307e8db889e0fb6742c"},
        {"test_case_130", "6748a3dee07db9d5dd8140d94cb5a0bb"},
        {"test_case_131", "76b33e6261e5c6793b8a2b5a5eb9dfb3"},
        {"test_case_132", "846e514940d53ce07c91f049694bb459"},
        {"test_case_133", "a5610d2be35eb92c2a7ecdd6bbbe729a"},
        {"test_case_134", "cbc82eecd0117ed80a0779f0aa6c3453"},
        {"test_case_135", "a9b269129b9649169d6c1cab6b17d214"},
        {"test_case_136", "b8afe266c46ad2060eddca5b5f5a4cd1"},
        {"test_case_137", "4d3c7e2d3ea0614f5549c0d2bfcb50b7"},
        {"test_case_138", "ebd780c29f7db8f8b2fa8e236d06bdc4"},
        {"test_case_139", "9d512e90ecedcf80a47c96bf67ce2333"},
        {"test_case_140", "37b47ad79fc50b6067ee6c29173f039e"},
        {"test_case_141", "fb6ec11e9ecb6f0ab59e3b41227202a6"},
        {"test_case_142", "036492ff8cf81f2373304bb181b33179"},
        {"test_case_143", "f7e7519152a860801687d435b17a1324"},
        {"test_case_144", "99fb50e5fe2c09c0d29b0309a7c70ab7"},
        {"test_case_145", "9cd3fa325ca1051eedab8b2264fca104"},
        {"test_case_146", "35ecfdcabe198f0c5321ed041c27f797"},
        {"test_case_147", "1afbfa352f4641423c86b48cfe4148c5"},
        {"test_case_148", "91d880c175f156ada1e9114e4f10cd95"},
        {"test_case_149", "e57a8687d32a3403d33a15a460bb5757"}
    };

    int passed = 0;
    std::cout << "Running MD5 test suite with " << tests.size() << " test cases...\n" << std::endl;
    
    for (size_t i = 0; i < tests.size(); i++) {
        std::string computed = MD5::hash(tests[i].input);
        bool isPassed = (computed == tests[i].expected);
        
        if (isPassed) {
            std::cout << "Test " << i + 1 << " passed." << std::endl;
            passed++;
        }
        else {
            std::cout << "Test " << i + 1 << " FAILED:" << std::endl
                      << "  Input:    \"" << tests[i].input << "\"" << std::endl
                      << "  Expected: " << tests[i].expected << std::endl
                      << "  Computed: " << computed << std::endl;
        }
    }
    
    std::cout << "\nTest summary: " << passed << " out of " << tests.size() 
              << " tests passed (" << (passed * 100 / tests.size()) << "%)" << std::endl;
    
    if (passed == tests.size()) {
        std::cout << "\nAll tests passed! The MD5 implementation is correct." << std::endl;
    } else {
        std::cout << "\nSome tests failed. The MD5 implementation needs fixing." << std::endl;
    }
    
    return (passed == tests.size()) ? 0 : 1;
}