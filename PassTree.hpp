#ifndef PassTree_hpp
#define PassTree_hpp
#include "Hash.hpp"
class PassTree
{
private:
    bool find_password = false;
    int thread_counts = 0;
    float seconds_passed = 0.0f;

public:
    PassTree(Hash hash);
    ~PassTree();
    void generate_passTree();
};
#endif