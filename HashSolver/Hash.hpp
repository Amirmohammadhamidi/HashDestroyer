#ifndef Hash_hpp
#define Hash_hpp

#include <iostream>
#include <algorithm>
class Hash
{
private:
    char *hash;
    char *type;
    char charSet[70];
    char *wordlistPath;
    bool isWomen;
    int length_lower_bound = 4;
    int length_upper_bound = 10;

public:
    Hash(char *hash)
    {
        this->hash = hash;
    };

    ~Hash();
    void set_type(char *type)
    {
        this->type = type;
    };
    void is_women(bool isWomen)
    {
        this->isWomen = isWomen;
    };
    void set_length_bounds(int length_lower_bound = 4, int length_upper_bound = 10)
    {
        this->length_lower_bound = length_lower_bound;
        this->length_upper_bound = length_upper_bound;
    };
    void loadcharSet(char *charSet, size_t size)
    {
        std::copy(charSet, charSet + size, this->charSet);
    };
    void loadWordList(char *wordlistPath)
    {
        this->wordlistPath = wordlistPath;
    };
    void crack();
    void wordlistCracker();
    void finish_cracking();
};

#endif