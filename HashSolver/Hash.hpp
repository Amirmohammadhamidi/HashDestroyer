#ifndef Hash_hpp
#define Hash_hpp

#include <iostream>
#include <algorithm>
#include <string>
#include <map>

using namespace std;

class Hash
{
private:
    string hash = "";
    string type = "";
    const char *charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!#$%*()^&";
    map<char, int> mapper;
    string wordlistPath = "";
    bool isWomen;
    int length_lower_bound = 4;
    int length_upper_bound = 10;

public:
    Hash(string hash)
    {
        this->hash = hash;
        for (int i = 0; charset[i] != '\0'; ++i)
        {
            mapper[charset[i]] = i;
        }
    };

    ~Hash() {

    };
    void set_type(string type)
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

    void loadWordList(string wordlistPath)
    {
        this->wordlistPath = wordlistPath;
    };
    void crack();
    void wordlistCracker(bool flag);
    string id_to_string(int id);
    int string_to_id(string str);
    void sort_prefixes_wordlist(string path, int max_frequncy);
    void finish_cracking();
};

#endif