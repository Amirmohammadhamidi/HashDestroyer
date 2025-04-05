#ifndef Hash_hpp
#define Hash_hpp

class Hash
{
private:
    char *hash;
    char *type;
    char charSet[70];
    char *passwordListPath;
    bool isWomen;
    int length_lower_bound = 4;
    int length_upper_bound = 10;

public:
    Hash(char *hash)
    {
        this->hash = hash;
    };

    ~Hash();
    void set_type(char *type);
    void is_women(bool isWomen);
    void set_length_bounds(int length_lower_bound = 4, int length_upper_bound = 10);
    void loadcharSet(char charSet[]);
    void loadPasswordList(char *passwordListPath);
    void start_cracking();
    void finish_cracking();
};

#endif