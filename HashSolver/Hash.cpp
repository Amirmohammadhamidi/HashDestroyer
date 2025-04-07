#include <iostream>
#include "Hash.hpp"
#include <fstream>

using namespace std;
int max_size = 100;

int main()
{
    char *input = new char[max_size];
    cin.getline(input, max_size);
    Hash *hash = new Hash(input);
    hash->loadWordList(input);
    hash->crack();
}

void Hash::crack()
{
    cout << this->wordlistPath << endl;
    if (this->wordlistPath && this->wordlistPath[0] != '\0')
        wordlistCracker();
};

void Hash::wordlistCracker()
{
    ofstream file;
    file.open(this->wordlistPath);
    if (!file)
    {
        cout << "file can't be open!!" << endl;
        return;
    }

    file << "push this to file!" << endl;
    file.close();
};
