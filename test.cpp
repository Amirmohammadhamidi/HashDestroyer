#include <mutex>
#include <stdlib.h>
#include <stdio.h>
int main(int argc, char const *argv[])
{
    std::mutex mtx;

    printf("hello world");
    return 0;
}
