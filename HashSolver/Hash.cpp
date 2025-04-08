#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <queue>
#include <mutex>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <condition_variable>
#include <sys/stat.h>
#include <cstring>
#include "Hash.hpp"
#include "../HashCrackerEngine.hpp"
using namespace std;

int max_size = 100;

class ThreadPool
{
public:
    explicit ThreadPool(size_t numThreads) : done(false)
    {
        for (size_t i = 0; i < numThreads; ++i)
        {
            threads.emplace_back([this, i]
                                 {
                while (!done) {
                    unique_lock<mutex> lock(queueMutex);
                    condition.wait(lock, [this] { return !tasks.empty() || done; });
                    if (done && tasks.empty()) break;
                    auto task = move(tasks.front());
                    tasks.pop();
                    lock.unlock();
                    task();
                } });
        }
    }

    template <class Task>
    void enqueue(Task &&task)
    {
        {
            lock_guard<mutex> lock(queueMutex);
            tasks.push(forward<Task>(task));
        }
        condition.notify_one();
    }

    ~ThreadPool()
    {
        {
            lock_guard<mutex> lock(queueMutex);
            done = true;
        }
        condition.notify_all();
        for (auto &t : threads)
        {
            t.join();
        }
    }

private:
    vector<thread> threads;
    queue<function<void()>> tasks;
    mutex queueMutex;
    condition_variable condition;
    atomic<bool> done;
};

int main()
{
    char *hash_str = new char[max_size];
    cout << "Enter your hash: ";
    cin.getline(hash_str, max_size);

    char *input = new char[max_size];
    cout << "Enter your wordlist path: ";
    cin.getline(input, max_size);

    string hashString(hash_str);
    string wordlistPath(input);

    delete[] hash_str;
    delete[] input;

    Hash *hash = new Hash(hashString);
    hash->loadWordList(wordlistPath);
    hash->crack();

    delete hash;
    return 0;
}

void Hash::crack()
{
    if (!this->wordlistPath.empty())
        wordlistCracker(true);
};

void Hash::wordlistCracker(bool flag)
{
    if (!flag)
    {
        // Open file using memory-mapping for efficient processing
        // fd : file descriptor
        // open file with read only flag
        int fd = open(this->wordlistPath.c_str(), O_RDONLY);
        if (fd == -1)
        {
            cout << "Error opening wordlist!" << endl;
            return;
        }

        struct stat sb;
        if (fstat(fd, &sb) == -1)
        {
            cout << "Error getting file info!" << endl;
            close(fd);
            return;
        }

        // Memory-map the file for fast access
        char *fileData = static_cast<char *>(mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
        if (fileData == MAP_FAILED)
        {
            cout << "Error memory-mapping file!" << endl;
            close(fd);
            return;
        }

        vector<thread> threads;
        atomic<bool> cracked(false);
        mutex print_mutex; // Mutex for safe printing to avoid multiple threads printing simultaneously

        // Worker function to process lines
        auto processLines = [&](int startOffset, int endOffset)
        {
            string line;
            int offset = startOffset;

            for (int i = startOffset; i < endOffset && !cracked.load(); ++i)
            {
                // If newline or null character is found, process the current line
                if (fileData[i] == '\n' || fileData[i] == '\0')
                {
                    line = string(fileData + offset, i - offset);
                    offset = i + 1;

                    if (line.empty())
                        continue;

                    if (MD5::hash(line) == this->hash) // Replace with your actual check
                    {
                        cracked.store(true);
                        lock_guard<mutex> lock(print_mutex);
                        cout << "Word found: " << line << " at offset " << startOffset << endl;
                        break;
                    }
                }
            }
        };

        // Create a thread pool to process chunks of data
        ThreadPool threadPool(20);
        int totalSize = sb.st_size;
        int chunkSize = totalSize / 20; // Split work into chunks for each thread

        // Enqueue tasks into the thread pool
        for (int i = 0; i < 20; ++i)
        {
            int startOffset = i * chunkSize;
            int endOffset = (i == 19) ? totalSize : startOffset + chunkSize; // Ensure the last thread handles the rest of the file
            threadPool.enqueue([=]
                               { processLines(startOffset, endOffset); });
        }

        // Wait for all threads to finish
        // The thread pool will automatically wait for all threads
        munmap(fileData, sb.st_size);
        close(fd);

        if (!cracked.load())
        {
            cout << "No matching word found." << endl;
        }
    }
    else
    {
        int charset_len = strlen(this->charset);
        int total_ids = charset_len * charset_len * charset_len;
        int *prefixes_counter = new int[total_ids](); // zero-initialize

        ifstream wordlist("../wordlists/rockyou.txt");
        fstream prefix_file("../wordlists/rockyou_prefix_dictionary.txt", ios::out);

        if (!wordlist || !prefix_file)
            return;

        string line;
        int max_frequency = 0;
        while (getline(wordlist, line))
        {
            if (line.size() < 3)
                continue;
            int id = string_to_id(line.substr(0, 3));
            prefixes_counter[id]++;
            if (prefixes_counter[id] > max_frequency)
            {
                max_frequency = prefixes_counter[id];
            }
        }
        wordlist.close();

        vector<string> arr[max_frequency + 1];
        for (int i = 0; i < total_ids; i++)
        {
            arr[prefixes_counter[i]].push_back(id_to_string(i));
        }

        for (int i = max_frequency; i >= 0; i--)
        {
            if (arr[i].empty())
                continue;
            for (size_t j = 0; j < arr[i].size() - 1; j++)
            {
                prefix_file << arr[i][j] << ",";
            }
            prefix_file << arr[i][arr[i].size() - 1] << ":" << i << endl;
        }
        prefix_file.close();

        delete[] prefixes_counter;
    }
}
string Hash::id_to_string(int id)
{
    int charset_len = strlen(this->charset);
    string res(3, ' ');
    res[2] = this->charset[id % charset_len];
    id /= charset_len;
    res[1] = this->charset[id % charset_len];
    id /= charset_len;
    res[0] = this->charset[id % charset_len];
    return res;
}

int Hash::string_to_id(string str)
{
    int charset_len = strlen(this->charset);
    int id = 0;
    id += this->mapper[str[0]];
    id += charset_len * this->mapper[str[1]];
    id += charset_len * charset_len * this->mapper[str[2]];
    return id;
}
