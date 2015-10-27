#ifndef _GALOIS_UTILS_H_
#define _GALOIS_UTILS_H_

#include <vector>
#include <cstdio>
#include <cstdlib>

#define CHECK(A, M, ...) if(!(A)) { fprintf(stderr, "[ERROR] %s:%d: " M "\n", __FILE__, __LINE__, ##__VA_ARGS__); exit(EXIT_FAILURE); }

using namespace std;

namespace gs {
    
    template <typename T>
    const bool Contains(vector<T>& Vec, const T& Element)
    {
        if (find(Vec.begin(), Vec.end(), Element) != Vec.end())
            return true;
        else
            return false;
    }
    
}

#endif
