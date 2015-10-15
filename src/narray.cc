# include "galois/narray.h"
# include <cassert>

namespace gs
{
    
    template<typename T>
    NArray<T>::NArray(int m, int n, int o, int k) {
        assert(m > 0);
        assert(n >= 0);
        assert(o >= 0);
        assert(k >= 0);

        size = m;
        dims.push_back(m);
        if (n > 0) {
            size *= n;
            dims.push_back(n);
        }
        if (o > 0) {
            size *= o;
            dims.push_back(o);
        }
        if (k > 0) {
            size *= k;
            dims.push_back(k);
        }
        data = new T[size];
    }
    
    template<typename T>
    NArray<T>::NArray(vector<int> nums) : dims{} {
        size = 1;
        for (auto m : nums) {
            dims.push_back(m);
            size *= m;
        }
        data = new T[size];
    }
    
    template<typename T>
    NArray<T>::~NArray() {
        if (data) {
            delete[] data;
        }
    }
    
    template class NArray<float>;
    template class NArray<double>;

}
