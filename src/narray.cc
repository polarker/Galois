# include "galois/narray.h"

namespace gs
{
    
    template<typename T>
    NArray<T>::NArray(int m) {
        NArray<T>({m});
    }
    
    template<typename T>
    NArray<T>::NArray(int m, int n) {
        NArray<T>({m, n});
    }
    
    template<typename T>
    NArray<T>::NArray(int m, int n, int o) {
        NArray<T>({m, n, o});
    }
    
    template<typename T>
    NArray<T>::NArray(int m, int n, int o, int k) {
        NArray<T>({m, n, o, k});
    }
    
    template<typename T>
    NArray<T>::NArray(initializer_list<int> nums) : dims{} {
        NArray<T>(vector<int>(nums));
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