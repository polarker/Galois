#ifndef _GALOIS_NARRAY_H_
#define _GALOIS_NARRAY_H_


#include <vector>

using namespace std;

namespace gs
{
    
    template<typename T>
    class NArray
    {
    public:
        NArray(int m, int n＝0, int o=0, int k=0);
        NArray(vector<int>);
        NArray() = delete;
        NArray(const NArray& other) = delete;
        NArray& operator=(const NArray&) = delete;
        ~NArray();
        
        vector<int> get_dims() { return dims; }
        int get_size() { return size; }
        
    private:
        vector<int> dims = {};
        int size = 0;
        T *data = nullptr;
    };
    template<typename T>
    using SP_NArray = shared_ptr<NArray<T>>;

}

#endif
