#ifndef _GALOIS_NARRAY_H_
#define _GALOIS_NARRAY_H_


#include <vector>
#include <Accelerate/Accelerate.h>

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
        T* get_dataptr() { assert(data); return data; }
        
        void copy_data(const vector<int> &, T*);
        
    private:
        vector<int> dims = {};
        int size = 0;
        T *data = nullptr;
    };
    template<typename T>
    using SP_NArray = shared_ptr<NArray<T>>;
    
    void GEMM(const char tA, const char tB,
              const float alpha, const SP_NArray<float> A, const SP_NArray<float> B,
              const float beta, const SP_NArray<float> C);
    void GEMM(const char tA, const char tB,
              const double alpha, const SP_NArray<double> A, const SP_NArray<double> B,
              const double beta, const SP_NArray<double> C);
    
    template<typename T>
    void MAP_TO (const function<T(T)>& f, const SP_NArray<T> A, const SP_NArray<T> B);
    template<typename T>
    void MAP_ADD(const function<T(T)>& f, const SP_NArray<T> A, const SP_NArray<T> B);

}

#endif
