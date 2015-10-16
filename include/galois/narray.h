#ifndef _GALOIS_NARRAY_H_
#define _GALOIS_NARRAY_H_


#include <vector>
#include <Accelerate/Accelerate.h>

using namespace std;

namespace gs
{
    const int   NARRAY_DIM_ZERO = 0;
    const int   NARRAY_DIM_ONE = 1;
    
    template<typename T>
    class NArray
    {
    public:
        explicit NArray(int m);
        explicit NArray(int m, int n);
        explicit NArray(int m, int n, int o);
        explicit NArray(int m, int n, int o, int k);
        explicit NArray(vector<int>);
        NArray() = delete;
        NArray(const NArray& other) = delete;
        NArray& operator=(const NArray&) = delete;
        ~NArray();
        
        vector<int> get_dims() { return dims; }
        int get_size() {
            if (dims.empty()) return 0;
            int size = 1;
            for (auto d : dims) {
                size *= d;
            }
            return size;
        }
        T* get_dataptr() { assert(data); return data; }
        
        void copy_data(const vector<int> &, T*);
        void normalize_for(int dim);
        
    private:
        const vector<int> dims = {};
        T *data = nullptr;
    };
    template<typename T>
    using SP_NArray = shared_ptr<NArray<T>>;
    
    template<typename T>
    void GEMM(const char tA, const char tB,
              const T alpha, const SP_NArray<T> A, const SP_NArray<T> B,
              const T beta, const SP_NArray<T> C);
    
//    template<typename T>
//    void MAP (const function<T(T)>& f,
//              const SP_NArray<T> A, const SP_NArray<T> B,
//              const bool overwrite);
    template<typename T>
    void MAP_TO (const function<T(T)>& f, const SP_NArray<T> Y, const SP_NArray<T> X);
    template<typename T>
    void MAP_ADD(const function<T(T)>& f, const SP_NArray<T> Y, const SP_NArray<T> X);
    
    template<typename T>
    void PROJ_MAP_TO (const function<T(T)>& f,
                      const SP_NArray<T> Y, const SP_NArray<T> X,
                      const SP_NArray<T> idx);
    template<typename T>
    void PROJ_MAP_TO (const function<T(T)>& f,
                      const SP_NArray<T> Y, const SP_NArray<T> X,
                      const SP_NArray<T> idx);
    
    template<typename T>
    void SUB_MAP_TO (const function<T(T)>& f,
                    const SP_NArray<T> Y, const SP_NArray<T> X,
                    const SP_NArray<T> a, const SP_NArray<T> b);
    template<typename T>
    void SUB_MAP_ADD(const function<T(T)>& f,
                    const SP_NArray<T> Y, const SP_NArray<T> X,
                    const SP_NArray<T> a, const SP_NArray<T> b);

}

#endif
