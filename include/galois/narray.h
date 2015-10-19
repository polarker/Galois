#ifndef _GALOIS_NARRAY_H_
#define _GALOIS_NARRAY_H_


#include <random>
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
        static default_random_engine galois_rn_generator;
        
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
            assert(!dims.empty());
            int size = 1;
            for (auto d : dims) {
                size *= d;
            }
            return size;
        }
        T* get_data() { assert(data); return data; }
        bool opaque() { return data_opaque; }
        void set_opaque(bool b) { data_opaque = b; }
        
        void copy_data(const vector<int> &, T*);
        void uniform(T lower, T upper) {
            // future : move random generator to a single file
            uniform_real_distribution<T> distribution(lower, upper);
            for (int i = 0; i < get_size(); i++) {
                data[i] = distribution(galois_rn_generator);
            }
            data_opaque = false;
        }
        void normalize_for(int dim);
        void fill(T x) {
            for (int i = 0; i < get_size(); i++) {
                data[i] = x;
            }
            data_opaque = false;
        }
        
    private:
        const vector<int> dims = {};
        T *data = nullptr;
        bool data_opaque = true;
    };
    template<typename T>
    using SP_NArray = shared_ptr<NArray<T>>;
    template<typename T>
    default_random_engine NArray<T>::galois_rn_generator(0);
    
    template<typename T>
    ostream& operator<<(std::ostream &strm, const SP_NArray<T> M) {
        // currently, only surpport matrix
        auto M_dims = M->get_dims();
        auto M_ptr = M->get_data();
        assert(M_dims.size() <= 2);
        int m = 0;
        int n = 0;
        if (M_dims.size() == 1) {
            m = 1;
            n = M_dims[0];
        } else {
            m = M_dims[0];
            n = M_dims[1];
        }
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                strm << M_ptr[i*n + j] << '\t';
            }
            strm << endl;
        }
        return strm;
    }
    
    template<typename T>
    void SUM_POSITIVE_VALUE (const SP_NArray<T> A, T *res);
    
    // currently, only two dimensional array are supported
    template<typename T>
    void ADD_TO_ROW (const SP_NArray<T> A, const SP_NArray<T> a);
    
//    template<typename T>
//    void GEMM (const char tA, const char tB,
//               const T alpha, const SP_NArray<T> A, const SP_NArray<T> B,
//               const T beta, const SP_NArray<T> C);
    template<typename T>
    void GEMM (const SP_NArray<T> Y,
               const char tA, const char tB,
               const SP_NArray<T> A, const SP_NArray<T> B);
    
    template<typename T>
    void MAP (const SP_NArray<T> Y,
                 const function<T(T)>& f,
                 const SP_NArray<T> X);
    template<typename T>
    void MAP (const SP_NArray<T> Y,
                 const function<T(T, T)>& f,
                 const SP_NArray<T> X, const SP_NArray<T> Z);
    
    // currently, only two dimensional array are supported
    template<typename T>
    void PROJ_MAP (const SP_NArray<T> Y,
                      const function<T(T)>& f,
                      const SP_NArray<T> X,
                      const SP_NArray<T> idx);
    
    // currently, only two dimensional array are supported
    template<typename T>
    void SUB_MAP (const SP_NArray<T> Y,
                     const function<T(T)>& f,
                     const SP_NArray<T> X,
                     const SP_NArray<T> a, const SP_NArray<T> b);
    
}

#endif
