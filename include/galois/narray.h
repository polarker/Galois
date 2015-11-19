#ifndef _GALOIS_NARRAY_H_
#define _GALOIS_NARRAY_H_

#include "galois/utils.h"
#include <random>
#include <vector>
#include <Accelerate/Accelerate.h>

using namespace std;

namespace gs
{
    template<typename T>
    class NArray;
    template<typename T>
    using SP_NArray = shared_ptr<NArray<T>>;

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
        int get_size() { return size; }
        T* get_data() { CHECK(data, "data should be non-empty"); return data; }
        bool opaque() { return data_opaque; }
        void reopaque() { data_opaque = true; }
        void setclear() { data_opaque = false; }
        
//        void copy_from(const vector<int> &, const T*);
        void copy_from(const SP_NArray<T>);
        void copy_from(const vector<int> &, const SP_NArray<T>);
        void copy_from(const vector<int> &, int, const SP_NArray<T>);
        void copy_from(const int, const int, const SP_NArray<T>);
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
        int size = 0;
        T *data = nullptr;
        bool data_opaque = true;
    };
    template<typename T>
    default_random_engine NArray<T>::galois_rn_generator(0);
    
    template<typename T>
    ostream& operator<<(std::ostream &strm, const SP_NArray<T> M) {
        // currently, only surpport matrix
        auto M_dims = M->get_dims();
        auto M_ptr = M->get_data();
        CHECK(M_dims.size() <= 2, "only support upto 2 dimensional array");
        int m = 0;
        int n = 0;
        if (M_dims.size() == 1) {
            m = 1;
            n = M_dims[0];
        } else {
            m = M_dims[0];
            n = M_dims[1];
        }
        
        strm << '[' << endl;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                strm << M_ptr[i*n + j] << '\t';
            }
            strm << endl;
        }
        strm << ']' << endl;
        return strm;
    }
    
}

#include "narray_functors.cpp"

#endif
