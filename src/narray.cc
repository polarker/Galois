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
        
        int size = m;
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
        int size = 1;
        for (auto m : nums) {
            assert(m > 0);
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
    
    template<typename T>
    void NArray<T>::copy_data(const vector<int> &dims, T* data) {
        assert(dims == this->dims);
        for (int i = 0; i < this->get_size(); i++) {
            this->data[i] = data[i];
        }
    }
    
    template<typename T>
    void NArray<T>::norm_for(int dim) {
        // currently, only two dimensional array are supported
        assert(dim == NARRAY_DIM_ZERO || dim == NARRAY_DIM_ONE);
        assert(this->dims.size() == 2);
        
        if (dim == NARRAY_DIM_ZERO) {
            for (int i = 0; i < this->dims[0]; i++) {
                T sum = 0;
                for (int j = 0; j < this->dims[1]; j++) {
                    sum += this->data[i*this->dims[1] + j];
                }
                for (int j = 0; j < this->dims[1]; j++) {
                    this->data[i*this->dims[1] + j] /= sum;
                }
            }
        }
        if (dim == NARRAY_DIM_ONE) {
            for (int i = 0; i < this->dims[1]; i++) {
                T sum = 0;
                for (int j = 0; j < this->dims[0]; j++) {
                    sum += this->data[j*this->dims[1] + i];
                }
                for (int j = 0; j < this->dims[1]; j++) {
                    this->data[i*this->dims[1] + j] /= sum;
                }
            }
        }
    }
    
    inline void _GEMM(const enum CBLAS_ORDER _order,
                      const enum CBLAS_TRANSPOSE _tranA, const enum CBLAS_TRANSPOSE _tranB,
                      const int _M, const int _N, const int _K,
                      const float _alpha,
                      const float *_A, const int _lda,
                      const float *_B, const int _ldb,
                      const float _beta,
                      float *_C, const int _ldc) {
        cblas_sgemm(_order, _tranA, _tranB, _M, _N, _K, _alpha, _A, _lda, _B, _ldb, _beta, _C, _ldc);
    }
    
    inline void _GEMM(const enum CBLAS_ORDER _order,
                      const enum CBLAS_TRANSPOSE _tranA, const enum CBLAS_TRANSPOSE _tranB,
                      const int _M, const int _N, const int _K,
                      const double _alpha,
                      const double *_A, const int _lda,
                      const double *_B, const int _ldb,
                      const double _beta,
                      double *_C, const int _ldc) {
        cblas_dgemm(_order, _tranA, _tranB, _M, _N, _K, _alpha, _A, _lda, _B, _ldb, _beta, _C, _ldc);
    }
    
    template<typename L>
    void GEMM(const char tA, const char tB,
              const L alpha, const SP_NArray<L> A, const SP_NArray<L> B,
              const L beta, const SP_NArray<L> C) {
        assert(tA == 'T' || tA == 'N');
        assert(tB == 'T' || tB == 'N');
        assert(A->get_dims().size() == 2);
        assert(B->get_dims().size() == 2);
        auto A_dims = A->get_dims();
        auto B_dims = B->get_dims();
        auto C_dims = C->get_dims();
        auto t_A = tA=='T' ? CblasTrans : CblasNoTrans;
        auto t_B = tB=='T' ? CblasTrans : CblasNoTrans;
        auto M   = tA=='T' ? A_dims[1] : A_dims[0];
        auto K_A = tA=='T' ? A_dims[0] : A_dims[1];
        auto K_B = tB=='T' ? B_dims[1] : B_dims[0];
        auto N   = tA=='T' ? B_dims[0] : B_dims[1];
        assert(K_A == K_B);
        assert(C->get_dims() == vector<int>({M, N}));
        auto K   = K_A;
        auto lda = A_dims[1];
        auto ldb = B_dims[1];
        auto ldc = C_dims[1];
        _GEMM(CblasRowMajor,
              t_A, t_B,
              M, N, K,
              alpha,
              A->get_dataptr(), lda,
              B->get_dataptr(), ldb,
              beta,
              C->get_dataptr(), ldc);
    }
    
    template<typename T>
    void MAP (const function<T(T)>& f,
              const SP_NArray<T> Y, const SP_NArray<T> X,
              const bool overwrite) {
        assert(Y->get_dims() == X->get_dims());
        auto Y_ptr = Y->get_dataptr();
        auto X_ptr = X->get_dataptr();
        for (int i = 0; i < Y->get_size(); i++) {
            if (overwrite) {
                Y_ptr[i] = f(X_ptr[i]);
            } else {
                Y_ptr[i] += f(X_ptr[i]);
            }
        }
    }
    
    
    template<typename T>
    void MAP_TO (const function<T(T)>& f, const SP_NArray<T> Y, const SP_NArray<T> X) {
        MAP(f, Y, X, true);
    }
    
    template<typename T>
    void MAP_ADD(const function<T(T)>& f, const SP_NArray<T> Y, const SP_NArray<T> X) {
        MAP(f, Y, X, false);
    }
    
    // currently, only two dimensional array are supported
    template<typename T>
    void SUB_MAP (const function<T(T)>& f,
                  const SP_NArray<T> Y, const SP_NArray<T> X,
                  const SP_NArray<int> a, const SP_NArray<int> b,
                  const bool overwrite) {
        assert(Y->get_dims() == X->get_dims());
        assert(Y->get_dims().size() == 2);
        int stride = Y->get_dims()[1];
        assert(a->get_size() == b->get_size());
        int size = a->get_size();
        
        auto Y_ptr = Y->get_dataptr();
        auto X_ptr = X->get_dataptr();
        auto a_ptr = a->get_dataptr();
        auto b_ptr = b->get_dataptr();
        for (int i = 0; i < size; i++) {
            if (overwrite) {
                Y_ptr[a_ptr[i]*stride + b_ptr[i]] = f(X_ptr[i]);
            } else {
                Y_ptr[a_ptr[i]*stride + b_ptr[i]] += f(X_ptr[i]);
            }
        }
    }
    
    // currently, only two dimensional array are supported
    template<typename T>
    void SUB_MAP_TO (const function<T(T)>& f,
                     const SP_NArray<T> Y, const SP_NArray<T> X,
                     const SP_NArray<int> a, const SP_NArray<int> b) {
        SUB_MAP(f, Y, X, a, b, true);
    }
    
    // currently, only two dimensional array are supported
    template<typename T>
    void SUB_MAP_ADD(const function<T(T)>& f,
                     const SP_NArray<T> Y, const SP_NArray<T> X,
                     const SP_NArray<int> a, const SP_NArray<int> b) {
        SUB_MAP(f, Y, X, a, b, false);
    }
    
    
    template class NArray<float>;
    template class NArray<double>;
    template void GEMM(const char tA, const char tB,
                       const float alpha, const SP_NArray<float> A, const SP_NArray<float> B,
                       const float beta, const SP_NArray<float> C);
    template void GEMM(const char tA, const char tB,
                       const double alpha, const SP_NArray<double> A, const SP_NArray<double> B,
                       const double beta, const SP_NArray<double> C);
    //    template void MAP (const function<float(float)>& f,
    //                       const SP_NArray<float> A, const SP_NArray<float> B,
    //                       const bool overwrite);
    //    template void MAP (const function<double(double)>& f,
    //                       const SP_NArray<double> A, const SP_NArray<double> B,
    //                       const bool overwrite);
    template void MAP_TO (const function<float(float)>& f, const SP_NArray<float> Y, const SP_NArray<float> X);
    template void MAP_TO (const function<double(double)>& f, const SP_NArray<double> Y, const SP_NArray<double> X);
    template void MAP_ADD(const function<float(float)>& f, const SP_NArray<float> Y, const SP_NArray<float> X);
    template void MAP_ADD(const function<double(double)>& f, const SP_NArray<double> Y, const SP_NArray<double> X);
    template void SUB_MAP_TO (const function<float(float)>& f,
                              const SP_NArray<float> Y, const SP_NArray<float> X,
                              const SP_NArray<int> a, const SP_NArray<int> b);
    template void SUB_MAP_TO (const function<double(double)>& f,
                              const SP_NArray<double> Y, const SP_NArray<double> X,
                              const SP_NArray<int> a, const SP_NArray<int> b);
    template void SUB_MAP_ADD(const function<float(float)>& f,
                              const SP_NArray<float> Y, const SP_NArray<float> X,
                              const SP_NArray<int> a, const SP_NArray<int> b);
    template void SUB_MAP_ADD(const function<double(double)>& f,
                              const SP_NArray<double> Y, const SP_NArray<double> X,
                              const SP_NArray<int> a, const SP_NArray<int> b);
    
}
