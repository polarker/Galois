# include "galois/narray.h"
# include <cassert>

namespace gs
{
    template<typename T>
    NArray<T>::NArray(int m) : dims{m} {
        assert(m > 0);
        data = new T[get_size()];
    }
    
    template<typename T>
    NArray<T>::NArray(int m, int n) : dims{m, n} {
        assert(m > 0);
        assert(n > 0);
        data = new T[get_size()];
    }
    
    template<typename T>
    NArray<T>::NArray(int m, int n, int o) : dims{m, n, o} {
        assert(m > 0);
        assert(n > 0);
        assert(o > 0);
        data = new T[get_size()];
    }
    
    template<typename T>
    NArray<T>::NArray(int m, int n, int o, int k) : dims{m, n, o, k} {
        assert(m > 0);
        assert(n > 0);
        assert(o > 0);
        assert(k > 0);
        data = new T[get_size()];
    }
    
    template<typename T>
    NArray<T>::NArray(vector<int> nums) : dims{nums} {
        for (auto m : nums) {
            assert(m > 0);
        }
        data = new T[get_size()];
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
    void NArray<T>::normalize_for(int dim) {
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
    
    template<typename T>
    void SUM_POSITIVE_VALUE (const SP_NArray<T> A, T *res) {
        T sum = 0;
        auto A_ptr = A->get_dataptr();
        auto A_size = A->get_size();
        for (int i = 0; i < A_size; i++) {
            sum += A_ptr[i];
        }
        *res = sum;
    }
    
    template<typename T>
    void ADD_TO_ROW (const SP_NArray<T> A, const SP_NArray<T> b) {
        auto A_dims = A->get_dims();
        auto b_dims = b->get_dims();
        assert(A_dims.size() == 2);
        assert(b_dims.size() == 1);
        assert(A_dims[1] == b_dims[0]);
        
        auto m = A_dims[0];
        auto A_ptr = A->get_dataptr();
        auto n = A_dims[1];
        auto b_ptr = b->get_dataptr();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                A_ptr[i*n+j] += b_ptr[j];
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
    void GEMM (const char tA, const char tB,
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
    void MAP (const SP_NArray<T> Y,
              const function<T(T)>& f,
              const SP_NArray<T> X,
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
    void MAP (const SP_NArray<T> Y,
              const function<T(T, T)>& f,
              const SP_NArray<T> X, const SP_NArray<T> Z,
              const bool overwrite) {
        assert(Y->get_dims() == X->get_dims());
        assert(Y->get_dims() == Z->get_dims());
        auto Y_ptr = Y->get_dataptr();
        auto X_ptr = X->get_dataptr();
        auto Z_ptr = Z->get_dataptr();
        for (int i = 0; i < Y->get_size(); i++) {
            if (overwrite) {
                Y_ptr[i] = f(X_ptr[i], Z_ptr[i]);
            } else {
                Y_ptr[i] += f(X_ptr[i], Z_ptr[i]);
            }
        }
    }
    
    template<typename T>
    void MAP_TO (const SP_NArray<T> Y, const function<T(T)>& f, const SP_NArray<T> X) {
        MAP(Y, f, X, true);
    }
    
    template<typename T>
    void MAP_ON (const SP_NArray<T> Y, const function<T(T)>& f, const SP_NArray<T> X) {
        MAP(Y, f, X, false);
    }
    
    template<typename T>
    void MAP_TO (const SP_NArray<T> Y,
                 const function<T(T, T)>& f,
                 const SP_NArray<T> X, const SP_NArray<T> Z) {
        MAP(Y, f, X, Z, true);
    }

    template<typename T>
    void MAP_ON (const SP_NArray<T> Y,
                 const function<T(T, T)>& f,
                 const SP_NArray<T> X, const SP_NArray<T> Z) {
        MAP(Y, f, X, Z, false);
    }
    
    // currently, only two dimensional array are supported
    // X[m][n] -> Y[m]
    template<typename T>
    void PROJ_MAP (const SP_NArray<T> Y,
                   const function<T(T)>& f,
                   const SP_NArray<T> X,
                   const SP_NArray<T> idx,
                   const bool overwrite) {
        assert(X->get_dims().size() == 2);
        assert(Y->get_dims().size() == 1);
        int m = X->get_dims()[0];
        int n = X->get_dims()[1];
        assert(m == Y->get_dims()[0]);
        assert(idx->get_dims().size() == 1);
        assert(m == idx->get_dims()[0]);
        
        auto Y_ptr = Y->get_dataptr();
        auto X_ptr = X->get_dataptr();
        auto idx_ptr = idx->get_dataptr();
        for (int i = 0; i < m; i++) {
            int j = idx_ptr[i];
            assert(j < n);
            if (overwrite) {
                Y_ptr[i] = f(X_ptr[i*n + j]);
            } else {
                Y_ptr[i] += f(X_ptr[i*n + j]);
            }
        }
    }
    
    // currently, only two dimensional array are supported
    // X[m][n] -> Y[m]
    template<typename T>
    void PROJ_MAP_TO (const SP_NArray<T> Y,
                      const function<T(T)>& f,
                      const SP_NArray<T> X,
                      const SP_NArray<T> idx) {
        PROJ_MAP(Y, f, X, idx, true);
    }
    
    // currently, only two dimensional array are supported
    // X[m][n] -> Y[m]
    template<typename T>
    void PROJ_MAP_ON (const SP_NArray<T> Y,
                      const function<T(T)>& f,
                      const SP_NArray<T> X,
                      const SP_NArray<T> idx) {
        PROJ_MAP(Y, f, X, idx, false);
    }
    
    // currently, only two dimensional array are supported
    // to be fixed
    template<typename T>
    void SUB_MAP (const SP_NArray<T> Y,
                  const function<T(T)>& f,
                  const SP_NArray<T> X,
                  const SP_NArray<T> a, const SP_NArray<T> b,
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
            int a_idx = int(a_ptr[i]);
            int b_idx = int(b_ptr[i]);
            if (overwrite) {
                Y_ptr[a_idx*stride + b_idx] = f(X_ptr[i]);
            } else {
                Y_ptr[a_idx*stride + b_idx] += f(X_ptr[i]);
            }
        }
    }
    
    // currently, only two dimensional array are supported
    template<typename T>
    void SUB_MAP_TO (const SP_NArray<T> Y,
                     const function<T(T)>& f,
                     const SP_NArray<T> X,
                     const SP_NArray<T> a, const SP_NArray<T> b) {
        SUB_MAP(Y, f, X, a, b, true);
    }
    
    // currently, only two dimensional array are supported
    template<typename T>
    void SUB_MAP_ON (const SP_NArray<T> Y,
                     const function<T(T)>& f,
                     const SP_NArray<T> X,
                     const SP_NArray<T> a, const SP_NArray<T> b) {
        SUB_MAP(Y, f, X, a, b, false);
    }
    
    
    template class NArray<int>;
    template class NArray<float>;
    template class NArray<double>;
    template
    void SUM_POSITIVE_VALUE (const SP_NArray<float> A, float *res);
    template
    void SUM_POSITIVE_VALUE (const SP_NArray<double> A, double *res);
    template
    void ADD_TO_ROW (const SP_NArray<float> A, const SP_NArray<float> a);
    template
    void ADD_TO_ROW (const SP_NArray<double> A, const SP_NArray<double> a);
    template
    void GEMM(const char tA, const char tB,
              const float alpha, const SP_NArray<float> A, const SP_NArray<float> B,
              const float beta, const SP_NArray<float> C);
    template
    void GEMM(const char tA, const char tB,
              const double alpha, const SP_NArray<double> A, const SP_NArray<double> B,
              const double beta, const SP_NArray<double> C);
    //    template void MAP (const function<float(float)>& f,
    //                       const SP_NArray<float> A, const SP_NArray<float> B,
    //                       const bool overwrite);
    //    template void MAP (const function<double(double)>& f,
    //                       const SP_NArray<double> A, const SP_NArray<double> B,
    //                       const bool overwrite);
    template
    void MAP_TO (const SP_NArray<float> Y,
                 const function<float(float)>& f,
                 const SP_NArray<float> X);
    template
    void MAP_TO (const SP_NArray<double> Y,
                 const function<double(double)>& f,
                 const SP_NArray<double> X);
    template
    void MAP_ON (const SP_NArray<float> Y,
                 const function<float(float)>& f,
                 const SP_NArray<float> X);
    template
    void MAP_ON (const SP_NArray<double> Y,
                 const function<double(double)>& f,
                 const SP_NArray<double> X);
    template
    void MAP_TO (const SP_NArray<float> Y,
                 const function<float(float, float)>& f,
                 const SP_NArray<float> X, const SP_NArray<float> Z);
    template
    void MAP_TO (const SP_NArray<double> Y,
                 const function<double(double, double)>& f,
                 const SP_NArray<double> X, const SP_NArray<double> Z);
    template
    void MAP_ON (const SP_NArray<float> Y,
                 const function<float(float, float)>& f,
                 const SP_NArray<float> X, const SP_NArray<float> Z);
    template
    void MAP_ON (const SP_NArray<double> Y,
                 const function<double(double, double)>& f,
                 const SP_NArray<double> X, const SP_NArray<double> Z);
    template
    void PROJ_MAP_TO (const SP_NArray<float> Y,
                      const function<float(float)>& f,
                      const SP_NArray<float> X,
                      const SP_NArray<float> idx);
    template
    void PROJ_MAP_TO (const SP_NArray<double> Y,
                      const function<double(double)>& f,
                      const SP_NArray<double> X,
                      const SP_NArray<double> idx);
    template
    void PROJ_MAP_ON (const SP_NArray<float> Y,
                      const function<float(float)>& f,
                      const SP_NArray<float> X,
                      const SP_NArray<float> idx);
    template
    void PROJ_MAP_ON (const SP_NArray<double> Y,
                      const function<double(double)>& f,
                      const SP_NArray<double> X,
                      const SP_NArray<double> idx);
    template
    void SUB_MAP_TO (const SP_NArray<float> Y,
                     const function<float(float)>& f,
                     const SP_NArray<float> X,
                     const SP_NArray<float> a, const SP_NArray<float> b);
    template
    void SUB_MAP_TO (const SP_NArray<double> Y,
                     const function<double(double)>& f,
                     const SP_NArray<double> X,
                     const SP_NArray<double> a, const SP_NArray<double> b);
    template
    void SUB_MAP_ON (const SP_NArray<float> Y,
                     const function<float(float)>& f,
                     const SP_NArray<float> X,
                     const SP_NArray<float> a, const SP_NArray<float> b);
    template
    void SUB_MAP_ON (const SP_NArray<double> Y,
                     const function<double(double)>& f,
                     const SP_NArray<double> X,
                     const SP_NArray<double> a, const SP_NArray<double> b);
    
}
