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
    
    template<typename T>
    void NArray<T>::copy_data(const vector<int> &dims, T* data) {
        assert(dims == this->dims);
        for (int i = 0; i < this->size; i++) {
            this->data[i] = data[i];
        }
    }

    void GEMM(const char tA, const char tB,
              const float alpha, const SP_NArray<float> A, const SP_NArray<float> B,
              const float beta, const SP_NArray<float> C) {
        assert(A->get_dims().size() == 2);
        assert(B->get_dims().size() == 2);
        auto t_A = tA=='T' ? CblasTrans : CblasNoTrans;
        auto t_B = tB=='T' ? CblasTrans : CblasNoTrans;
        auto M = tA=='T' ? A->get_dims()[1] : A->get_dims()[0];
        auto K_A = tA=='T' ? A->get_dims()[0] : A->get_dims()[1];
        auto K_B = tB=='T' ? B->get_dims()[1] : B->get_dims()[0];
        auto N = tB=='T' ? B->get_dims()[0] : B->get_dims()[1];
        assert(K_A == K_B);
        auto K = K_A;
        cblas_sgemm(CblasRowMajor, t_A, t_B,
                    M, N, K,
                    alpha,
                    A->get_dataptr(), K,
                    B->get_dataptr(), N,
                    beta,
                    C->get_dataptr(), N);
    }

    void GEMM(const char tA, const char tB,
              const double alpha, const SP_NArray<double> A, const SP_NArray<double> B,
              const double beta, const SP_NArray<double> C) {
        assert(A->get_dims().size() == 2);
        assert(B->get_dims().size() == 2);
        auto t_A = tA=='T' ? CblasTrans : CblasNoTrans;
        auto t_B = tB=='T' ? CblasTrans : CblasNoTrans;
        auto M = tA=='T' ? A->get_dims()[1] : A->get_dims()[0];
        auto K_A = tA=='T' ? A->get_dims()[0] : A->get_dims()[1];
        auto K_B = tB=='T' ? B->get_dims()[1] : B->get_dims()[0];
        auto N = tB=='T' ? B->get_dims()[0] : B->get_dims()[1];
        assert(K_A == K_B);
        auto K = K_A;
        cblas_dgemm(CblasRowMajor, t_A, t_B,
                    M, N, K,
                    alpha,
                    A->get_dataptr(), K,
                    B->get_dataptr(), N,
                    beta,
                    C->get_dataptr(), N);
    }
    
    template class NArray<float>;
    template class NArray<double>;

}
