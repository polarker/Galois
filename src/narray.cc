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
        cblas_sgemm(CblasRowMajor, t_A, t_B,
                    M, N, K,
                    alpha,
                    A->get_dataptr(), lda,
                    B->get_dataptr(), ldb,
                    beta,
                    C->get_dataptr(), ldc);
    }

    void GEMM(const char tA, const char tB,
              const double alpha, const SP_NArray<double> A, const SP_NArray<double> B,
              const double beta, const SP_NArray<double> C) {
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
        cblas_dgemm(CblasRowMajor, t_A, t_B,
                    M, N, K,
                    alpha,
                    A->get_dataptr(), lda,
                    B->get_dataptr(), ldb,
                    beta,
                    C->get_dataptr(), ldc);
    }
    
    template class NArray<float>;
    template class NArray<double>;

}
