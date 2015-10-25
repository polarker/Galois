namespace gs
{

    template<typename T>
    void SUM_POSITIVE_VALUE (const SP_NArray<T> A, T *res) {
        T sum = 0;
        auto A_ptr = A->get_data();
        auto A_size = A->get_size();
        for (int i = 0; i < A_size; i++) {
            sum += A_ptr[i];
        }
        *res = sum;
    }
    
    template<typename T>
    void ADD_TO_ROW (const SP_NArray<T> Y, const SP_NArray<T> b) {
        auto Y_dims = Y->get_dims();
        auto b_dims = b->get_dims();
        assert(Y_dims.size() == 2);
        assert(b_dims.size() == 1);
        assert(Y_dims[1] == b_dims[0]);
        
        auto m = Y_dims[0];
        auto n = Y_dims[1];
        auto Y_ptr = Y->get_data();
        auto b_ptr = b->get_data();
        if (Y->opaque()) {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    Y_ptr[i*n+j] = b_ptr[j];
                }
            }
            Y->setclear();
        } else {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    Y_ptr[i*n+j] += b_ptr[j];
                }
            }
        }
    }
    
    template<typename T>
    void SUM_TO_ROW (const SP_NArray<T> b, const SP_NArray<T> X) {
        auto b_dims = b->get_dims();
        auto X_dims = X->get_dims();
        assert(b_dims.size() == 1);
        assert(X_dims.size() == 2);
        assert(b_dims[0] == X_dims[1]);

        auto m = X_dims[0];
        auto n = X_dims[1];
        auto X_ptr = X->get_data();
        auto b_ptr = b->get_data();
        if (b->opaque()) {
            for (int j = 0; j < n; j++) {
                b_ptr[j] = X_ptr[j];
            }
            b->setclear();
        } else {
            for (int j = 0; j < n; j++) {
                b_ptr[j] += X_ptr[j];
            }
        }
        for (int i = 1; i < m; i++) {
            for (int j = 0; j < n; j++) {
                b_ptr[j] += X_ptr[i*n + j];
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
        auto N   = tB=='T' ? B_dims[0] : B_dims[1];
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
              A->get_data(), lda,
              B->get_data(), ldb,
              beta,
              C->get_data(), ldc);
        C->setclear();
    }
    
    template<typename L>
    void GEMM (const SP_NArray<L> Y,
               const char tA, const char tB,
               const SP_NArray<L> A, const SP_NArray<L> B) {
        if (Y->opaque()) {
            GEMM(tA, tB, static_cast<L>(1.0), A, B, static_cast<L>(0.0), Y);
            Y->setclear();
        } else {
            GEMM(tA, tB, static_cast<L>(1.0), A, B, static_cast<L>(1.0), Y);
        }
    }
    
    template<typename T>
    void _MAP (const SP_NArray<T> Y,
               const function<T(T)>& f,
               const SP_NArray<T> X,
               const bool overwrite) {
        assert(Y->get_dims() == X->get_dims());
        auto Y_ptr = Y->get_data();
        auto X_ptr = X->get_data();
        if (overwrite) {
            for (int i = 0; i < Y->get_size(); i++) {
                Y_ptr[i] = f(X_ptr[i]);
            }
        } else {
            for (int i = 0; i < Y->get_size(); i++) {
                Y_ptr[i] += f(X_ptr[i]);
            }
        }
    }
    
    template<typename T>
    void _MAP (const SP_NArray<T> Y,
               const function<T(T, T)>& f,
               const SP_NArray<T> X, const SP_NArray<T> Z,
               const bool overwrite) {
        assert(Y->get_dims() == X->get_dims());
        assert(Y->get_dims() == Z->get_dims());
        auto Y_ptr = Y->get_data();
        auto X_ptr = X->get_data();
        auto Z_ptr = Z->get_data();
        if (overwrite) {
            for (int i = 0; i < Y->get_size(); i++) {
                Y_ptr[i] = f(X_ptr[i], Z_ptr[i]);
            }
        } else {
            for (int i = 0; i < Y->get_size(); i++) {
                Y_ptr[i] += f(X_ptr[i], Z_ptr[i]);
            }
        }
    }
    
    template<typename T>
    void MAP (const SP_NArray<T> Y, const function<T(T)>& f, const SP_NArray<T> X) {
        if (Y->opaque()) {
            _MAP(Y, f, X, true);
            Y->setclear();
        } else {
            _MAP(Y, f, X, false);
        }
    }
    
    template<typename T>
    void MAP (const SP_NArray<T> Y,
              const function<T(T, T)>& f,
              const SP_NArray<T> X, const SP_NArray<T> Z) {
        if (Y->opaque()) {
            _MAP(Y, f, X, Z, true);
            Y->setclear();
        } else {
            _MAP(Y, f, X, Z, false);
        }
    }
    
    // currently, only two dimensional array are supported
    // X[m][n] -> Y[m]
    template<typename T>
    void _PROJ_MAP (const SP_NArray<T> Y,
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
        
        auto Y_ptr = Y->get_data();
        auto X_ptr = X->get_data();
        auto idx_ptr = idx->get_data();
        if (overwrite) {
            for (int i = 0; i < m; i++) {
                int j = idx_ptr[i];
                assert(j < n);
                Y_ptr[i] = f(X_ptr[i*n + j]);
            }
        } else {
            for (int i = 0; i < m; i++) {
                int j = idx_ptr[i];
                assert(j < n);
                Y_ptr[i] += f(X_ptr[i*n + j]);
            }
        }
    }
    
    // currently, only two dimensional array are supported
    // X[m][n] -> Y[m]
    template<typename T>
    void PROJ_MAP (const SP_NArray<T> Y,
                   const function<T(T)>& f,
                   const SP_NArray<T> X,
                   const SP_NArray<T> idx) {
        if (Y->opaque()) {
            _PROJ_MAP(Y, f, X, idx, true);
            Y->setclear();
        } else {
            _PROJ_MAP(Y, f, X, idx, false);
        }
    }
    
    // currently, only two dimensional array are supported
    // to be fixed
    template<typename T>
    void _SUB_MAP (const SP_NArray<T> Y,
                   const function<T(T)>& f,
                   const SP_NArray<T> X,
                   const SP_NArray<T> a, const SP_NArray<T> b,
                   const bool overwrite) {
        assert(Y->get_dims() == X->get_dims());
        assert(Y->get_dims().size() == 2);
        auto m = Y->get_dims()[0];
        auto n = Y->get_dims()[1];
        auto Y_ptr = Y->get_data();
        auto X_ptr = X->get_data();
        
        if (a == nullptr) {
            if (b == nullptr) {
                assert(m == n);
                if (overwrite) {
                    for (int i = 0; i < m; i++) {
                        Y_ptr[i*n + i] = f(X_ptr[i*n + i]);
                    }
                } else {
                    for (int i = 0; i < m; i++) {
                        Y_ptr[i*n + i] += f(X_ptr[i*n + i]);
                    }
                }
            } else {
                assert(b->get_dims().size() == 1);
                assert(b->get_size() == m);
                auto b_ptr = b->get_data();
                if (overwrite) {
                    for (int i = 0; i < m; i++) {
                        auto j = static_cast<int>(b_ptr[i]);
                        Y_ptr[i*n + j] = f(X_ptr[i*n + j]);
                    }
                } else {
                    for (int i = 0; i < m; i++) {
                        auto j = static_cast<int>(b_ptr[i]);
                        Y_ptr[i*n + j] += f(X_ptr[i*n + j]);
                    }
                }
            }
        } else {
            assert(a->get_dims().size() == 1);
            auto a_ptr = a->get_data();
            if (b == nullptr) {
                assert(a->get_size() == n);
                if (overwrite) {
                    for (int j = 0; j < n; j++) {
                        auto i = static_cast<int>(a_ptr[j]);
                        Y_ptr[i*n + j] = f(X_ptr[i*n + j]);
                    }
                } else {
                    for (int j = 0; j < n; j++) {
                        auto i = static_cast<int>(a_ptr[j]);
                        Y_ptr[i*n + j] += f(X_ptr[i*n + j]);
                    }
                }
            } else {
                assert(b->get_dims().size() == 1);
                assert(a->get_size() == b->get_size());
                auto size = a->get_size();
                auto b_ptr = b->get_data();
                if (overwrite) {
                    for (int k = 0; k < size; k++) {
                        auto i = static_cast<int>(a_ptr[k]);
                        auto j = static_cast<int>(b_ptr[k]);
                        Y_ptr[i*n + j] = f(X_ptr[i*n + j]);
                    }
                } else {
                    for (int k = 0; k < size; k++) {
                        auto i = static_cast<int>(a_ptr[k]);
                        auto j = static_cast<int>(b_ptr[k]);
                        Y_ptr[i*n + j] += f(X_ptr[i*n + j]);
                    }
                }
            }
        }
    }
    
    // currently, only two dimensional array are supported
    template<typename T>
    void SUB_MAP (const SP_NArray<T> Y,
                  const function<T(T)>& f,
                  const SP_NArray<T> X,
                  const SP_NArray<T> a, const SP_NArray<T> b) {
        if (Y->opaque()) {
            _SUB_MAP(Y, f, X, a, b, true);
            Y->setclear();
        } else {
            _SUB_MAP(Y, f, X, a, b, false);
        }
    }

}
