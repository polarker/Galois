# include "galois/narray.h"
# include <cassert>

namespace gs
{
    template<typename T>
    NArray<T>::NArray(int m) : dims{m}, size{m} {
        assert(m > 0);
        data = new T[get_size()];
    }
    
    template<typename T>
    NArray<T>::NArray(int m, int n) : dims{m, n}, size{m*n} {
        assert(m > 0);
        assert(n > 0);
        data = new T[get_size()];
    }
    
    template<typename T>
    NArray<T>::NArray(int m, int n, int o) : dims{m, n, o}, size{m*n*o} {
        assert(m > 0);
        assert(n > 0);
        assert(o > 0);
        data = new T[get_size()];
    }
    
    template<typename T>
    NArray<T>::NArray(int m, int n, int o, int k) : dims{m, n, o, k}, size{m*n*o*k} {
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
        size = 1;
        for (auto d : dims) {
            size *= d;
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
    void NArray<T>::copy_from(const vector<int> &dims, const T* data) {
        assert(dims == this->dims);
        for (int i = 0; i < this->get_size(); i++) {
            this->data[i] = data[i];
        }
        setclear();
    }

    template<typename T>
    void NArray<T>::copy_from(const vector<int> &dims, const SP_NArray<T> dataset) {
        assert(dims.size() == this->get_dims()[0]);
        assert(dataset->get_dims().size() == this->dims.size());
        for (int i = 1; i < this->dims.size(); i++) {
            assert(this->dims[i] == dataset->get_dims()[i]);
        }
        int batch_size = dims.size();
        int stride = this->get_size() / batch_size;
        for (int i = 0; i < batch_size; i++) {
            assert(dims[i] < dataset->get_dims()[0]);
            auto dataset_ptr = dataset->get_data();
            for (int j = 0; j < stride; j++) {
                this->data[i*stride + j] = dataset_ptr[dims[i]*stride + j];
            }
        }
        setclear();
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
    
    template class NArray<int>;
    template class NArray<float>;
    template class NArray<double>;
    
}
