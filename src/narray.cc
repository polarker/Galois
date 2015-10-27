# include "galois/narray.h"
# include <cassert>

namespace gs
{
    template<typename T>
    NArray<T>::NArray(int m) : dims{m}, size{m} {
        CHECK(m > 0, "m should be positive");
        data = new T[get_size()];
    }
    
    template<typename T>
    NArray<T>::NArray(int m, int n) : dims{m, n}, size{m*n} {
        CHECK(m > 0, "m should be positive");
        CHECK(n > 0, "n should be positive");
        data = new T[get_size()];
    }
    
    template<typename T>
    NArray<T>::NArray(int m, int n, int o) : dims{m, n, o}, size{m*n*o} {
        CHECK(m > 0, "m should be positive");
        CHECK(n > 0, "n should be positive");
        CHECK(o > 0, "o should be positive");
        data = new T[get_size()];
    }
    
    template<typename T>
    NArray<T>::NArray(int m, int n, int o, int k) : dims{m, n, o, k}, size{m*n*o*k} {
        CHECK(m > 0, "m should be positive");
        CHECK(n > 0, "n should be positive");
        CHECK(o > 0, "o should be positive");
        CHECK(k > 0, "k should be positive");
        data = new T[get_size()];
    }
    
    template<typename T>
    NArray<T>::NArray(vector<int> nums) : dims{nums} {
        for (auto m : nums) {
            CHECK(m > 0, "each dimension should be positive");
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
        CHECK(dims == this->dims, "the dimension should be equal");
        for (int i = 0; i < this->get_size(); i++) {
            this->data[i] = data[i];
        }
        setclear();
    }

    template<typename T>
    void NArray<T>::copy_from(const vector<int> &idxs, const SP_NArray<T> dataset) {
        // copy a batch from dataset
        auto dataset_dims = dataset->get_dims();
        CHECK(idxs.size() == this->dims[0], "first dimension should be equal to batch size");
        CHECK(dataset_dims.size() == this->dims.size(), "number of dimensions should be equal");
        for (int i = 1; i < this->dims.size(); i++) {
            CHECK(dataset_dims[i] == this->dims[i], "dimensions should be equal");
        }

        int batch_size = idxs.size();
        int stride = this->get_size() / batch_size;
        auto dataset_ptr = dataset->get_data();
        for (int i = 0; i < batch_size; i++) {
            CHECK(idxs[i] < dataset_dims[0], "invalid index");
            for (int j = 0; j < stride; j++) {
                this->data[i*stride + j] = dataset_ptr[idxs[i]*stride + j];
            }
        }
        setclear();
    }

    
    template<typename T>
    void NArray<T>::normalize_for(int dim) {
        // currently, only two dimensional array are supported
        CHECK(dim == NARRAY_DIM_ZERO || dim == NARRAY_DIM_ONE, "only support upto 2 dimensional array");
        CHECK(this->dims.size() == 2, "only support upto 2 dimensional array");
        
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
