#include "linear.h"
#include "galois/narray.h"
#include <cassert>

using namespace std;

namespace gs {
    
    template<typename T>
    Linear<T>::Linear(int in_size, int out_size) : in_size(in_size), out_size(out_size) {
        T s = sqrt(6. / (in_size + out_size));
        this->w  = make_shared<NArray<T>>(in_size, out_size);
        this->w->uniform(-s, s);
        this->b  = make_shared<NArray<T>>(out_size);
        this->b->uniform(-s, s);
        this->dw = make_shared<NArray<T>>(in_size, out_size);
        this->db = make_shared<NArray<T>>(out_size);
        this->opaque = true;
    }
    
    template<typename T>
    void Linear<T>::set_dims(SP_Signal<T> in_signal, SP_Signal<T> out_signal, int batch_size) {
        if (in_signal->empty()) {
            in_signal->set_dims(batch_size, in_size);
        } else {
            assert(in_signal->get_dims() ==  vector<int>({batch_size, in_size}));
        }
        if (out_signal->empty()) {
            out_signal->set_dims(batch_size, out_size);
        } else {
            assert(out_signal->get_dims() == vector<int>({batch_size, out_size}));
        }
    }
    
    template<typename T>
    void Linear<T>::set_dims(const vector<SP_Signal<T>> &in_signals, const vector<SP_Signal<T>> &out_signals, int batch_size) {
        assert(in_signals.size() == 1);
        assert(out_signals.size() == 1);
        
        set_dims(in_signals[0], out_signals[0], batch_size);
    }
    
    template<typename T>
    void Linear<T>::forward(const vector<SP_Signal<T>> &in_signals, const vector<SP_Signal<T>> &out_signals) {
        assert(in_signals.size() == 1);
        assert(out_signals.size() == 1);
        auto in_data = in_signals[0]->data;
        auto out_data = out_signals[0]->data;
        
        if (out_signals[0]->opaque) {
            GEMM<T>('N', 'N', 1.0, in_data, w, 0.0, out_data);
            ADD_TO_ROW<T>(out_data, b);
            out_signals[0]->opaque = false;
        } else {
            GEMM<T>('N', 'N', 1.0, in_data, w, 1.0, out_data);
            ADD_TO_ROW<T>(out_data, b);
        }
    }
    
    template class Linear<float>;
    template class Linear<double>;

}
