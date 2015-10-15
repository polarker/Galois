#include "linear.h"
#include "galois/narray.h"
#include <cassert>

using namespace std;

namespace gs {
    
    template<typename T>
    Linear<T>::Linear(int in_size, int out_size) : in_size(in_size), out_size(out_size) {
        w = make_shared<NArray<T>>(in_size, out_size);
//        b = make_shared<NArray<T>>(out_size);
        dw = make_shared<NArray<T>>(in_size, out_size);
//        db = make_shared<NArray<T>>(out_size);
        this->opaque = true;
    }
    
    template<typename T>
    void Linear<T>::set_dims(SP_Signal<T> in_signal, SP_Signal<T> out_signal, int batch_size) {
        if (in_signal->dims.empty()) {
//            in_signal->dims.insert(in_signal->dims.end(), {batch_size, in_size});
            in_signal->set_dims(batch_size, in_size);
        } else {
            assert(in_signal->dims ==  vector<int>({batch_size, in_size}));
            // for allocation
        }
        if (out_signal->dims.empty()) {
//            out_signal->dims.insert(out_signal->dims.end(), {batch_size, out_size});
            out_signal->set_dims(batch_size, out_size);
        } else {
            assert(out_signal->dims == vector<int>({batch_size, out_size}));
            // for allocation
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
            GEMM('N', 'N', 1.0, in_data, w, 0.0, out_data);
        } else {
            GEMM('N', 'N', 1.0, in_data, w, 1.0, out_data);
        }
    }
    
    template class Linear<float>;
    template class Linear<double>;

}
