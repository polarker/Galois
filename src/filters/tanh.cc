#include "tanh.h"
#include <cassert>

namespace gs {
    
    template<typename T>
    void Tanh<T>::set_dims(SP_Signal<T> in_signal, SP_Signal<T> out_signal, int batch_size) {
        assert(!in_signal->dims.empty());
        if (out_signal->dims.empty()) {
//            out_signal->dims.insert(out_signal->dims.end(), in_signal->dims.begin(), in_signal->dims.end());
            out_signal->set_dims(in_signal->dims);
            // for allocation
        } else {
            assert(in_signal->dims == out_signal->dims);
        }
    }
    
    template<typename T>
    void Tanh<T>::set_dims(vector<SP_Signal<T>> in_signals, vector<SP_Signal<T>> out_signals, int batch_size) {
        assert(in_signals.size() == 1);
        assert(out_signals.size() == 1);
        
        set_dims(in_signals[0], out_signals[0], batch_size);
    }
    
    template class Tanh<float>;
    template class Tanh<double>;
    
}
