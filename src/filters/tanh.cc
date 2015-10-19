#include "tanh.h"
#include <cassert>
#include <cmath>

namespace gs {
    
    template<typename T>
    void Tanh<T>::set_dims(SP_Signal<T> in_signal, SP_Signal<T> out_signal, int batch_size) {
        assert(!in_signal->empty());
        auto in_dims = in_signal->get_data_dims();
        if (out_signal->empty()) {
            out_signal->set_data_dims(in_dims);
        } else {
            assert(in_dims == out_signal->get_data_dims());
        }
    }
    
    template<typename T>
    void Tanh<T>::set_dims(const vector<SP_Signal<T>> &in_signals,
                           const vector<SP_Signal<T>> &out_signals,
                           int batch_size) {
        assert(in_signals.size() == 1);
        assert(out_signals.size() == 1);
        
        set_dims(in_signals[0], out_signals[0], batch_size);
    }
    
    template<typename T>
    void Tanh<T>::forward(const vector<SP_Signal<T>> &in_signals, const vector<SP_Signal<T>> &out_signals) {
        assert(in_signals.size() == 1);
        assert(out_signals.size() == 1);
        auto in_data = in_signals[0]->get_data();
        auto out_data = out_signals[0]->get_data();
        
        MAP<T>(out_data, [](T x){ return tanh(x); }, in_data);
    }
    
    template<typename T>
    void Tanh<T>::backward(const vector<SP_Signal<T>> &in_signals, const vector<SP_Signal<T>> &out_signals) {
        assert(in_signals.size() == 1);
        assert(out_signals.size() == 1);
        auto in_signal = in_signals[0];
        auto out_signal = out_signals[0];
        
        auto in_grad = in_signal->get_grad();
        auto out_data = out_signal->get_data();
        auto out_grad = out_signal->get_grad();
        
        MAP<T>(in_grad, [](T dy, T y){return dy*(1-y*y);}, out_grad, out_data);
    }
    
    template class Tanh<float>;
    template class Tanh<double>;
    
}
