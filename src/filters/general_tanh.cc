#include "galois/narray.h"
#include "galois/narray_functors.h"
#include "galois/filters/general_tanh.h"
#include <cmath>

namespace gs {
    
    template<typename T>
    SP_Filter<T> GeneralTanh<T>::share() {
        CHECK(in_signal == nullptr, "in signal should not be set");
        CHECK(out_signal == nullptr, "out signal should not be set");
        return make_shared<GeneralTanh<T>>();
    }
    
    template<typename T>
    void GeneralTanh<T>::install_signals(const vector<SP_Signal<T>> &in_signals, const vector<SP_Signal<T>> &out_signals) {
        CHECK(in_signal == nullptr, "in signal should not be initialized");
        CHECK(out_signal == nullptr, "out signal should not be initialized");
        CHECK(in_signals.size() == 1, "need only 1 in signal");
        CHECK(out_signals.size() == 1, "need only 1 out signal");
        
        in_signal = in_signals[0];
        out_signal = out_signals[0];
    }
    
    template<typename T>
    void GeneralTanh<T>::set_dims(int batch_size) {
        CHECK(!in_signal->empty(), "in signal should be initialized");
        auto in_dims = in_signal->get_data_dims();
        if (out_signal->empty()) {
            out_signal->set_data_dims(in_dims);
        } else {
            CHECK(in_dims == out_signal->get_data_dims(), "in signal and out signal should have the same dimensions");
        }
    }
        
    template<typename T>
    void GeneralTanh<T>::forward() {
        auto in_data = in_signal->get_data();
        auto out_data = out_signal->get_data();
        
        MAP(out_data, [](T x){ return tanh(x); }, in_data);
    }
    
    template<typename T>
    void GeneralTanh<T>::backward() {
        auto in_data = in_signal->get_data();
        auto in_grad = in_signal->get_grad();
        auto out_grad = out_signal->get_grad();
        
        MAP(in_grad, [](T dy, T x){return dy*(1-tanh(x)*tanh(x));}, out_grad, in_data);
    }
    
    template class GeneralTanh<float>;
    template class GeneralTanh<double>;
    
}
