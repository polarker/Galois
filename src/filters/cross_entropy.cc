#include "cross_entropy.h"
#include <cmath>

namespace gs {
    
    template<typename T>
    void CrossEntropy<T>::set_dims(SP_Signal<T> in_signal, SP_Signal<T> out_signal, int batch_size) {
        assert(!in_signal->empty());
        auto in_dims = in_signal->get_data_dims();
        
        assert(out_signal->empty());
        out_signal->set_data_dims(in_dims);
        out_signal->set_target_dims(batch_size);
        out_signal->set_extra_dims(batch_size);
    }

    template<typename T>
    void CrossEntropy<T>::set_dims(const vector<SP_Signal<T>> &in_signals,
                      const vector<SP_Signal<T>> &out_signals,
                      int batch_size) {
        assert(in_signals.size() == 1);
        assert(out_signals.size() == 1);
        
        set_dims(in_signals[0], out_signals[0], batch_size);
    }

    template<typename T>
    void CrossEntropy<T>::forward(const vector<SP_Signal<T>> &in_signals, const vector<SP_Signal<T>> &out_signals) {
        assert(in_signals.size() == 1);
        assert(out_signals.size() == 1);
        auto in_signal = in_signals[0];
        auto out_signal = out_signals[0];
        assert(out_signal->get_type() == OutputSignal);
        
        auto in_data = in_signal->get_data();
        auto out_data = out_signal->get_data();
        assert(out_data->opaque());
        
        // softmax function
        MAP_TO<T>(out_data, [](T x){return exp(x);}, in_data);
        out_data->normalize_for(NARRAY_DIM_ZERO);
        
        auto target = out_signal->get_target();
        auto loss = out_signal->get_extra();
        PROJ_MAP_TO<T>(loss, [](T x){return -log(x);}, out_data, target);
        SUM_POSITIVE_VALUE<T>(loss, out_signal->get_loss().get());
        
        out_data->set_opaque(false);
    }
    
    template<typename T>
    void CrossEntropy<T>::backward(const vector<SP_Signal<T>> &in_signals, const vector<SP_Signal<T>> &out_signals) {
        assert(in_signals.size() == 1);
        assert(out_signals.size() == 1);
        auto in_signal = in_signals[0];
        auto out_signal = out_signals[0];
        assert(out_signal->get_type() == OutputSignal);
        
        auto in_grad = in_signal->get_grad();
        auto out_data = out_signal->get_data();
        int batch_size = in_signal->get_data_dims()[0];
        if (in_grad->opaque()) {
            MAP_TO<T>(in_grad, [batch_size](T y){return y/T(batch_size);}, out_data);
            // todo
            in_grad->set_opaque(false);
        } else {
            MAP_ON<T>(in_grad, [batch_size](T y){return y/T(batch_size);}, out_data);
            // todo
        }
    }

    template class CrossEntropy<float>;
    template class CrossEntropy<double>;

}
