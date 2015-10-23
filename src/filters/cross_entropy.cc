#include "cross_entropy.h"
#include <cmath>

namespace gs {
    
    template<typename T>
    void CrossEntropy<T>::install_signals(const vector<SP_Signal<T>> &in_signals, const vector<SP_Signal<T>> &out_signals) {
        assert(in_signal == nullptr);
        assert(out_signal == nullptr);
        assert(in_signals.size() == 1);
        assert(out_signals.size() == 1);
        
        in_signal = in_signals[0];
        out_signal = out_signals[0];
        assert(out_signal->get_type() == OutputSignal);
    }
    
    template<typename T>
    void CrossEntropy<T>::set_dims(int batch_size) {
        assert(!in_signal->empty());
        auto in_dims = in_signal->get_data_dims();
        
        assert(out_signal->get_type() == OutputSignal);
        assert(out_signal->empty());
        out_signal->set_data_dims(in_dims);
        out_signal->set_target_dims(batch_size);
        out_signal->set_extra_dims(batch_size);
        out_signal->initialize_loss();
    }

    template<typename T>
    void CrossEntropy<T>::forward() {
        auto in_data = in_signal->get_data();
        auto out_data = out_signal->get_data();
        assert(out_data->opaque());
        
        // softmax function
        MAP<T>(out_data, [](T x){return exp(x);}, in_data);
        out_data->normalize_for(NARRAY_DIM_ZERO);
        
        auto target = out_signal->get_target();
        auto losses = out_signal->get_extra();
        PROJ_MAP<T>(losses, [](T x){return -log(x);}, out_data, target);
        auto loss = out_signal->get_loss();
        SUM_POSITIVE_VALUE<T>(losses, loss.get());
        *loss /= target->get_size();
    }
    
    template<typename T>
    void CrossEntropy<T>::backward() {
        auto in_grad = in_signal->get_grad();
        auto out_data = out_signal->get_data();
        auto target = out_signal->get_target();
        int batch_size = in_signal->get_data_dims()[0];
        MAP<T>(in_grad, [batch_size](T y){return y/static_cast<T>(batch_size);}, out_data);
        SUB_MAP<T>(in_grad, [batch_size](T y){return -1/static_cast<T>(batch_size);}, in_grad, nullptr, target);
    }

    template class CrossEntropy<float>;
    template class CrossEntropy<double>;

}
