#include "cross_entropy.h"
#include <cmath>

namespace gs {
    
    template<typename T>
    void CrossEntropy<T>::set_dims(SP_Signal<T> in_signal, SP_Signal<T> out_signal, int batch_size) {
        assert(!in_signal->empty());
        auto in_dims = in_signal->get_dims();
        
        auto os = dynamic_pointer_cast<OutputSignal<T>>(out_signal);
        assert(os);
        assert(os->empty()); // output should only be set once for dimensions
        
        os->set_dims(in_dims);
        os->set_target_dims(batch_size);
        os->set_extra_dims(batch_size);
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
        assert(out_signal->opaque_data);
        auto in_data = in_signal->data;
        auto out_data = out_signal->data;
        
        // softmax function
        MAP_TO<T>(out_data, [](T x){return exp(x);}, in_data);
        cout << "cross entropy forward:\n" << out_data << '\n';
        out_data->normalize_for(NARRAY_DIM_ZERO);
        cout << "cross entropy forward:\n" << out_data << '\n';
        
        auto os = dynamic_pointer_cast<OutputSignal<T>>(out_signal);
        assert(os);
        auto target_data = os->target_data;
        auto loss_data = os->extra_data;
        PROJ_MAP_TO<T>(loss_data, [](T x){return -log(x);}, out_data, target_data);
        SUM_POSITIVE_VALUE<T>(loss_data, &os->loss);
        
        out_signal->opaque_data = false;
    }

    template class CrossEntropy<float>;
    template class CrossEntropy<double>;

}
