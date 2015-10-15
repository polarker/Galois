#include "cross_entropy.h"

namespace gs {
    
    template<typename T>
    void CrossEntropy<T>::set_dims(SP_Signal<T> in_signal, SP_Signal<T> out_signal, int batch_size) {
        auto in_dims = in_signal->dims;
        assert(!in_dims.empty());
        
        auto os = dynamic_pointer_cast<OutputSignal<T>>(out_signal);
        assert(os);
        assert(os->dims.empty()); // output should only be set once for dimensions
        
        os->set_dims(in_dims);
        os->set_target_dims(batch_size);
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
        ;
    }

    template class CrossEntropy<float>;
    template class CrossEntropy<double>;

}
