#include "linear.h"
#include <cassert>

using namespace std;

namespace gs {
    
    template<typename T>
    Linear<T>::Linear(int in_size, int out_size) {
        this->in_size = in_size;
        this->out_size = out_size;
    }
    
    template<typename T>
    void Linear<T>::set_dims(SP_Signal<T> in_signal, SP_Signal<T> out_signal, int batch_size) {
        if (in_signal->dims.empty()) {
            in_signal->dims.insert(in_signal->dims.end(), {batch_size, in_size});
        } else {
            assert(in_signal->dims ==  vector<int>({batch_size, in_size}));
            // for allocation
        }
        if (out_signal->dims.empty()) {
            out_signal->dims.insert(out_signal->dims.end(), {batch_size, out_size});
        } else {
            assert(out_signal->dims == vector<int>({batch_size, out_size}));
            // for allocation
        }
    }
    
    template<typename T>
    void Linear<T>::set_dims(vector<SP_Signal<T>> in_signals, vector<SP_Signal<T>> out_signals, int batch_size) {
        assert(in_signals.size() == 1);
        assert(out_signals.size() == 1);
        
        set_dims(in_signals[0], out_signals[0], batch_size);
    }
    
    template class Linear<float>;
    template class Linear<double>;

}
