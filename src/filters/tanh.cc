#include "tanh.h"
#include <cassert>

namespace gs {
    void Tanh::set_dims(shared_ptr<Signal> in_signal, shared_ptr<Signal> out_signal, int batch_size) {
        assert(!in_signal->dims.empty());
        if (out_signal->dims.empty()) {
            out_signal->dims.insert(out_signal->dims.end(), in_signal->dims.begin(), in_signal->dims.end());
            // for allocation
        } else {
            assert(in_signal->dims == out_signal->dims);
        }
    }
    
    void Tanh::set_dims(vector<shared_ptr<Signal>> in_signals, vector<shared_ptr<Signal>> out_signals, int batch_size) {
        assert(in_signals.size() == 1);
        assert(out_signals.size() == 1);
        
        set_dims(in_signals[0], out_signals[0], batch_size);
    }
}
