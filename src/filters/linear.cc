#include "linear.h"
#include <cassert>

using namespace std;

namespace gs {
    Linear::Linear(int in_size, int out_size) {
        this->in_size = in_size;
        this->out_size = out_size;
    }
    
    void Linear::set_dims(shared_ptr<Signal> in_signal, shared_ptr<Signal> out_signal, int batch_size) {
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
    
    void Linear::set_dims(vector<shared_ptr<Signal>> in_signals, vector<shared_ptr<Signal>> out_signals, int batch_size) {
        assert(in_signals.size() == 1);
        assert(out_signals.size() == 1);
        
        set_dims(in_signals[0], out_signals[0], batch_size);
    }
}
