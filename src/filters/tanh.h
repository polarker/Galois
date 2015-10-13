#include "galois/base.h"

namespace gs {
    
    class Tanh : public BFilter {
    public:
        void set_dims(shared_ptr<Signal> in_signal, shared_ptr<Signal> out_signal, int batch_size);
        void set_dims(vector<shared_ptr<Signal>> in_signals,
                      vector<shared_ptr<Signal>> out_signals,
                      int batch_size) override;
        void Forward(shared_ptr<Signal> inputs, shared_ptr<Signal> outputs) override {
            inputs->opaque = true;
            outputs->opaque = true;
            cout << "forward" << endl;
        }
            
        void Backward(shared_ptr<Signal> inputs, shared_ptr<Signal> outputs) override {
            inputs->opaque = true;
            outputs->opaque = true;
            cout << "backward" << endl;
        }
    };
    
}