#include "galois/base.h"

namespace gs {
    
    class Linear : public PFilter {
    public:
        int in_size;
        int out_size;
        GArray *w;
        GArray *b;
        GArray *dw;
        GArray *db;
        
        Linear() = delete;
        Linear(int in_size, int out_size);
        void set_dims(shared_ptr<Signal> in_signal, shared_ptr<Signal> out_signal, int batch_size);
        void set_dims(const vector<shared_ptr<Signal>> in_signals,
                      const vector<shared_ptr<Signal>> out_signals,
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