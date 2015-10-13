#include "galois/base.h"

namespace gs {
    
    template<typename T>
    class Tanh : public BFilter<T> {
    public:
        void set_dims(SP_Signal<T> in_signal, SP_Signal<T> out_signal, int batch_size);
        void set_dims(vector<SP_Signal<T>> in_signals,
                      vector<SP_Signal<T>> out_signals,
                      int batch_size) override;
        void Forward(SP_Signal<T> inputs, SP_Signal<T> outputs) override {
            inputs->opaque = true;
            outputs->opaque = true;
            cout << "forward" << endl;
        }
            
        void Backward(SP_Signal<T> inputs, SP_Signal<T> outputs) override {
            inputs->opaque = true;
            outputs->opaque = true;
            cout << "backward" << endl;
        }
    };
    
}