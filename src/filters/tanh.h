#ifndef _GALOIS_TANH_H_
#define _GALOIS_TANH_H_

#include "galois/base.h"

namespace gs {
    
    template<typename T>
    class Tanh : public BFilter<T> {
    public:
        Tanh() {}
        Tanh(const Tanh&) = delete;
        Tanh& operator=(const Tanh&) = delete;
        void set_dims(SP_Signal<T> in_signal, SP_Signal<T> out_signal, int batch_size);
        void set_dims(const vector<SP_Signal<T>> &in_signals,
                      const vector<SP_Signal<T>> &out_signals,
                      int batch_size) override;

        void forward(const vector<SP_Signal<T>> &in_signals, const vector<SP_Signal<T>> &out_signals) override;
        void backward(const vector<SP_Signal<T>> &in_signals, const vector<SP_Signal<T>> &out_signals) override;
    };
    
}

#endif
