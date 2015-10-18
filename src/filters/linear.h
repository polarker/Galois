#ifndef _GALOIS_LINEAR_H_
#define _GALOIS_LINEAR_H_

#include "galois/base.h"

namespace gs {
    
    template<typename T>
    class Linear : public PFilter<T> {
    public:
        int in_size = 0;
        int out_size = 0;
        SP_NArray<T> w = nullptr;
        SP_NArray<T> b = nullptr;
        SP_NArray<T> dw = nullptr;
        SP_NArray<T> db = nullptr;
        
        Linear() = delete;
        Linear(const Linear&) = delete;
        Linear& operator=(const Linear&) = delete;
        Linear(int in_size, int out_size);
        void set_dims(SP_Signal<T> in_signal, SP_Signal<T> out_signal, int batch_size);
        void set_dims(const vector<SP_Signal<T>> &in_signals,
                      const vector<SP_Signal<T>> &out_signals,
                      int batch_size) override;
        
        void forward(const vector<SP_Signal<T>> &in_signals, const vector<SP_Signal<T>> &out_signals) override;
            
        void backward(const vector<SP_Signal<T>> &in_signals, const vector<SP_Signal<T>> &out_signals) override;
    };
    
}

#endif
