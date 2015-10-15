#ifndef _GALOIS_CROSSENTROPY_H_
#define _GALOIS_CROSSENTROPY_H_

#include "galois/base.h"

namespace gs {
    
    template<typename T>
    class CrossEntropy : public BFilter<T> {
    public:
        CrossEntropy() {}
        CrossEntropy(const CrossEntropy&) = delete;
        CrossEntropy& operator=(const CrossEntropy&) = delete;
        void set_dims(SP_Signal<T> in_signal, SP_Signal<T> out_signal, int batch_size);
        void set_dims(const vector<SP_Signal<T>> &in_signals,
                      const vector<SP_Signal<T>> &out_signals,
                      int batch_size) override;
        
        void forward(const vector<SP_Signal<T>> &in_signals, const vector<SP_Signal<T>> &out_signals) override;
        void backward(const vector<SP_Signal<T>> &in_signals, const vector<SP_Signal<T>> &out_signals) override {
            cout << "backward" << endl;
        }
    };
    
}

#endif
