#ifndef _GALOIS_CROSSENTROPY_H_
#define _GALOIS_CROSSENTROPY_H_

#include "galois/base.h"

namespace gs {
    
    template<typename T>
    class CrossEntropy : public BFilter<T> {
    private:
        SP_Signal<T> in_signal = nullptr;
        SP_Signal<T> out_signal = nullptr;
    public:
        CrossEntropy() {}
        CrossEntropy(const CrossEntropy&) = delete;
        CrossEntropy& operator=(const CrossEntropy&) = delete;
        
        void install_signals(const vector<SP_Signal<T>> &in_signals, const vector<SP_Signal<T>> &out_signals) override;
        void set_dims(int batch_size) override;
        
        void forward(const vector<SP_Signal<T>> &in_signals, const vector<SP_Signal<T>> &out_signals) override;
        void backward(const vector<SP_Signal<T>> &in_signals, const vector<SP_Signal<T>> &out_signals) override;
    };
    
}

#endif
