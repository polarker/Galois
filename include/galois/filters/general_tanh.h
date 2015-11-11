#ifndef _GALOIS_GENERALTANH_H_
#define _GALOIS_GENERALTANH_H_

#include "galois/base.h"

namespace gs {
    
    template<typename T>
    class GeneralTanh : public BFilter<T> {
    private:
        SP_Signal<T> in_signal = nullptr;
        SP_Signal<T> out_signal = nullptr;
    public:
        GeneralTanh() {}
        GeneralTanh(const GeneralTanh&) = delete;
        GeneralTanh& operator=(const GeneralTanh&) = delete;
        
        SP_Filter<T> share() override;
        
        void install_signals(const vector<SP_Signal<T>> &in_signals, const vector<SP_Signal<T>> &out_signals) override;
        void set_dims(int batch_size) override;

        void forward() override;
        void backward() override;
    };
    
}

#endif
