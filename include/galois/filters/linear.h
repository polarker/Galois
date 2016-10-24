#ifndef _GALOIS_LINEAR_H_
#define _GALOIS_LINEAR_H_

#include "galois/base.h"

namespace gs {

    template<typename T>
    class Linear : public PFilter<T> {
    private:
        SP_Signal<T> in_signal = nullptr;
        SP_Signal<T> out_signal = nullptr;
        int in_size = 0;
        int out_size = 0;

        SP_NArray<T> w = nullptr;
        SP_NArray<T> b = nullptr;
        SP_NArray<T> dw = nullptr;
        SP_NArray<T> db = nullptr;

    public:
        Linear(const bool just_for_share) {}
        Linear(const Linear&) = delete;
        Linear& operator=(const Linear&) = delete;
        Linear(int in_size, int out_size);

        SP_Filter<T> share() override;

        void install_signals(const vector<SP_Signal<T>> &in_signals, const vector<SP_Signal<T>> &out_signals) override;
        void set_dims(int batch_size) override;
        void reopaque() override;

        vector<SP_NArray<T>> get_params() override;
        vector<SP_NArray<T>> get_grads() override;

        void forward() override;
        void backward() override;
    };

}

#endif
