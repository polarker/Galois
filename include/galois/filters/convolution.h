#ifndef _GALOIS_CONVOLUTION_H_
#define _GALOIS_CONVOLUTION_H_

#include "galois/base.h"

namespace gs {

    template<typename T>
    class Convolution : public PFilter<T> {
        /* TODO: support striding, padding */
    private:
        SP_Signal<T> in_signal = nullptr;
        SP_Signal<T> out_signal = nullptr;
        size_t num_rows = 0;
        size_t num_columns = 0;
        size_t in_channels = 0;
        size_t out_channels = 0;
        size_t kernel_rows = 0;
        size_t kernel_columns = 0;

        // size_t padding_w = 0;
        // size_t padding_h = 0;

        SP_NArray<T> w = nullptr;
        SP_NArray<T> b = nullptr;
        SP_NArray<T> dw = nullptr;
        SP_NArray<T> db = nullptr;

    public:
        Convolution(const bool for_clone_or_share) {}
        Convolution(const Convolution&) = delete;
        Convolution& operator=(const Convolution&) = delete;
        Convolution(size_t num_rows, size_t num_columns, size_t in_channels, size_t out_channels, size_t kernel_w, size_t kernel_h);

        SP_Filter<T> share() override;
        SP_Filter<T> clone() override;

        void install_signals(const vector<SP_Signal<T>> &in_signals, const vector<SP_Signal<T>> &out_signals) override;
        void set_dims(size_t batch_size) override;
        void reopaque() override;

        vector<SP_NArray<T>> get_params() override;
        vector<SP_NArray<T>> get_grads() override;

        void forward() override;
        void backward() override;
    };
}

#endif
