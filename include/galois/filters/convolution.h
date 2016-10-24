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
        int num_rows = 0;
        int num_columns = 0;
        int in_channels = 0;
        int out_channels = 0;
        int kernel_rows = 0;
        int kernel_columns = 0;

        // int padding_w = 0;
        // int padding_h = 0;

        SP_NArray<T> w = nullptr;
        SP_NArray<T> b = nullptr;
        SP_NArray<T> dw = nullptr;
        SP_NArray<T> db = nullptr;

    public:
        Convolution(const bool for_clone_or_share) {}
        Convolution(const Convolution&) = delete;
        Convolution& operator=(const Convolution&) = delete;
        Convolution(int num_rows, int num_columns, int in_channels, int out_channels, int kernel_w, int kernel_h);

        SP_Filter<T> share() override;
        SP_Filter<T> clone() override;

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
