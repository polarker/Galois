#ifndef _GALOIS_MAX_POOLING_H_
#define _GALOIS_MAX_POOLING_H_

#include "galois/base.h"

namespace gs {

    template<typename T>
    class MaxPooling : public BFilter<T> {
        /* TODO: support padding */
    private:
        SP_Signal<T> in_signal = nullptr;
        SP_Signal<T> out_signal = nullptr;
        int kernel_rows = 0;
        int kernel_columns = 0;
        int stride_rows = 0;
        int stride_columns = 0;

        // the following are set in set_dims
        int num_rows = 0;
        int num_columns = 0;
        int channels = 0;
        SP_NArray<T> max_indexes = nullptr;

    public:
        MaxPooling(const MaxPooling&) = delete;
        MaxPooling& operator=(const MaxPooling) = delete;
        MaxPooling(int kernel_rows, int kernel_columns, int stride_rows, int stride_columns);

        SP_Filter<T> share() override;

        void install_signals(const vector<SP_Signal<T>> &in_signals, const vector<SP_Signal<T>> &out_signals) override;
        void set_dims(int batch_size) override;
        void reopaque() override {}

        void forward() override;
        void backward() override;
    };

}

#endif