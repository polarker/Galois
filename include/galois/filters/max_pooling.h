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
        size_t kernel_rows = 0;
        size_t kernel_columns = 0;
        size_t stride_rows = 0;
        size_t stride_columns = 0;

        // the following are set in set_dims
        size_t num_rows = 0;
        size_t num_columns = 0;
        size_t channels = 0;
        SP_NArray<T> max_indexes = nullptr;

    public:
        MaxPooling(const MaxPooling&) = delete;
        MaxPooling& operator=(const MaxPooling) = delete;
        MaxPooling(size_t kernel_rows, size_t kernel_columns, size_t stride_rows, size_t stride_columns);

        SP_Filter<T> share() override;

        void install_signals(const vector<SP_Signal<T>> &in_signals, const vector<SP_Signal<T>> &out_signals) override;
        void set_dims(size_t batch_size) override;
        void reopaque() override {}

        void forward() override;
        void backward() override;
    };

}

#endif