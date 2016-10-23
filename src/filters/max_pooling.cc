#include "galois/filters/max_pooling.h"

namespace gs {

    template<typename T>
    MaxPooling<T>::MaxPooling(
        int kernel_rows,
        int kernel_columns,
        int stride_rows,
        int stride_columns):
        kernel_rows(kernel_rows),
        kernel_columns(kernel_columns),
        stride_rows(stride_rows),
        stride_columns(stride_columns) {
        CHECK(kernel_rows > 0 && kernel_columns > 0 && stride_rows > 0 && stride_columns > 0,
              "all parameters should be positive");
    }

    template<typename T>
    SP_Filter<T> MaxPooling<T>::share() {
        CHECK(in_signal == nullptr, "in signal should not be set");
        CHECK(out_signal == nullptr, "out signal should not be set");
        return make_shared<MaxPooling<T>>(
            this->kernel_rows,
            this->kernel_columns,
            this->stride_rows,
            this->stride_columns
        );
    }

    template<typename T>
    void MaxPooling<T>::install_signals(const vector<SP_Signal<T>> &in_signals, const vector<SP_Signal<T>> &out_signals) {
        CHECK(in_signal == nullptr, "in signal should not be initialized");
        CHECK(out_signal == nullptr, "out signal should not be initialized");
        CHECK(in_signals.size() == 1, "need only 1 in signal");
        CHECK(out_signals.size() == 1, "need only 1 out signal");

        in_signal = in_signals[0];
        out_signal = out_signals[0];
    }

    template<typename T>
    void MaxPooling<T>::set_dims(int batch_size) {
        CHECK(!in_signal->empty(), "in signal should be initialized");
        auto in_dims = in_signal->get_data_dims();
        CHECK(in_dims.size() == 4 && in_dims[0] == batch_size, "should has 4 dimensions");
        this->num_rows = in_dims[1];
        this->num_columns = in_dims[2];
        this->channels = in_dims[3];
        CHECK(num_rows >= kernel_rows && ((num_rows - kernel_rows) % stride_rows == 0) &&
              num_columns >= kernel_columns && ((num_columns - kernel_columns) % stride_columns == 0),
              "these conditions should be satisfied");
        int out_rows = (num_rows - kernel_rows) / stride_rows + 1;
        int out_columns = (num_columns - kernel_columns) / stride_columns + 1;
        vector<int> expected_out_dims { batch_size, out_rows, out_columns, channels };
        if (out_signal->empty()) {
            out_signal->set_data_dims(expected_out_dims);
        } else {
            CHECK(expected_out_dims == out_signal->get_data_dims(), "wrong dimensions for out signal");
        }
        this->max_indexes = make_shared<NArray<T>>(
            vector<int>{ batch_size, out_rows, out_columns, channels, 2 }
        );
    }

    template<typename T>
    void MaxPooling<T>::forward() {
        auto in_data = in_signal->get_data();
        CHECK(!in_data->opaque(), "in_data should not be opaque");
        auto out_data = out_signal->get_data();

        auto in_data_ptr = in_data->get_data();
        auto in_size = in_data->get_dims();
        int in_s1 = in_size[3];
        int in_s2 = in_s1 * in_size[2];
        int in_s3 = in_s2 * in_size[1];
        auto out_data_ptr = out_data->get_data();
        auto out_size = out_data->get_dims();
        int out_s1 = out_size[3];
        int out_s2 = out_s1 * out_size[2];
        int out_s3 = out_s2 * out_size[1];
        int batch_size = in_size[0];
        auto max_indexes_ptr = max_indexes->get_data();
        bool overwrite = out_data->opaque();
        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < out_size[1]; i++) {
                for (int j = 0; j < out_size[2]; j++) {
                    for (int c = 0; c < channels; c++) {
                        T max = in_data_ptr[batch*in_s3 + i*stride_rows*in_s2 + j*stride_columns*in_s1 + c];
                        int max_shift_m = 0;
                        int max_shift_n = 0;
                        for (int m = 0; m < kernel_rows; m++) {
                            for (int n = 0; n < kernel_columns; n++) {
                                auto v = in_data_ptr[batch*in_s3 + (i*stride_rows+m)*in_s2 + (j*stride_columns+n)*in_s1 + c];
                                if (max < v) {
                                    max = v;
                                    max_shift_m = m;
                                    max_shift_n = n;
                                }
                            }
                        }
                        int offset = batch*out_s3 + i*out_s2 + j*out_s1 + c;
                        if (overwrite) {
                            out_data_ptr[offset] = max;
                        } else {
                            out_data_ptr[offset] += max;
                        }
                        max_indexes_ptr[offset*2 + 0] = i*stride_rows + max_shift_m;
                        max_indexes_ptr[offset*2 + 1] = j*stride_columns + max_shift_n;
                    }
                }
            }
        }
        out_data->setclear();
    }

    template<typename T>
    void MaxPooling<T>::backward() {
        auto in_grad = in_signal->get_grad();
        auto out_grad = out_signal->get_grad();
        CHECK(!out_grad->opaque(), "out_grad should not be opaque")

        auto in_grad_ptr = in_grad->get_data();
        auto out_grad_ptr = out_grad->get_data();
        if (in_grad->opaque()) {
            for (int i = 0; i < in_grad->get_size(); i++) {
                in_grad_ptr[i] = 0;
            }
        }
        in_grad->setclear();
        int in_s1 = channels;
        int in_s2 = in_s1 * num_columns;
        int in_s3 = in_s2 * num_rows;
        auto out_size = out_grad->get_dims();
        int batch_size = out_size[0];
        int out_rows = out_size[1];
        int out_columns = out_size[2];
        int out_s1 = channels;
        int out_s2 = out_s1 * out_columns;
        int out_s3 = out_s2 * out_rows;
        auto max_indexes_ptr = max_indexes->get_data();
        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < out_rows; i++) {
                for (int j = 0; j < out_columns; j++) {
                    for (int c = 0; c < channels; c++) {
                        int offset = batch*out_s3 + i*out_s2 + j*out_s1 + c;
                        int max_index_s = max_indexes_ptr[offset*2 + 0];
                        int max_index_t = max_indexes_ptr[offset*2 + 1];
                        in_grad_ptr[batch*in_s3 + max_index_s*in_s2 + max_index_t*in_s1 + c] += out_grad_ptr[offset];
                    }
                }
            }
        }
    }

    template class MaxPooling<float>;
    template class MaxPooling<double>;

}