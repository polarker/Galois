#include "galois/narray.h"
#include "galois/filters/convolution.h"

using namespace std;

namespace gs {

    // TODO: optimization

    template<typename T>
    SP_Filter<T> Convolution<T>::share() {
        CHECK(in_signal == nullptr, "in signal should not be set");
        CHECK(out_signal == nullptr, "out signal should not be set");
        bool just_for_share = true;
        auto res = make_shared<Convolution<T>>(just_for_share);
        res->num_rows = this->num_rows;
        res->num_columns = this->num_columns;
        res->in_channels = this->in_channels;
        res->out_channels = this->out_channels;
        res->kernel_rows = this->kernel_rows;
        res->kernel_columns = this->kernel_columns;
        // res->padding_w = this->padding_w;
        // res->padding_w = this->padding_h;
        res->w = this->w;
        res->b = this->b;
        res->dw = this->dw;
        res->db = this->db;
        return res;
    }

    template<typename T>
    Convolution<T>::Convolution(
        int num_rows,
        int num_columns,
        int in_channels,
        int out_channels,
        int kernel_rows,
        int kernel_columns):
        num_rows(num_rows),
        num_columns(num_columns),
        in_channels(in_channels),
        out_channels(out_channels),
        kernel_rows(kernel_rows),
        kernel_columns(kernel_columns) {
        CHECK(
            num_rows > 0 &&
            num_columns > 0 &&
            in_channels > 0 &&
            out_channels > 0 &&
            kernel_rows > 0 && kernel_columns > 0,
            "all parameters should be positive");

        // CHECK((kernel_rows % 2) == 1 && (kernel_columns % 2) == 1, "size of kernel should be odd numbers");
        // this->padding_w =(kernel_rows - 1) / 2;
        // this->padding_h = ()
        T s = 1.0 / sqrt(num_rows * num_columns * in_channels);
        this->w = make_shared<NArray<T>>(kernel_rows, kernel_columns, in_channels, out_channels);
        this->w->uniform(-s, s);
        this->b = make_shared<NArray<T>>(out_channels);
        this->b->uniform(-s, s);
        this->dw = make_shared<NArray<T>>(kernel_rows, kernel_columns, in_channels, out_channels);
        this->db = make_shared<NArray<T>>(out_channels);
    }

    template<typename T>
    void Convolution<T>::install_signals(const vector<SP_Signal<T>> &in_signals, const vector<SP_Signal<T>> &out_signals) {
        CHECK(in_signal == nullptr, "in signal should not be initialized");
        CHECK(out_signal == nullptr, "out signal should not be initialized");
        CHECK(in_signals.size() == 1, "only need 1 in signal");
        CHECK(out_signals.size() == 1, "only need 1 out signal");

        in_signal = in_signals[0];
        out_signal = out_signals[0];
    }

    template<typename T>
    void Convolution<T>::set_dims(int batch_size) {
        auto expected_in_sizes = vector<int>{batch_size, num_rows, num_columns, in_channels};
        if (in_signal->empty()) {
            in_signal->set_data_dims(expected_in_sizes);
        } else {
            CHECK(in_signal->get_data_dims() == expected_in_sizes, "the dimensions of in signal are wrong");
        }
        auto expected_out_sizes = vector<int>{batch_size, num_rows - kernel_rows + 1, num_columns - kernel_columns + 1, out_channels};
        if (out_signal->empty()) {
            out_signal->set_data_dims(expected_out_sizes);
        } else {
            CHECK(out_signal->get_data_dims() == expected_out_sizes, "the dimensions of out signal are wrong");
        }
    }

    template<typename T>
    void Convolution<T>::reopaque() {
        this->dw->reopaque();
        this->db->reopaque();
    }

    template<typename T>
    vector<SP_NArray<T>> Convolution<T>::get_params() {
        return vector<SP_NArray<T>>{ this->w, this->b };
    }

    template<typename T>
    vector<SP_NArray<T>> Convolution<T>::get_grads() {
        return vector<SP_NArray<T>>{ this->dw, this->db };
    }

    /* optimize with GEMM */
    template<typename T>
    void Convolution<T>::forward() {
        auto in_data = in_signal->get_data();
        CHECK(!in_data->opaque(), "in_data should not be opaque");
        auto out_data = out_signal->get_data();

        // TODO: abstract the following as functionality of narray without performance overhead
        auto in_data_ptr = in_data->get_data(); // batch_size x num_rows x num_columns x in_channels
        auto in_size = in_data->get_dims();
        int in_s1 = in_size[3];
        int in_s2 = in_s1 * in_size[2];
        int in_s3 = in_s2 * in_size[1];
        auto out_data_ptr = out_data->get_data(); // batch_size x num_rows` x num_columns` x out_channels
        auto out_size = out_data->get_dims();
        int out_s1 = out_size[3];
        int out_s2 = out_s1 * out_size[2];
        int out_s3 = out_s2 * out_size[1];
        auto w_ptr = this->w->get_data(); // kernel_rows x kernel_columns x in_channels x out_channels
        auto w_size = this->w->get_dims();
        int w_s1 = w_size[3];
        int w_s2 = w_s1 * w_size[2];
        int w_s3 = w_s2 * w_size[1];
        auto b_ptr = this->b->get_data();
        // Y[i, j, oc] = sum(m, n, ic)(w[m, n, ic, oc]*X[i+m, j+m, ic]) + b[oc]
        int batch_size = in_data->get_dims()[0];
        bool overwrite = out_data->opaque(); // if opaque, then overwrite
        for (int batch = 0; batch < batch_size; batch++) {
            for (int i = 0; i < out_size[1]; i++) {
                for (int j = 0; j < out_size[2]; j++) {
                    for (int oc = 0; oc < out_channels; oc++) {
                        T sum = 0;
                        for (int m = 0; m < kernel_rows; m++) {
                            for (int n = 0; n < kernel_columns; n++) {
                                for (int ic = 0; ic < in_channels; ic++) {
                                    sum += w_ptr[m*w_s3 + n*w_s2 + ic*w_s1 + oc] *
                                           in_data_ptr[batch*in_s3 + (i+m)*in_s2 + (j+n)*in_s1 + ic];
                                }
                            }
                        }
                        sum += b_ptr[oc];
                        if (overwrite) {
                            out_data_ptr[batch*out_s3 + i*out_s2 + j*out_s1 + oc] = sum;
                        } else {
                            out_data_ptr[batch*out_s3 + i*out_s2 + j*out_s1 + oc] += sum;
                        }
                    }
                }
            }
        }
        out_data->setclear();
    }

    /* optimize with GEMM */
    template<typename T>
    void Convolution<T>::backward() {
        auto in_data = in_signal->get_data();
        auto out_grad = out_signal->get_grad();
        CHECK(!out_grad->opaque(), "out_grad should not be opaque");

        auto in_data_ptr = in_data->get_data(); // batch_size x num_rows x num_columns x in_channels
        auto in_size = in_data->get_dims();
        int in_s1 = in_size[3];
        int in_s2 = in_s1 * in_size[2];
        int in_s3 = in_s2 * in_size[1];
        auto out_grad_ptr = out_grad->get_data(); // batch_size x num_rows` x num_columns` x in_channels
        auto out_size = out_grad->get_dims();
        int out_s1 = out_size[3];
        int out_s2 = out_s1 * out_size[2];
        int out_s3 = out_s2 * out_size[1];
        auto dw_ptr = this->dw->get_data(); // num_rows x num_columns x in_channels x out_channels
        auto dw_size = this->dw->get_dims();
        int dw_s1 = dw_size[3];
        int dw_s2 = dw_s1 * dw_size[2];
        int dw_s3 = dw_s2 * dw_size[1];
        int batch_size = in_data->get_dims()[0];

        if (in_signal->get_type() == InnerSignal) {
            auto in_grad = in_signal->get_grad();
            auto in_grad_ptr = in_grad->get_data();
            auto w_ptr = this->w->get_data();
            bool dx_overwrite = in_grad->opaque();
            // D(X)[s, t, ic] = sum(m, n, oc)(D(Y)[s-m, t-n, oc] * w[m, n, ic, oc])
            for (int batch = 0; batch < batch_size; batch++) {
                for (int s = 0; s < num_rows; s++) {
                    for (int t = 0; t < num_columns; t++) {
                        for (int ic = 0; ic < in_channels; ic++) {
                            T sum = 0;
                            for (int m = 0; m < kernel_rows; m++) {
                                for (int n = 0; n < kernel_rows; n++) {
                                    for (int oc = 0; oc < out_channels; oc++) {
                                        if ((s - m) >= 0 && (s - m) < out_size[1] && (t - n) >= 0 && (t - n) < out_size[2]) {
                                            sum += out_grad_ptr[batch*out_s3 + (s-m)*out_s2 + (t-n)*out_s1 + oc] *
                                                w_ptr[m*dw_s3 + n*dw_s2 + ic*dw_s1 + oc];
                                        }
                                    }
                                }
                            }
                            if (dx_overwrite) {
                                in_grad_ptr[batch*in_s3 + s*in_s2 + t*in_s1 + ic] = sum;
                            } else {
                                in_grad_ptr[batch*in_s3 + s*in_s2 + t*in_s1 + ic] += sum;
                            }
                        }
                    }
                }
            }
            in_grad->setclear();
        }

        // D(w)[m, n, ic, oc] = sum(i, j)(D(Y)[i, j, oc] * X[i+m, j+n, ic])
        bool dw_overwrite = this->dw->opaque();
        for (int m = 0; m < kernel_rows; m++) {
            for (int n = 0; n < kernel_columns; n++) {
                for (int ic = 0; ic < in_channels; ic++) {
                    for (int oc = 0; oc < out_channels; oc++) {
                        T sum = 0;
                        for (int batch = 0; batch < batch_size; batch++) {
                            for (int i = 0; i < out_size[1]; i++) {
                                for (int j = 0; j < out_size[2]; j++) {
                                    sum += out_grad_ptr[batch*out_s3 + i*out_s2 + j*out_s1 + oc] *
                                           in_data_ptr[batch*in_s3 + (i+m)*in_s2 + (j+n)*in_s1 + ic];
                                }
                            }
                        }
                        if (dw_overwrite) {
                            dw_ptr[m*dw_s3 + n*dw_s2 + ic*dw_s1 + oc] = sum;
                        } else {
                            dw_ptr[m*dw_s3 + n*dw_s2 + ic*dw_s1 + oc] += sum;
                        }
                    }
                }
            }
        }
        this->dw->setclear();
        auto db_ptr = this->db->get_data();
        // D(b)[oc] = sum(i, j)(D(Y)[i, j, oc])
        bool db_overwrite = this->db->opaque();
        for (int oc = 0; oc < out_channels; oc++) {
            T sum = 0;
            for (int batch = 0; batch < batch_size; batch++) {
                for (int i = 0; i < out_size[1]; i++) {
                    for (int j = 0; j < out_size[2]; j++) {
                        sum += out_grad_ptr[batch*out_s3 + i*out_s2 + j*out_s1 + oc];
                    }
                }
            }
            if (db_overwrite) {
                db_ptr[oc] = sum;
            } else {
                db_ptr[oc] += sum;
            }
        }
        this->db->setclear();
    }

    template class Convolution<float>;
    template class Convolution<double>;
}
