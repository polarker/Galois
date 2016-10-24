#include "galois/narray.h"
#include "galois/narray_functors.h"
#include "galois/filters/linear.h"

using namespace std;

namespace gs {

    template<typename T>
    SP_Filter<T> Linear<T>::share() {
        bool just_for_share = true;
        auto res = make_shared<Linear<T>>(just_for_share);
        res->in_size = this->in_size;
        res->out_size = this->out_size;
        res->w = this->w;
        res->b = this->b;
        res->dw = this->dw;
        res->db = this->db;
        return res;
    }

    template<typename T>
    SP_Filter<T> Linear<T>::clone() {
        bool for_clone_or_share = true;
        auto res = make_shared<Linear<T>>(for_clone_or_share);
        res->in_size = this->in_size;
        res->out_size = this->out_size;
        res->w = make_shared<NArray<T>>(this->w->get_dims());
        res->w->copy_from(this->w);
        res->b = make_shared<NArray<T>>(this->b->get_dims());
        res->b->copy_from(this->b);
        res->dw = make_shared<NArray<T>>(this->dw->get_dims());
        res->db = make_shared<NArray<T>>(this->db->get_dims());
        return res;
    }

    template<typename T>
    Linear<T>::Linear(int in_size, int out_size) : in_size(in_size), out_size(out_size) {
        CHECK(in_size > 0 && out_size > 0, "both size should be positive");
        T s = sqrt(6. / (in_size + out_size));
        this->w  = make_shared<NArray<T>>(in_size, out_size);
        this->w->uniform(-s, s);
        this->b  = make_shared<NArray<T>>(out_size);
        this->b->uniform(-s, s);
        this->dw = make_shared<NArray<T>>(in_size, out_size);
        this->db = make_shared<NArray<T>>(out_size);
    }

    template<typename T>
    void Linear<T>::install_signals(const vector<SP_Signal<T>> &in_signals, const vector<SP_Signal<T>> &out_signals) {
        CHECK(in_signals.size() == 1, "only need 1 in signal");
        CHECK(out_signals.size() == 1, "only need 1 out signal");

        in_signal = in_signals[0];
        out_signal = out_signals[0];
    }

    template<typename T>
    void Linear<T>::set_dims(int batch_size) {
        if (in_signal->empty()) {
            in_signal->set_data_dims(batch_size, in_size);
        } else {
            auto in_dims = in_signal->get_data_dims();
            int in_batch_size = in_dims[0];
            int in_rest_dim = in_signal->get_data()->get_size() / in_batch_size;
            CHECK(in_dims.size() >= 2 && in_batch_size == batch_size && in_rest_dim == in_size, "the dimension of in signal is wrong");
        }
        if (out_signal->empty()) {
            out_signal->set_data_dims(batch_size, out_size);
        } else {
            CHECK(out_signal->get_data_dims() == vector<int>({batch_size, out_size}), "the dimension of out signal is wrong");
        }
    }

    template<typename T>
    void Linear<T>::reopaque() {
        this->dw->reopaque();
        this->db->reopaque();
    }

    template<typename T>
    vector<SP_NArray<T>> Linear<T>::get_params() {
        return vector<SP_NArray<T>>{ this->w, this->b };
    }

    template<typename T>
    vector<SP_NArray<T>> Linear<T>::get_grads() {
        return vector<SP_NArray<T>>{ this->dw, this->db };
    }

    template<typename T>
    void Linear<T>::forward() {
        auto in_data = in_signal->get_data();
        CHECK(!in_data->opaque(), "in_data should not be opaque");
        auto out_data = out_signal->get_data();

        GEMM(out_data, 'N', 'N', in_data, w);
        ADD_TO_ROW(out_data, b);
    }

    template<typename T>
    void Linear<T>::backward() {
        auto out_grad = out_signal->get_grad();
        CHECK(!out_grad->opaque(), "out_grad should not be opaque");
        if (in_signal->get_type() == InnerSignal) {
            auto in_grad = in_signal->get_grad();
            GEMM(in_grad, 'N', 'T', out_grad, this->w);
        }

        if (this->is_params_fixed()) {
            return;
        }

        auto in_data = in_signal->get_data();
        GEMM(this->dw, 'T', 'N', in_data, out_grad);
        SUM_TO_ROW(this->db, out_grad);
    }

    template class Linear<float>;
    template class Linear<double>;

}
