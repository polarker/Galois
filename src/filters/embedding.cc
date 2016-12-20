#include "galois/narray.h"
#include "galois/narray_functors.h"
#include "galois/filters/embedding.h"

using namespace std;

namespace gs {

    template<typename T>
    SP_Filter<T> Embedding<T>::share() {
        bool just_for_share = true;
        auto res = make_shared<Embedding<T>>(just_for_share);
        res->in_size = this->in_size;
        res->out_size = this->out_size;
        res->w = this->w;
        res->dw = this->dw;
        return res;
    }

    template<typename T>
    SP_Filter<T> Embedding<T>::clone() {
        bool just_for_clone = true;
        auto res = make_shared<Embedding<T>>(just_for_clone);
        res->in_size = this->in_size;
        res->out_size = this->out_size;
        res->w = make_shared<NArray<T>>(this->w->get_dims());
        res->w->copy_from(this->w);
        res->dw = make_shared<NArray<T>>(this->dw->get_dims());
        return res;
    }

    template<typename T>
    Embedding<T>::Embedding(size_t in_size, size_t out_size) : in_size(in_size), out_size(out_size) {
        CHECK(in_size > 0 && out_size > 0, "both size should be positive");
        T s = sqrt(6. / (in_size + out_size));
        this->w  = make_shared<NArray<T>>(in_size, out_size);
        this->w->uniform(-s, s);
        this->dw = make_shared<NArray<T>>(in_size, out_size);
    }

    template<typename T>
    void Embedding<T>::install_signals(const vector<SP_Signal<T>> &in_signals, const vector<SP_Signal<T>> &out_signals) {
        CHECK(in_signals.size() == 1, "only need 1 in signal");
        CHECK(out_signals.size() == 1, "only need 1 out signal");

        in_signal = in_signals[0];
        out_signal = out_signals[0];
    }

    template<typename T>
    void Embedding<T>::set_dims(size_t batch_size) {
        CHECK(in_signal->empty(), "in signal should be empty");
        in_signal->set_data_dims(batch_size);
        if (out_signal->empty()) {
            out_signal->set_data_dims(batch_size, out_size);
        } else {
            CHECK(out_signal->get_data_dims() == vector<size_t>({batch_size, out_size}), "the dimension of out signal is wrong");
        }
    }

    template<typename T>
    void Embedding<T>::reopaque() {
        this->dw->reopaque();
    }

    template<typename T>
    vector<SP_NArray<T>> Embedding<T>::get_params() {
        return vector<SP_NArray<T>>{ this->w };
    }

    template<typename T>
    vector<SP_NArray<T>> Embedding<T>::get_grads() {
        return vector<SP_NArray<T>>{ this->dw };
    }

    template<typename T>
    void Embedding<T>::forward() {
        auto in_data = in_signal->get_data();
        CHECK(!in_data->opaque(), "in_data should not be opaque");
        auto out_data = out_signal->get_data();

        TAKE_ROWS(out_data, in_data, w);
    }

    template<typename T>
    void Embedding<T>::backward() {
        if (this->is_params_fixed()) {
            return;
        }

        auto in_data = in_signal->get_data();
        auto out_grad = out_signal->get_grad();
        CHECK(!out_grad->opaque(), "out_grad should not be opaque");

        PUT_ROWS(this->dw, in_data, out_grad);
    }

    template class Embedding<float>;
    template class Embedding<double>;

}
