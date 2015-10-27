#include "galois/narray.h"
#include "galois/filters/linear.h"
#include <cassert>

using namespace std;

namespace gs {
    
    template<typename T>
    Linear<T>::Linear(int in_size, int out_size) : in_size(in_size), out_size(out_size) {
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
        CHECK(in_signal == nullptr, "in signal should not be initialized");
        CHECK(out_signal == nullptr, "out signal should not be initialized");
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
            CHECK(in_signal->get_data_dims() ==  vector<int>({batch_size, in_size}), "the dimension of in signal is wrong");
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
        auto out_data = out_signal->get_data();
        
        GEMM(out_data, 'N', 'N', in_data, w);
        ADD_TO_ROW(out_data, b);
    }

    template<typename T>
    void Linear<T>::backward() {
        auto in_data = in_signal->get_data();
        auto out_grad = out_signal->get_grad();

        GEMM(this->dw, 'T', 'N', in_data, out_grad);
        SUM_TO_ROW(this->db, out_grad);
        
        if (in_signal->get_type() == InnerSignal) {
            auto in_grad = in_signal->get_grad();
            GEMM(in_grad, 'N', 'T', out_grad, this->w);
        }
    }
    
    template class Linear<float>;
    template class Linear<double>;

}
