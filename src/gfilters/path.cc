#include "galois/gfilters/path.h"
#include "galois/utils.h"

namespace gs
{

    template<typename T>
    void Path<T>::add_filter(SP_Filter<T> filter) {
        links.push_back(filter);

        if (auto p = dynamic_pointer_cast<PFilter<T>>(filter)) {
            pfilters.insert(p);
        }
        if (auto g = dynamic_pointer_cast<GFilter<T>>(filter)) {
            auto s = g->get_pfilters();
            pfilters.insert(s.begin(), s.end());
        }

        if (links.size() > 1) {
            inner_signals.push_back(make_shared<Signal<T>>(InnerSignal));
        }
    }

    template<typename T>
    SP_Filter<T> Path<T>::share() {
        auto res = make_shared<Path<T>>();
        for (auto const& filter : links) {
            auto copy_of_filter = filter->share();
            res->add_filter(copy_of_filter);
        }
        return res;
    }

    template<typename T>
    SP_Filter<T> Path<T>::clone() {
        auto res = make_shared<Path<T>>();
        for (auto const& filter : links) {
            auto clone_of_filter = filter->clone();
            res->add_filter(clone_of_filter);
        }
        return res;
    }

    template<typename T>
    set<SP_PFilter<T>> Path<T>::get_pfilters() {
        return pfilters;
    }

    template<typename T>
    void Path<T>::install_signals(const vector<SP_Signal<T>>& in_signals, const vector<SP_Signal<T>>& out_signals) {
        CHECK(in_signals.size() == 1 && out_signals.size() == 1, "Only support 1 in signal and 1 out signal right now");
        CHECK(links.size() == inner_signals.size()+1, "The number of filters and inner signals does not match");
        auto in_signal = in_signals[0];
        auto out_signal = out_signals[0];
        for (size_t i = 0; i < links.size(); i++) {
            SP_Signal<T> in = nullptr;
            SP_Signal<T> out = nullptr;
            if (i == 0) {
                in = in_signal;
            } else {
                in = inner_signals[i-1];
            }
            if (i == links.size()-1) {
                out = out_signal;
            } else {
                out = inner_signals[i];
            }
            auto filter = links[i];
            filter->install_signals(vector<SP_Signal<T>>{in}, vector<SP_Signal<T>>{out});
        }
    }

    template<typename T>
    void Path<T>::set_dims(size_t batch_size) {
        CHECK(links.size() == inner_signals.size()+1, "The number of filters and inner signals does not match");
        for (auto const& filter : links) {
            filter->set_dims(batch_size);
        }
    }

    template<typename T>
    void Path<T>::reopaque() {
        for (auto const& signal : inner_signals) {
            signal->reopaque();
        }
        for (auto const& filter : links) {
            filter->reopaque();
        }
    }

    template<typename T>
    void Path<T>::forward() {
        for (auto const& filter : links) {
            filter->forward();
        }
    }

    template<typename T>
    void Path<T>::backward() {
        for (int i = links.size()-1; i >= 0; i--) {
            auto filter = links[i];
            filter->backward();
        }
    }

    template class Path<float>;
    template class Path<double>;

}