#ifndef _GALOIS_BASENET_H_
#define _GALOIS_BASENET_H_

#include "galois/base.h"
#include <string>
#include <vector>
#include <set>
#include <map>
#include <algorithm>

using namespace std;

namespace gs
{

    template<typename T>
    class BaseNet : public GFilter<T>
    {
    protected:
        vector<tuple<const vector<string>, const vector<string>, SP_Filter<T>>> links = {};

        map<string, SP_Signal<T>> inner_signals = {};

        vector<string> input_ids = {};
        vector<string> output_ids = {};

        vector<SP_Filter<T>> fp_filters = {};

        // the network can be fixed only when members in above are set
        bool fixed = false;

    protected:
        void _remove_signal(string);
        SP_Signal<T> _get_signal(string id,
                                 const vector<SP_Signal<T>> in_signals,
                                 const vector<SP_Signal<T>> out_signals);
        vector<SP_Signal<T>> _get_signal(vector<string> id,
                                         const vector<SP_Signal<T>> in_signals,
                                         const vector<SP_Signal<T>> out_signals);

    public:
        BaseNet() {}
        BaseNet(const BaseNet& other) = delete;
        BaseNet& operator=(const BaseNet&) = delete;

        virtual void add_link(const vector<string>&, const vector<string>&, SP_Filter<T>) = 0;
        void add_link(const initializer_list<string>, const initializer_list<string>, SP_Filter<T>);
        void add_link(const initializer_list<string>, const string, SP_Filter<T>);
        void add_link(const string, const initializer_list<string>, SP_Filter<T>);
        void add_link(const string, const string, SP_Filter<T>);

        // in order to share a net, methods above should be called and methods below should not be called
        set<SP_PFilter<T>> get_pfilters() override;

        void install_signals(const vector<SP_Signal<T>> &in_signals, const vector<SP_Signal<T>> &out_signals) override;
        void set_dims(int batch_size) override;
        void reopaque() override;

        void forward() override;
        void backward() override;
    };

    template<typename T>
    void BaseNet<T>::add_link(const initializer_list<string> ins, const initializer_list<string> outs, SP_Filter<T> filter){
        CHECK(!fixed, "network should not be fixed");
        add_link(vector<string>(ins), vector<string>(outs), filter);
    }

    template<typename T>
    void BaseNet<T>::add_link(const initializer_list<string> ins, const string outs, SP_Filter<T> filter){
        CHECK(!fixed, "network should not be fixed");
        add_link(ins, {outs}, filter);
    }

    template<typename T>
    void BaseNet<T>::add_link(const string ins, const initializer_list<string> outs, SP_Filter<T> filter){
        CHECK(!fixed, "network should not be fixed");
        add_link({ins}, outs, filter);
    }

    template<typename T>
    void BaseNet<T>::add_link(const string ins, const string outs, SP_Filter<T> filter){
        CHECK(!fixed, "network should not be fixed");
        add_link({ins}, {outs}, filter);
    }

    template<typename T>
    void BaseNet<T>::_remove_signal(string id) {
        if (inner_signals.count(id) > 0) {
            inner_signals.erase(id);
        }
    }

    template<typename T>
    SP_Signal<T> BaseNet<T>::_get_signal(string id,
                                         const vector<SP_Signal<T>> in_signals,
                                         const vector<SP_Signal<T>> out_signals) {
        auto in_idx = find(input_ids.begin(), input_ids.end(), id);
        if (in_idx != input_ids.end()) {
            auto idx = in_idx - input_ids.begin();
            return in_signals[idx];
        }
        auto out_idx = find(output_ids.begin(), output_ids.end(), id);
        if (out_idx != output_ids.end()) {
            auto idx = out_idx - output_ids.begin();
            return out_signals[idx];
        }
        return inner_signals[id];
    }

    template<typename T>
    vector<SP_Signal<T>> BaseNet<T>::_get_signal(vector<string> ids,
                                                 const vector<SP_Signal<T>> in_signals,
                                                 const vector<SP_Signal<T>> out_signals) {
        vector<SP_Signal<T>> res{};
        for (auto id : ids) {
            res.push_back(_get_signal(id, in_signals, out_signals));
        }
        return res;
    }

    template<typename T>
    set<SP_PFilter<T>> BaseNet<T>::get_pfilters() {
        CHECK(fixed, "network should be fixed");
        set<SP_PFilter<T>> pfilters{};
        for (auto filter : fp_filters) {
            if (auto p = dynamic_pointer_cast<PFilter<T>>(filter)) {
                pfilters.insert(p);
            }
            if (auto g = dynamic_pointer_cast<GFilter<T>>(filter)) {
                auto s = g->get_pfilters();
                pfilters.insert(s.begin(), s.end());
            }
        }
        return pfilters;
    }

    template<typename T>
    void BaseNet<T>::install_signals(const vector<SP_Signal<T>> &in_signals, const vector<SP_Signal<T>> &out_signals) {
        CHECK(fixed, "network should be fixed");
        for (auto t : links) {
            auto ins = get<0>(t);
            auto outs = get<1>(t);
            auto filter = get<2>(t);
            filter->install_signals(_get_signal(ins, in_signals, out_signals),
                                    _get_signal(outs, in_signals, out_signals));
        }
    }

    template<typename T>
    void BaseNet<T>::set_dims(int batch_size) {
        CHECK(fixed, "network should be fixed");
        CHECK(!fp_filters.empty(), "fp filters should have been set");
        for (auto filter : fp_filters) {
            filter->set_dims(batch_size);
        }
    }

    template<typename T>
    void BaseNet<T>::reopaque() {
        CHECK(fixed, "network should be fixed");
        for (auto &kv : inner_signals) {
            auto signal = kv.second;
            signal->reopaque();
        }
        for (auto filter : fp_filters) {
            filter->reopaque();
        }
    }

    template<typename T>
    void BaseNet<T>::forward() {
        CHECK(fixed, "network should be fixed");
        for (auto filter : fp_filters) {
            filter->forward();
        }
    }

    template<typename T>
    void BaseNet<T>::backward() {
        CHECK(fixed, "network should be fixed");
        for (int i = fp_filters.size()-1; i >= 0; i--) {
            auto filter = fp_filters[i];
            filter->backward();
        }
    }

}

#endif
