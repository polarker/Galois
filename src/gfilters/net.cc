#include "galois/gfilters/net.h"
#include "galois/utils.h"
#include <vector>

namespace gs {
    
    template<typename T>
    Net<T>::Net() : links{}
            , pfilters{}
            , fp_graph{}
            , bp_graph{}
            , inner_signals{}
            , input_ids{}
            , output_ids{}
            , fp_order{}
            , bp_order{}{
    }
    
    template<typename T>
    void Net<T>::add_link(const vector<string> &ins, const vector<string> &outs, SP_Filter<T> filter){
        // add to links
        auto idx = links.size();
        links.push_back(tuple<const vector<string>,const vector<string>,SP_Filter<T>>
                        (vector<string>{ins}, vector<string>{outs}, filter));
        
        // add to pfilters
        if (auto p = dynamic_pointer_cast<PFilter<T>>(filter)) {
            pfilters.insert(p);
        }
        if (auto g = dynamic_pointer_cast<GFilter<T>>(filter)) {
            auto s = g->get_pfilters();
            pfilters.insert(s.begin(), s.end());
        }
        
        // add to graph
        for (auto s : ins) {
            if (fp_graph.count(s) == 0) {
                fp_graph[s] = vector<tuple<string,int>>();
            }
        }
        for (auto s : outs) {
            if (bp_graph.count(s) == 0) {
                bp_graph[s] = vector<tuple<string,int>>();
            }
        }
        for (auto s1 : ins) {
            for (auto s2 : outs) {
                fp_graph[s1].push_back(tuple<string,int>(s2, idx));
                bp_graph[s2].push_back(tuple<string,int>(s1, idx));
            }
        }
        
        // initial signals
        for (auto s : ins) {
            if (!Contains(input_ids, s) && (inner_signals.count(s) == 0)) {
                inner_signals[s] = make_shared<Signal<T>>(InnerSignal);
            }
        }
        for (auto s : outs) {
            if (!Contains(output_ids, s) && (inner_signals.count(s) == 0)) {
                inner_signals[s] = make_shared<Signal<T>>(InnerSignal);
            }
        }
    }
    
    template<typename T>
    void Net<T>::add_link(const initializer_list<string> ins, const initializer_list<string> outs, SP_Filter<T> filter){
        add_link(vector<string>(ins), vector<string>(outs), filter);
    }
    
    template<typename T>
    void Net<T>::add_link(const initializer_list<string> ins, const string outs, SP_Filter<T> filter){
        add_link(ins, {outs}, filter);
    }
    
    template<typename T>
    void Net<T>::add_link(const string ins, const initializer_list<string> outs, SP_Filter<T> filter){
        add_link({ins}, outs, filter);
    }
    
    template<typename T>
    void Net<T>::add_link(const string ins, const string outs, SP_Filter<T> filter){
        add_link({ins}, {outs}, filter);
    }
    
    template<typename T>
    void Net<T>::_remove_signal(string id) {
        if (inner_signals.count(id) > 0) {
            inner_signals.erase(id);
        }
    }
    
    template<typename T>
    void Net<T>::set_input_ids(const string id) {
        set_input_ids({id});
    }
    
    template<typename T>
    void Net<T>::set_input_ids(const initializer_list<string> ids) {
        set_input_ids(vector<string>(ids));
    }
    
    template<typename T>
    void Net<T>::set_input_ids(const vector<string>& ids) {
        CHECK(input_ids.empty(), "input_ids should not be set");
        input_ids.insert(input_ids.end(), ids.begin(), ids.end());
        for (auto id : ids) {
            _remove_signal(id);
        }
    }
    
    template<typename T>
    void Net<T>::set_output_ids(const string id) {
        set_output_ids({id});
    }
    
    template<typename T>
    void Net<T>::set_output_ids(const initializer_list<string> ids) {
        set_output_ids(vector<string>(ids));
    }
    
    template<typename T>
    void Net<T>::set_output_ids(const vector<string>& ids) {
        CHECK(output_ids.empty(), "output_ids should not be set");
        output_ids.insert(output_ids.end(), ids.begin(), ids.end());
        for (auto id : ids) {
            _remove_signal(id);
        }
    }
    
    template<typename T>
    void Net<T>::_set_fp_order(string out_id) {
        // input_ids should be non-empty
        CHECK(!input_ids.empty(), "input ids should have been set");
        if (Contains(input_ids, out_id)) {
            if (bp_graph.count(out_id) == 0) {
                return;
            }
        } else {
            CHECK(bp_graph.count(out_id) == 1, "out_id should be in the keys of bp_graph");
        }
        
        for (auto t : bp_graph[out_id]) {
            string in_id = get<0>(t);
            int link_idx = get<1>(t);
            if (!Contains(fp_order, link_idx)) {
                _set_fp_order(in_id);
            }
        }
        for (auto t : bp_graph[out_id]) {
            int link_idx = get<1>(t);
            if (!Contains(fp_order, link_idx)) {
                fp_order.push_back(link_idx);
            }
        }
    }
    
    template<typename T>
    void Net<T>::_set_bp_order(string in_id) {
        CHECK(!output_ids.empty(), "output ids should have been set");
        if (Contains(output_ids, in_id)) {
            if (fp_graph.count(in_id) == 0) {
                return;
            }
        } else {
            CHECK(fp_graph.count(in_id) == 1, "in_id should be in the keys of fp_graph");
        }
        for (auto t : fp_graph[in_id]) {
            string out_id = get<0>(t);
            int link_idx = get<1>(t);
            if (!Contains(bp_order, link_idx)) {
                _set_bp_order(out_id);
            }
        }
        for (auto t : fp_graph[in_id]) {
            int link_idx = get<1>(t);
            if (!Contains(bp_order, link_idx)) {
                bp_order.push_back(link_idx);
            }
        }
    }
    
    template<typename T>
    void Net<T>::set_p_order() {
        CHECK(fp_order.empty(), "fp order should not be set");
        CHECK(bp_order.empty(), "bp order should not be set");
        CHECK(fp_filters.empty(), "fp filters should not be set");
        CHECK(bp_filters.empty(), "bp filters should not be set");
        for (auto out_id : output_ids) {
            _set_fp_order(out_id);
        }
        for (auto in_id : input_ids) {
            _set_bp_order(in_id);
        }
        for (auto link_idx : fp_order) {
            auto t = links[link_idx];
            auto filter = get<2>(t);
            fp_filters.push_back(filter);
        }
        for (auto link_idx : bp_order) {
            auto t = links[link_idx];
            auto filter = get<2>(t);
            bp_filters.push_back(filter);
        }
    }
    
    template<typename T>
    SP_Filter<T> Net<T>::share() {
        CHECK(!fp_order.empty() && !bp_order.empty(), "these order should have been set and the net is fixed");
        auto res = make_shared<Net<T>>();
        for (auto t : this->links) {
            auto ins = get<0>(t);
            auto outs = get<1>(t);
            auto filter = get<2>(t);
            auto copy_of_filter = filter->share();
            res->add_link(ins, outs, copy_of_filter);
        }
        res->set_input_ids(this->input_ids);
        res->set_output_ids(this->output_ids);
        res->set_p_order();
        
        CHECK(res->fp_graph == this->fp_graph &&
              res->bp_graph == this->bp_graph &&
              res->fp_order == this->fp_order &&
              res->bp_order == this->bp_order, "these should be equal");
        
        return res;
    }
    
    template<typename T>
    set<SP_PFilter<T>> Net<T>::get_pfilters() {
        CHECK(!fp_order.empty() && !bp_order.empty(), "these order should have been set and the net is fixed");
        return pfilters;
    }
    
    template<typename T>
    SP_Signal<T> Net<T>::_get_signal(string id,
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
    vector<SP_Signal<T>> Net<T>::_get_signal(vector<string> ids,
                                                const vector<SP_Signal<T>> in_signals,
                                                const vector<SP_Signal<T>> out_signals) {
        vector<SP_Signal<T>> res{};
        for (auto id : ids) {
            res.push_back(_get_signal(id, in_signals, out_signals));
        }
        return res;
    }
    
    template<typename T>
    void Net<T>::install_signals(const vector<SP_Signal<T>> &in_signals, const vector<SP_Signal<T>> &out_signals) {
        for (auto t : links) {
            auto ins = get<0>(t);
            auto outs = get<1>(t);
            auto filter = get<2>(t);
            filter->install_signals(_get_signal(ins, in_signals, out_signals),
                                    _get_signal(outs, in_signals, out_signals));
        }
    }
    
    template<typename T>
    void Net<T>::set_dims(int batch_size) {
        CHECK(!fp_order.empty(), "fp order should have been set");
        CHECK(!bp_order.empty(), "bp order should have been set");
        for (auto filter : fp_filters) {
            filter->set_dims(batch_size);
        }
    }

    template<typename T>
    void Net<T>::reopaque() {
        for (auto &kv : inner_signals) {
            auto signal = kv.second;
            signal->reopaque();
        }
        for (auto filter : fp_filters) {
            filter->reopaque();
        }
    }
    
    template<typename T>
    void Net<T>::forward() {
        CHECK(!fp_filters.empty(), "fp filters should have been set");
        for (auto filter : fp_filters) {
            filter->forward();
        }
    }

    template<typename T>
    void Net<T>::backward() {
        CHECK(!bp_filters.empty(), "bp filters should have been set");
        for (auto filter : bp_filters) {
            filter->backward();
        }
    }
    
    template class Net<float>;
    template class Net<double>;
    
}
