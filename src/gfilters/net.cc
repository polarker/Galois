#include "galois/gfilters/net.h"
#include "galois/utils.h"
#include <vector>

namespace gs {

    template<typename T>
    void Net<T>::add_link(const vector<string> &ins, const vector<string> &outs, SP_Filter<T> filter) {
        CHECK(!this->fixed, "network should not be fixed");

        // add to links
        auto idx = this->links.size();
        this->links.push_back(tuple<const vector<string>,const vector<string>,SP_Filter<T>>
                        (vector<string>{ins}, vector<string>{outs}, filter));

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
            if (!Contains(this->input_ids, s) && (this->inner_signals.count(s) == 0)) {
                this->inner_signals[s] = make_shared<Signal<T>>(InnerSignal);
            }
        }
        for (auto s : outs) {
            if (!Contains(this->output_ids, s) && (this->inner_signals.count(s) == 0)) {
                this->inner_signals[s] = make_shared<Signal<T>>(InnerSignal);
            }
        }
    }

    template<typename T>
    void Net<T>::add_input_ids(const string id) {
        add_input_ids({id});
    }

    template<typename T>
    void Net<T>::add_input_ids(const initializer_list<string> ids) {
        add_input_ids(vector<string>(ids));
    }

    template<typename T>
    void Net<T>::add_input_ids(const vector<string>& ids) {
        CHECK(!this->fixed, "network should not be fixed");
        CHECK(this->input_ids.empty(), "input_ids should not be set");
        this->input_ids.insert(this->input_ids.end(), ids.begin(), ids.end());
        for (auto id : ids) {
            this->_remove_signal(id);
        }
    }

    template<typename T>
    void Net<T>::add_output_ids(const string id) {
        add_output_ids({id});
    }

    template<typename T>
    void Net<T>::add_output_ids(const initializer_list<string> ids) {
        add_output_ids(vector<string>(ids));
    }

    template<typename T>
    void Net<T>::add_output_ids(const vector<string>& ids) {
        CHECK(!this->fixed, "network should not be fixed");
        CHECK(this->output_ids.empty(), "output_ids should not be set");
        this->output_ids.insert(this->output_ids.end(), ids.begin(), ids.end());
        for (auto id : ids) {
            this->_remove_signal(id);
        }
    }

    template<typename T>
    void Net<T>::_set_fp_order(string out_id, vector<int> &fp_order) {
        if (Contains(this->input_ids, out_id)) {
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
                _set_fp_order(in_id, fp_order);
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
    void Net<T>::set_p_order() {
        CHECK(!this->fixed, "network should not be fixed");
        CHECK(!this->input_ids.empty(), "input ids should have been set");
        CHECK(!this->output_ids.empty(), "output ids should have been set");
        CHECK(this->fp_filters.empty(), "fp filters should not be set");
        this->fixed = true;

        vector<int> fp_order{};
        for (auto out_id : this->output_ids) {
            _set_fp_order(out_id, fp_order);
        }
        for (auto link_idx : fp_order) {
            auto t = this->links[link_idx];
            auto filter = get<2>(t);
            this->fp_filters.push_back(filter);
        }
    }

    template<typename T>
    SP_Filter<T> Net<T>::share() {
        CHECK(this->fixed, "the network should be fixed");
        auto res = make_shared<Net<T>>();
        for (auto t : this->links) {
            auto ins = get<0>(t);
            auto outs = get<1>(t);
            auto filter = get<2>(t);
            auto copy_of_filter = filter->share();
            res->add_link(ins, outs, copy_of_filter);
        }
        res->add_input_ids(this->input_ids);
        res->add_output_ids(this->output_ids);
        res->set_p_order();

        CHECK(res->fp_filters == this->fp_filters, "these should be equal");

        return res;
    }

    template<typename T>
    SP_Filter<T> Net<T>::clone() {
        CHECK(this->fixed, "the network should be fixed");
        auto res = make_shared<Net<T>>();
        for (auto t : this->links) {
            auto ins = get<0>(t);
            auto outs = get<1>(t);
            auto filter = get<2>(t);
            auto clone_of_filter = filter->clone();
            res->add_link(ins, outs, clone_of_filter);
        }
        res->add_input_ids(this->input_ids);
        res->add_output_ids(this->output_ids);
        res->set_p_order();

        return res;
    }

    template class Net<float>;
    template class Net<double>;

}
