#include "galois/net.h"
#include "galois/utils.h"
#include <vector>
#include <cassert>

namespace gs {

void Net::add_link(const initializer_list<string> ins, const initializer_list<string> outs, Filter *filter){
    // add to links
    auto idx = links.size();
    links.push_back(tuple<const vector<string>,const vector<string>,Filter*>
            (vector<string>{ins}, vector<string>{outs}, filter));

    // add to pfilters
    if (auto p = dynamic_cast<PFilter*>(filter)) {
        pfilters.insert(p);
    }
    if (auto g = dynamic_cast<GFilter*>(filter)) {
        //for(auto e : g->get_pfilters())
        //    pfilters.insert(e);
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
}

void Net::add_link(const initializer_list<string> ins, const string outs, Filter *filter){
    add_link(ins, {outs}, filter);
}

void Net::add_link(const string ins, const initializer_list<string> outs, Filter *filter){
    add_link({ins}, outs, filter);
}

void Net::add_link(const string ins, const string outs, Filter *filter){
    add_link({ins}, {outs}, filter);
}

void Net::set_input_ids(const string id) {
    input_ids.push_back(id);
}

void Net::set_output_ids(const string id) {
    output_ids.push_back(id);
}

void Net::_set_fp_order(string out_id) {
    // input_ids should be non-empty
    assert(!input_ids.empty());
    if (Contains(input_ids, out_id)) {
        return;
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

void Net::_set_bp_order(string in_id) {
    assert(!output_ids.empty());
    if (Contains(output_ids, in_id)) {
        return;
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
    
void Net::set_p_order() {
    for (auto out_id : output_ids) {
        _set_fp_order(out_id);
    }
    for (auto in_id : input_ids) {
        _set_bp_order(in_id);
    }
}

}
