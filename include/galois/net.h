#ifndef _GALOIS_NET_H_
#define _GALOIS_NET_H_

#include "galois/base.h"
#include "galois/net.h"
#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <map>

using namespace std;

namespace gs
{
    template<typename T>
    class Net : public GFilter<T>
    {
    public:
        vector<tuple<const vector<string>, const vector<string>, SP_Filter<T>>> links;
        vector<tuple<const vector<SP_Signal<T>>, const vector<SP_Signal<T>>, SP_Filter<T>>> compiled_links;
        set<SP_PFilter<T>> pfilters;
        
        map<string, vector<tuple<string, int>>> fp_graph;
        map<string, vector<tuple<string, int>>> bp_graph;
        
        vector<int> fp_order;
        vector<int> bp_order;
        
        map<string, SP_Signal<T>> inner_signals;
        
        vector<string> input_ids;
        vector<string> output_ids;
        
    public:
        Net();
        Net(const Net& other) = delete;
        Net& operator=(const Net&) = delete;
        
        set<SP_PFilter<T>> get_pfilters() override { return pfilters; }
        
        void add_link(const initializer_list<string>, const initializer_list<string>, SP_Filter<T>);
        void add_link(const initializer_list<string>, const string, SP_Filter<T>);
        void add_link(const string, const initializer_list<string>, SP_Filter<T>);
        void add_link(const string, const string, SP_Filter<T>);
        void _remove_signal(string);
        void set_input_ids(const string);
        void set_input_ids(const initializer_list<string>);
        void set_input_ids(const vector<string>);
        void set_output_ids(const string);
        void set_output_ids(const initializer_list<string>);
        void set_output_ids(const vector<string>);
        void _set_fp_order(const string);
        void _set_bp_order(const string);
        void set_p_order();
        SP_Signal<T> _get_signal(string id,
                                       const vector<SP_Signal<T>> in_signals,
                                       const vector<SP_Signal<T>> out_signals);
        vector<SP_Signal<T>> _get_signal(vector<string> id,
                                               const vector<SP_Signal<T>> in_signals,
                                               const vector<SP_Signal<T>> out_signals);
        void compile(const vector<SP_Signal<T>> in_signal, const vector<SP_Signal<T>> out_signal);
        void set_dims(const SP_Signal<T> in_signal,
                      const SP_Signal<T> out_signal,
                      int batch_size);
        void set_dims(const initializer_list<SP_Signal<T>> &in_signals,
                      const initializer_list<SP_Signal<T>> &out_signals,
                      int batch_size);
        void set_dims(const vector<SP_Signal<T>> &in_signals,
                      const vector<SP_Signal<T>> &out_signals,
                      int batch_size) override;
        void reopaque() override;
        
        void forward(const vector<SP_Signal<T>> &in_signals, const vector<SP_Signal<T>> &out_signals) override;
        
        void backward(const vector<SP_Signal<T>> &in_signals, const vector<SP_Signal<T>> &out_signals) override;
    };

}
            
#endif
