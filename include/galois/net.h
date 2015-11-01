#ifndef _GALOIS_NET_H_
#define _GALOIS_NET_H_

#include "galois/base.h"
#include "galois/net.h"
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
        set<SP_PFilter<T>> pfilters;
        
        map<string, vector<tuple<string, int>>> fp_graph;
        map<string, vector<tuple<string, int>>> bp_graph;
        
        map<string, SP_Signal<T>> inner_signals;
        
        vector<string> input_ids;
        vector<string> output_ids;
        
        vector<int> fp_order;
        vector<SP_Filter<T>> fp_filters;
        vector<int> bp_order;
        vector<SP_Filter<T>> bp_filters;
        
    public:
        Net();
        Net(const Net& other) = delete;
        Net& operator=(const Net&) = delete;
        
        void add_link(const vector<string>&, const vector<string>&, SP_Filter<T>);
        void _remove_signal(string);
        void set_input_ids(const string);
        void set_input_ids(const initializer_list<string>);
        void set_input_ids(const vector<string>&);
        void set_output_ids(const string);
        void set_output_ids(const initializer_list<string>);
        void set_output_ids(const vector<string>&);
        void _set_fp_order(const string);
        void _set_bp_order(const string);
        void set_p_order();
        
        // in order to share a net, methods above should be called and methods below should not be called
        SP_Filter<T> share() override;
        set<SP_PFilter<T>> get_pfilters() override;
        
        SP_Signal<T> _get_signal(string id,
                                 const vector<SP_Signal<T>> in_signals,
                                 const vector<SP_Signal<T>> out_signals);
        vector<SP_Signal<T>> _get_signal(vector<string> id,
                                         const vector<SP_Signal<T>> in_signals,
                                         const vector<SP_Signal<T>> out_signals);
        void install_signals(const vector<SP_Signal<T>> &in_signals, const vector<SP_Signal<T>> &out_signals) override;
        void set_dims(int batch_size) override;
        void reopaque() override;
        
        void forward() override;
        void backward() override;
    };

}
            
#endif
