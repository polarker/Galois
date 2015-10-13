#ifndef _GALOIS_NET_H_
#define _GALOIS_NET_H_

#include <galois/base.h>
#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <map>

using namespace std;

namespace gs
{
    
    class Net : public GFilter
    {
    public:
        vector<tuple<const vector <string>, const vector <string>, shared_ptr<Filter>>> links;
        set<shared_ptr<PFilter>> pfilters;
        
        map<string, vector<tuple<string, int>>> fp_graph;
        map<string, vector<tuple<string, int>>> bp_graph;
        
        vector<int> fp_order;
        vector<int> bp_order;
        
        map<string, shared_ptr<Signal>> inner_signals;
        
        vector<string> input_ids;
        vector<string> output_ids;
        
    public:
        Net();
        Net( const Net& other ) = delete;
        Net& operator=( const Net& ) = delete;
        
        set<shared_ptr<PFilter>> get_pfilters() override { return pfilters; }
        
        void add_link(const initializer_list<string>, const initializer_list<string>, shared_ptr<Filter>);
        void add_link(const initializer_list<string>, const string, shared_ptr<Filter>);
        void add_link(const string, const initializer_list<string>, shared_ptr<Filter>);
        void add_link(const string, const string, shared_ptr<Filter>);
        void set_input_ids(const string);
        void set_input_ids(const string[]);
        void set_output_ids(const string);
        void set_output_ids(const string[]);
        void _set_fp_order(const string);
        void _set_bp_order(const string);
        void set_p_order();
        shared_ptr<Signal> _get_signal(string id,
                                       const vector<shared_ptr<Signal>> in_signals,
                                       const vector<shared_ptr<Signal>> out_signals);
        void set_dims(const vector<shared_ptr<Signal>> in_signals,
                      const vector<shared_ptr<Signal>> out_signals,
                      int batch_size) override;
        
        void Forward(shared_ptr<Signal> inputs, shared_ptr<Signal> outputs) override {
            inputs->opaque = true;
            outputs->opaque = true;
            cout << "forward" << endl;
        }
            
        void Backward(shared_ptr<Signal> inputs, shared_ptr<Signal> outputs) override {
            inputs->opaque = true;
            outputs->opaque = true;
            cout << "backward" << endl;
        }
    };

}
            
#endif
