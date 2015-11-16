#ifndef _GALOIS_NET_H_
#define _GALOIS_NET_H_

#include "galois/base.h"
#include "galois/gfilters/base_net.h"
#include <string>
#include <vector>
#include <set>
#include <map>

using namespace std;

namespace gs
{
    
    template<typename T>
    class Net : public BaseNet<T>
    {
    private:
        map<string, vector<tuple<string, int>>> fp_graph = {};
        map<string, vector<tuple<string, int>>> bp_graph = {};
        
    private:
        void _set_fp_order(const string, vector<int>&);
        
    public:
        Net() {}
        Net(const Net& other) = delete;
        Net& operator=(const Net&) = delete;
        
        void add_link(const vector<string>&, const vector<string>&, SP_Filter<T>) override;
        void add_input_ids(const string);
        void add_input_ids(const initializer_list<string>);
        void add_input_ids(const vector<string>&);
        void add_output_ids(const string);
        void add_output_ids(const initializer_list<string>);
        void add_output_ids(const vector<string>&);
        void set_p_order();
        
        SP_Filter<T> share() override;
    };

}

#endif
