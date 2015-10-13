#ifndef _GALOIS_NET_H_
#define _GALOIS_NET_H_

#include <Galois/base.h>
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
    vector<tuple<const vector <string>, const vector <string>, Filter*>> links;
    set<PFilter*> pfilters;

    map<string, vector<tuple<string, int>>> fp_graph;
    map<string, vector<tuple<string, int>>> bp_graph;

    vector<int> fp_order;
    vector<int> bp_order;

    map<string, Signal> inner_signals;

    vector<string> input_ids;
    vector<string> output_ids;

public:
    set<PFilter*> get_pfilters() override { return pfilters; }

    void add_link(const initializer_list<string>, const initializer_list<string>, Filter*);
    void add_link(const initializer_list<string>, const string, Filter*);
    void add_link(const string, const initializer_list<string>, Filter*);
    void add_link(const string, const string, Filter*);
    void set_input_ids(const string);
    void set_input_ids(const string[]);
    void set_output_ids(const string);
    void set_output_ids(const string[]);
    void _set_fp_order(const string);
    void _set_bp_order(const string);
    void set_p_order();

    void Forward(Signal *inputs, Signal *outputs) const override {
        inputs->opaque = true;
        outputs->opaque = true;
        cout << "forward" << endl;
    }

    void Backward(Signal *inputs, Signal *outputs) const override {
        inputs->opaque = true;
        outputs->opaque = true;
        cout << "backward" << endl;
    }
};

}

#endif
