#include "galois/gfilters/ordered_net.h"

namespace gs
{
    
    template<typename T>
    void OrderedNet<T>::add_link(const vector<string> &ins, const vector<string> &outs, SP_Filter<T> filter) {
        CHECK(!this->fixed, "network should not be fixed");
        
        this->links.push_back(tuple<const vector<string>,const vector<string>,SP_Filter<T>>
                              (vector<string>{ins}, vector<string>{outs}, filter));
        this->fp_filters.push_back(filter);
        
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
    void OrderedNet<T>::add_input_ids(const string id) {
        add_input_ids({id});
    }
    
    template<typename T>
    void OrderedNet<T>::add_input_ids(const initializer_list<string> ids) {
        add_input_ids(vector<string>(ids));
    }
    
    template<typename T>
    void OrderedNet<T>::add_input_ids(const vector<string>& ids) {
        CHECK(!this->fixed, "network should not be fixed");
        this->input_ids.insert(this->input_ids.end(), ids.begin(), ids.end());
        for (auto id : ids) {
            this->_remove_signal(id);
        }
    }
    
    template<typename T>
    void OrderedNet<T>::add_output_ids(const string id) {
        add_output_ids({id});
    }
    
    template<typename T>
    void OrderedNet<T>::add_output_ids(const initializer_list<string> ids) {
        add_output_ids(vector<string>(ids));
    }
    
    template<typename T>
    void OrderedNet<T>::add_output_ids(const vector<string>& ids) {
        CHECK(!this->fixed, "network should not be fixed");
        this->output_ids.insert(this->output_ids.end(), ids.begin(), ids.end());
        for (auto id : ids) {
            this->_remove_signal(id);
        }
    }
    
    template<typename T>
    void OrderedNet<T>::fix_net() {
        CHECK(!this->fixed, "network should not be fixed");
        this->fixed = true;

//        for (auto t : this->links) {
//            auto in_ids = get<0>(t);
//            auto out_ids = get<1>(t);
//            cout << in_ids[0] << " -> " << out_ids[0] <<endl;
//        }
    }
    
    template<typename T>
    SP_Filter<T> OrderedNet<T>::share() {
        CHECK(this->fixed, "network should be fixed");
        auto res = make_shared<OrderedNet<T>>();
        for (auto t : this->links) {
            auto ins = get<0>(t);
            auto outs = get<1>(t);
            auto filter = get<2>(t);
            auto copy_of_filter = filter->share();
            res->add_link(ins, outs, copy_of_filter);
        }
        res->add_input_ids(this->input_ids);
        res->add_output_ids(this->output_ids);
        res->fix_net();
        
        return res;
    }
    
    template<typename T>
    void OrderedNet<T>::forward(int idx) {
        CHECK(this->fixed, "network should be fixed");
        auto filter = this->fp_filters[idx];
        filter->forward();
    }
    
    template<typename T>
    void OrderedNet<T>::backward(int idx) {
        CHECK(this->fixed, "network should be fixed");
        auto filter = this->fp_filters[idx];
        filter->forward();
    }
    
    template class OrderedNet<float>;
    template class OrderedNet<double>;
    
}