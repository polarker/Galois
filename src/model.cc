#include "galois/model.h"
#include "prettyprint.hpp"

namespace gs
{
    
    template<typename T>
    Model<T>::Model(int batch_size, T learning_rate, string optimizer_name)
    : net()
    , batch_size(batch_size)
    , learning_rate(learning_rate) {
        if (optimizer_name == "sgd") {
            optimizer = make_shared<SGD_Optimizer<T>>(learning_rate);
        } else {
            throw(optimizer_name + " is not implemented");
        }
    }
    
    template<typename T>
    void Model<T>::add_link(const initializer_list<string> ins, const string outs, SP_Filter<T> filter){
        add_link(ins, {outs}, filter);
    }
    
    template<typename T>
    void Model<T>::add_link(const string ins, const initializer_list<string> outs, SP_Filter<T> filter){
        add_link({ins}, outs, filter);
    }
    
    template<typename T>
    void Model<T>::add_link(const string ins, const string outs, SP_Filter<T> filter){
        add_link({ins}, {outs}, filter);
    }

    template<typename T>
    void Model<T>::add_link(const initializer_list<string> ins, const initializer_list<string> outs, SP_Filter<T> filter){
        net.add_link(ins, outs, filter);
    }

    template<typename T>
    void Model<T>::set_input_ids(const string id) {
        set_input_ids({id});
    }
    
    template<typename T>
    void Model<T>::set_input_ids(const initializer_list<string> ids) {
        set_input_ids(vector<string>(ids));
    }
    
    template<typename T>
    void Model<T>::set_input_ids(const vector<string> ids) {
        assert(input_ids.empty());
        net.set_input_ids(ids);
        input_ids = ids;
        for (int i = 0; i < input_ids.size(); i++) {
            input_signals.push_back(make_shared<Signal<T>>(InputSignal));
        }
    }

    template<typename T>
    void Model<T>::set_output_ids(const string id) {
        set_output_ids({id});
    }
    
    template<typename T>
    void Model<T>::set_output_ids(const initializer_list<string> ids) {
        set_output_ids(vector<string>(ids));
    }
    
    template<typename T>
    void Model<T>::set_output_ids(const vector<string> ids) {
        assert(output_ids.empty());
        net.set_output_ids(ids);
        output_ids = ids;
        for (int j = 0; j < output_ids.size(); j++) {
            output_signals.push_back(make_shared<Signal<T>>(OutputSignal));
        }
    }

    template<typename T>
    void Model<T>::compile() {
        for (auto pfilter : net.get_pfilters()) {
            pfilters.push_back(pfilter);
        }
        assert(!pfilters.empty());
        optimizer->compile(pfilters);
        
        net.set_p_order();
        
        assert(!input_ids.empty());
        assert(!output_ids.empty());
        net.install_signals(input_signals, output_signals);
        net.set_dims(batch_size);
    }
    
    template<typename T>
    void Model<T>::fit() {
        net.reopaque();
        for (auto input_signal : input_signals) {
            input_signal->reopaque();
            input_signal->get_data()->fill(1.0);
        }
        for (auto output_signal : output_signals) {
            output_signal->reopaque();
            output_signal->get_target()->fill(1.0);
        }
        net.forward(input_signals, output_signals);
        net.backward(input_signals, output_signals);
        
        cout << "loss:\t" << *output_signals[0]->get_loss() << endl;
        optimizer->update();
    }

    template class Model<float>;
    template class Model<double>;

}
