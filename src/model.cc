#include "galois/model.h"

namespace gs
{
    
    template<typename T>
    Model<T>::Model(int batch_size, int num_epoch, T learning_rate, string optimizer_name)
            : net()
            , batch_size(batch_size)
            , num_epoch(num_epoch)
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
        CHECK(input_ids.empty(), "input_ids should not be set before");
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
        CHECK(output_ids.empty(), "output_ids should not be set before");
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
        optimizer->compile(pfilters);
        
        CHECK(!input_ids.empty(), "input_ids should have been set");
        CHECK(!output_ids.empty(), "output_ids should have been set");
        net.install_signals(input_signals, output_signals);
        
        net.set_p_order();
        net.set_dims(batch_size);
        // todo: check the dimension of dataset
    }
    
    template<typename T>
    void Model<T>::add_train_dataset(SP_NArray<T> data, SP_NArray<T> target) {
        CHECK(input_ids.size() == 1, "there should be only one input signal");
        CHECK(output_ids.size() == 1, "there should be only one output signal");
        CHECK(data->get_dims()[0] == target->get_dims()[0], "data and target should have the same number of samples");
        
        train_count = data->get_dims()[0];
        train_data = vector<SP_NArray<T>>{data};
        train_target = vector<SP_NArray<T>>{target};
    }
    
    template<typename T>
    T Model<T>::fit_one_batch(const bool update) {
        uniform_int_distribution<> distribution(0, train_count-1);
        vector<int> batch_ids(batch_size);
        for (int i = 0; i < batch_size; i++) {
            batch_ids[i] = distribution(galois_rn_generator);
        }
        
        net.reopaque();
        for (int i = 0; i < input_signals.size(); i++) {
            input_signals[i]->reopaque();
            input_signals[i]->get_data()->copy_from(batch_ids, train_data[i]);
        }
        for (int i = 0; i < output_signals.size(); i++) {
            output_signals[i]->reopaque();
            output_signals[i]->get_target()->copy_from(batch_ids, train_target[i]);
        }
        
        net.forward();
        net.backward();
        if (update) {
            optimizer->update();
        }
        
        T loss = 0;
        for (auto output_signal : output_signals) {
            loss += *output_signal->get_loss();
        }
        return loss;
    }
    
    template<typename T>
    void Model<T>::fit() {
        for (int k = 1; k < num_epoch+1; k++) {
            printf("Epoch: %2d", k);
            auto start = chrono::system_clock::now();
            T loss = 0;
            for (int i = 0; i < train_count/batch_size; i++) {
                loss += fit_one_batch();
            }
            loss /= T(train_count/batch_size);
            
            auto end = chrono::system_clock::now();
            chrono::duration<double> eplased_time = end - start;
            printf(", time: %.2fs", eplased_time.count());
            printf(", loss: %.6f", loss);
            printf("\n");
        }
    }

    template class Model<float>;
    template class Model<double>;

}
