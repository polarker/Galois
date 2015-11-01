#include "galois/mlp_model.h"

using namespace std;

namespace gs
{
    
    template<typename T>
    MLPModel<T>::MLPModel(int batch_size, int num_epoch, T learning_rate, string optimizer_name)
               : path()
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
    void MLPModel<T>::add_filter(SP_Filter<T> filter) {
        path.add_filter(filter);
    }
    
    template<typename T>
    void MLPModel<T>::compile() {
        for (auto pfilter : path.get_pfilters()) {
            pfilters.push_back(pfilter);
        }
        optimizer->compile(pfilters);
        
        CHECK(input_signal == nullptr && output_signal == nullptr, "these signals should not be set before");
        input_signal = make_shared<Signal<T>>(InputSignal);
        output_signal = make_shared<Signal<T>>(OutputSignal);
        
        path.install_signals(vector<SP_Signal<T>>{input_signal}, vector<SP_Signal<T>>{output_signal});
        path.set_dims(batch_size);
    }
    
    template<typename T>
    void MLPModel<T>::add_train_dataset(SP_NArray<T> data, SP_NArray<T> target) {
        CHECK(data->get_dims()[0] == target->get_dims()[0], "data and target should have the same number of samples");
        
        train_count = data->get_dims()[0];
        train_data = data;
        train_target = target;
    }
    
    template<typename T>
    T MLPModel<T>::fit_one_batch(const bool update) {
        uniform_int_distribution<> distribution(0, train_count-1);
        vector<int> batch_ids(batch_size);
        for (int i = 0; i < batch_size; i++) {
            batch_ids[i] = distribution(galois_rn_generator);
        }
        
        path.reopaque();
        input_signal->reopaque();
        input_signal->get_data()->copy_from(batch_ids, train_data);
        output_signal->reopaque();
        output_signal->get_target()->copy_from(batch_ids, train_target);
        
        path.forward();
        path.backward();
        if (update) {
            optimizer->update();
        }
        
        T loss = 0;
        loss += *output_signal->get_loss();
        
        return loss;
    }
    
    template<typename T>
    void MLPModel<T>::fit() {
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
    
}