#include "galois/models/mlp.h"

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
        CHECK(pfilters.empty() && params.empty() && grads.empty(), "these should not be set before");
        for (auto pfilter : path.get_pfilters()) {
            pfilters.push_back(pfilter);
        }
        for (auto pfilter : pfilters) {
            auto tmp_params = pfilter->get_params();
            auto tmp_grads = pfilter->get_grads();
            CHECK(tmp_params.size() == tmp_grads.size(), "numbers of params and grads should be equal");
            for (int i = 0; i < tmp_params.size(); i++) {
                auto param = tmp_params[i];
                auto grad = tmp_grads[i];
                CHECK(param->get_dims() == grad->get_dims(), "param and grad should have the same dimensions");
                
                // parameters might be shared by several pfilters
                if (Contains(params, param)) {
                    CHECK(Contains(grads, grad), "params and grads should match");
                } else {
                    CHECK(!Contains(grads, grad), "params and grads should match");
                    params.push_back(param);
                    grads.push_back(grad);
                }
            }
        }
        optimizer->compile(params, grads);
        
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
    void MLPModel<T>::add_test_dataset(SP_NArray<T> data, SP_NArray<T> target) {
        CHECK(data->get_dims()[0] == target->get_dims()[0], "data and target should have the same number of samples");
        
        test_count = data->get_dims()[0];
        test_data = data;
        test_target = target;
    }
    
    template<typename T>
    T MLPModel<T>::train_one_batch(const bool update) {
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
    double MLPModel<T>::compute_correctness(SP_Signal<T> output_signal) {
        return COUNT_EQUAL(output_signal->get_data(), output_signal->get_target());
    }

    template<typename T>
    double MLPModel<T>::test() {
        double correctness = 0;
        
        for (int i = 0; i < test_count; i += batch_size) {
            vector<int> batch_ids(batch_size);
            for (int j = 0; j < batch_size; j++) {
                batch_ids[j] = i+j;
            }
            
            path.reopaque();
            input_signal->reopaque();
            input_signal->get_data()->copy_from(batch_ids, test_data);
            output_signal->reopaque();
            output_signal->get_target()->copy_from(batch_ids, test_target);
            path.forward();
            
            correctness += compute_correctness(output_signal);
        }
        
        int test_num = test_count / batch_size * batch_size;
        return double(correctness) / double(test_num);
    }
    
    template<typename T>
    void MLPModel<T>::fit() {
        compile();
        CHECK(train_data != nullptr && train_target != nullptr, "training dataset should have been set");
        bool run_test = false;
        if (test_data != nullptr && test_target != nullptr) {
            run_test = true;
        }
        else if (test_data == nullptr && test_target == nullptr) {
            run_test = false;
        }
        else {
            CHECK(false, "testing dataset is not correctly set");
        }

        printf("Start training\n");
        
        for (int k = 1; k < num_epoch+1; k++) {
            printf("Epoch: %2d", k);
            auto start = chrono::system_clock::now();

            T loss = 0;
            for (int i = 0; i < train_count/batch_size; i++) {
                loss += train_one_batch();
            }
            loss /= T(train_count/batch_size);
            
            auto end = chrono::system_clock::now();
            chrono::duration<double> eplased_time = end - start;
            printf(", time: %.2fs", eplased_time.count());
            printf(", loss: %.6f", loss);
            if (run_test) {
                double accuracy;
                accuracy = test();
                printf(", accuracy: %.2f%%", accuracy*100);
            }
            printf("\n");
        }
    }

    template class MLPModel<float>;
    template class MLPModel<double>;
    
}
