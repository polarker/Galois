#ifndef _GALOIS_MODEL_H_
#define _GALOIS_MODEL_H_

#include "galois/base.h"
#include "galois/narray.h"
#include "galois/gfilters/net.h"
#include "galois/optimizer.h"

namespace gs
{

    template<typename T>
    class Model
    {
        static default_random_engine galois_rn_generator;

    protected:
        Net<T> net;

        vector<SP_PFilter<T>> pfilters = {};
        vector<SP_NArray<T>>  params = {};
        vector<SP_NArray<T>>  grads = {};

        vector<string>          input_ids = {};
        vector<SP_Signal<T>>    input_signals = {};
        vector<string>          output_ids = {};
        vector<SP_Signal<T>>    output_signals = {};

        size_t batch_size;
        int num_epoch;
        T learning_rate;
        SP_Optimizer<T> optimizer;

        size_t train_count = 0;
        vector<SP_NArray<T>> train_data = {};
        vector<SP_NArray<T>> train_target = {};
        size_t test_count = 0;
        vector<SP_NArray<T>> test_data = {};
        vector<SP_NArray<T>> test_target = {};

    public:
        Model(size_t batch_size, int num_epoch, T learning_rate, string optimizer_name);
        Model(const Model& other) = delete;
        Model& operator=(const Model&) = delete;

        void add_link(const vector<string>&, const vector<string>&, SP_Filter<T>);
        void add_link(const initializer_list<string>, const initializer_list<string>, SP_Filter<T>);
        void add_link(const initializer_list<string>, const string, SP_Filter<T>);
        void add_link(const string, const initializer_list<string>, SP_Filter<T>);
        void add_link(const string, const string, SP_Filter<T>);
        void add_input_ids(const string);
        void add_input_ids(const initializer_list<string>);
        void add_input_ids(const vector<string>);
        void add_output_ids(const string);
        void add_output_ids(const initializer_list<string>);
        void add_output_ids(const vector<string>);
        void compile();

        vector<SP_NArray<T>> get_params() {
            return params;
        }
        vector<SP_NArray<T>> get_grads() {
            return grads;
        }

        void add_train_dataset(const SP_NArray<T> data, const SP_NArray<T> target);
        void add_train_dataset(const initializer_list<SP_NArray<T>> data, const SP_NArray<T> target);
        void add_train_dataset(const SP_NArray<T> data, const initializer_list<SP_NArray<T>> target);
        void add_train_dataset(const initializer_list<SP_NArray<T>> data, const initializer_list<SP_NArray<T>> target);
        void add_train_dataset(const vector<SP_NArray<T>>& data, const vector<SP_NArray<T>>& target);
        void add_test_dataset(const SP_NArray<T> data, const SP_NArray<T> target);
        void add_test_dataset(const initializer_list<SP_NArray<T>> data, const SP_NArray<T> target);
        void add_test_dataset(const SP_NArray<T> data, const initializer_list<SP_NArray<T>> target);
        void add_test_dataset(const initializer_list<SP_NArray<T>> data, const initializer_list<SP_NArray<T>> target);
        void add_test_dataset(const vector<SP_NArray<T>>& data, const vector<SP_NArray<T>>& target);

        void fix_params() { net.fix_params(); }

        T train_one_batch(const bool update=true);
        double compute_correctness(SP_Signal<T>); // for most application, this one should be override
        double test();
        void fit();
    };
    template<typename T>
    default_random_engine Model<T>::galois_rn_generator(0);

}

#endif
