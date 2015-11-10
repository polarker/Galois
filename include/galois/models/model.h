#ifndef _GALOIS_MODEL_H_
#define _GALOIS_MODEL_H_

#include "galois/base.h"
#include "galois/narray.h"
#include "galois/net.h"
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
        
        int batch_size;
        int num_epoch;
        T learning_rate;
        SP_Optimizer<T> optimizer;
        
        int train_count = 0;
        vector<SP_NArray<T>> train_data = {};
        vector<SP_NArray<T>> train_target = {};
        
    public:
        Model(int batch_size, int num_epoch, T learning_rate, string optimizer_name);
        Model(const Model& other) = delete;
        Model& operator=(const Model&) = delete;
        
        void add_link(const vector<string>&, const vector<string>&, SP_Filter<T>);
        void add_link(const initializer_list<string>, const initializer_list<string>, SP_Filter<T>);
        void add_link(const initializer_list<string>, const string, SP_Filter<T>);
        void add_link(const string, const initializer_list<string>, SP_Filter<T>);
        void add_link(const string, const string, SP_Filter<T>);
        void set_input_ids(const string);
        void set_input_ids(const initializer_list<string>);
        void set_input_ids(const vector<string>);
        void set_output_ids(const string);
        void set_output_ids(const initializer_list<string>);
        void set_output_ids(const vector<string>);
        void compile();
        void add_train_dataset(const SP_NArray<T> data, const SP_NArray<T> target);
        void add_train_dataset(const initializer_list<SP_NArray<T>> data, const SP_NArray<T> target);
        void add_train_dataset(const SP_NArray<T> data, const initializer_list<SP_NArray<T>> target);
        void add_train_dataset(const initializer_list<SP_NArray<T>> data, const initializer_list<SP_NArray<T>> target);
        void add_train_dataset(const vector<SP_NArray<T>>& data, const vector<SP_NArray<T>>& target);
        T fit_one_batch(const bool update=true);
        void fit();
    };
    template<typename T>
    default_random_engine Model<T>::galois_rn_generator(0);
    
}

#endif
