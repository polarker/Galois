#include "galois/base.h"
#include "galois/narray.h"
#include "galois/net.h"
#include "optimizers/optimizer.h"

namespace gs
{
    
    template<typename T>
    class Model
    {
    public:
        static default_random_engine galois_rn_generator;
        
        Net<T> net;
        vector<SP_PFilter<T>> pfilters;
        vector<string> input_ids = {};
        vector<SP_Signal<T>> input_signals = {};
        vector<string> output_ids = {};
        vector<SP_Signal<T>> output_signals = {};
        
        int batch_size;
        T learning_rate;
        SP_Optimizer<T> optimizer;
        
        int          train_count = 0;
        vector<SP_NArray<T>> train_data = {};
        vector<SP_NArray<T>> train_target = {};
    public:
        Model(int batch_size, T learning_rate, string optimizer_name);
        
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
        void add_train_dataset(SP_NArray<T> data, SP_NArray<T> target);
        void fit();
    };
    template<typename T>
    default_random_engine Model<T>::galois_rn_generator(0);
    
}
