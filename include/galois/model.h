#include "galois/base.h"
#include "galois/net.h"
#include "optimizers/optimizer.h"

namespace gs
{
    
    template<typename T>
    class Model
    {
    public:
        Net<T> net;
        vector<SP_PFilter<T>> pfilters;
        vector<string> input_ids = {};
        vector<SP_Signal<T>> input_signals = {};
        vector<string> output_ids = {};
        vector<SP_Signal<T>> output_signals = {};
        
        int batch_size;
        T learning_rate;
        SP_Optimizer<T> optimizer;
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
        void fit();
    };
    
}
