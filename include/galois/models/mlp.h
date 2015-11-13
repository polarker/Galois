#ifndef _GALOIS_MLPMODEL_H_
#define _GALOIS_MLPMODEL_H_

#include "galois/base.h"
#include "galois/narray.h"
#include "galois/gfilters/path.h"
#include "galois/optimizer.h"
#include <vector>

using namespace std;

namespace gs
{
    
    template<typename T>
    class MLPModel
    {
        static default_random_engine galois_rn_generator;
        
    protected:
        Path<T> path;
        SP_Signal<T> input_signal = nullptr;
        SP_Signal<T> output_signal = nullptr;
        
        vector<SP_PFilter<T>> pfilters = {};
        vector<SP_NArray<T>>  params = {};
        vector<SP_NArray<T>>  grads = {};
        
        int batch_size;
        int num_epoch;
        T learning_rate;
        SP_Optimizer<T> optimizer;
        
        int train_count = 0;
        SP_NArray<T> train_data = nullptr;
        SP_NArray<T> train_target = nullptr;
        int test_count = 0;
        SP_NArray<T> test_data = nullptr;
        SP_NArray<T> test_target = nullptr;
        
    public:
        MLPModel(int batch_size, int num_epoch, T learning_rate, string optimizer_name);
        MLPModel(const MLPModel& other) = delete;
        MLPModel& operator=(const MLPModel&) = delete;
        
        void add_filter(SP_Filter<T>);
        void compile();
        
        vector<SP_NArray<T>> get_params() {
            return params;
        }
        vector<SP_NArray<T>> get_grads() {
            return grads;
        }
        
        void add_train_dataset(SP_NArray<T> data, SP_NArray<T> target);
        void add_test_dataset(const SP_NArray<T> data, const SP_NArray<T> target);
        
        T train_one_batch(const bool update=true);
        double compute_correctness(SP_Signal<T>); // for most application, this one should be override
        double test();
        void fit(const bool run_test=false);
    };
    template<typename T>
    default_random_engine MLPModel<T>::galois_rn_generator(0);
    
}

#endif
