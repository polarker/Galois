#ifndef _GALOIS_RNN_H
#define _GALOIS_RNN_H

#include "galois/base.h"
#include "galois/narray.h"
#include "galois/net.h"
#include "galois/models/model.h"
#include "galois/optimizer.h"

namespace gs
{
    
    template<typename T>
    class RNN : protected Model<T>
    {
    protected:
        int max_len; // length of rnn
        int input_size;
        int output_size;
        vector<int> hidden_sizes;

        int seq_len = 0; // length of dataset
        SP_NArray<T> X = nullptr;
        SP_NArray<T> Y = nullptr;
    public:
        RNN(int max_len,
            int input_size,
            int output_size,
            initializer_list<int> hidden_sizes,
            int batch_size,
            int num_epoch,
            T learning_rate,
            string optimizer_name);
        RNN(const RNN& other) = delete;
        RNN& operator=(const RNN&) = delete;
        
        void add_train_dataset(const SP_NArray<T> data, const SP_NArray<T> target);
        T fit_one_batch(const int start_from, const bool update=true);
        void fit();
    };

}

#endif
