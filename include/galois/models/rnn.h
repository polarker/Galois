#ifndef _GALOIS_RNN_H
#define _GALOIS_RNN_H

#include "galois/base.h"
#include "galois/narray.h"
#include "galois/gfilters/net.h"
#include "galois/models/model.h"
#include "galois/optimizer.h"

namespace gs
{

    template<typename T>
    class RNN : protected Model<T>
    {
    protected:
        size_t max_len; // length of rnn
        size_t input_size;
        size_t output_size;
        vector<size_t> hidden_sizes;

        bool use_embedding = false;

        size_t train_seq_len = 0;
        SP_NArray<T> train_X = nullptr;
        SP_NArray<T> train_Y = nullptr;
        size_t test_seq_len = 0;
        SP_NArray<T> test_X = nullptr;
        SP_NArray<T> test_Y = nullptr;
    public:
        RNN(size_t max_len,
            size_t input_size,
            size_t output_size,
            initializer_list<size_t> hidden_sizes,
            size_t batch_size,
            int num_epoch,
            T learning_rate,
            string optimizer_name,
            bool use_embedding=false);
        RNN(const RNN& other) = delete;
        RNN& operator=(const RNN&) = delete;

        using Model<T>::get_params;
        using Model<T>::get_grads;

        void add_train_dataset(const SP_NArray<T> data, const SP_NArray<T> target);
        void add_test_dataset(const SP_NArray<T> data, const SP_NArray<T> target);
        T train_one_batch(const int start_from, const bool update=true);
        void fit();
    };

}

#endif
