#ifndef _GALOIS_BISEQENCODERDECODER_H
#define _GALOIS_BISEQENCODERDECODER_H

#include "galois/base.h"
#include "galois/narray.h"
#include "galois/gfilters/net.h"
#include "galois/models/ordered_model.h"
#include "galois/optimizer.h"

namespace gs
{

    template<typename T>
    class BiSeqEncoderDecoder : protected OrderedModel<T>
    {
        static default_random_engine galois_rn_generator;

    protected:
        size_t max_len_one;
        size_t max_len_another;
        size_t input_size_one;
        size_t input_size_another;
        vector<size_t> hidden_sizes;

        size_t train_seq_count = 0;
        SP_NArray<T> train_one = nullptr;
        SP_NArray<T> train_another = nullptr;
    public:
        BiSeqEncoderDecoder(
            size_t max_len_one,
            size_t max_len_another,
            size_t input_size_one,
            size_t input_size_another,
            initializer_list<size_t> hidden_sizes,
            size_t batch_size,
            int num_epoch,
            T learning_rate,
            string optimizer_name);
        BiSeqEncoderDecoder(const BiSeqEncoderDecoder& other) = delete;
        BiSeqEncoderDecoder& operator=(const BiSeqEncoderDecoder&) = delete;

        using OrderedModel<T>::get_params;
        using OrderedModel<T>::get_grads;

        void add_train_dataset(const SP_NArray<T> data, const SP_NArray<T> target);
        T train_one_batch(const bool update=true);
        void fit();
    };
    template<typename T>
    default_random_engine BiSeqEncoderDecoder<T>::galois_rn_generator(0);

}

#endif
