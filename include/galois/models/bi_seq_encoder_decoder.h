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
        int max_len_one;
        int max_len_another;
        int input_size_one;
        int input_size_another;
        vector<int> hidden_sizes;

        int train_seq_count = 0;
        SP_NArray<T> train_one = nullptr;
        SP_NArray<T> train_another = nullptr;
    public:
        BiSeqEncoderDecoder(int max_len_one,
            int max_len_another,
            int input_size_one,
            int input_size_another,
            initializer_list<int> hidden_sizes,
            int batch_size,
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
