#ifndef _GALOIS_SEQENCODERDECODER_H
#define _GALOIS_SEQENCODERDECODER_H

#include "galois/base.h"
#include "galois/narray.h"
#include "galois/gfilters/net.h"
#include "galois/models/ordered_model.h"
#include "galois/optimizer.h"

namespace gs
{
    
    template<typename T>
    class SeqEncoderDecoder : protected OrderedModel<T>
    {
        static default_random_engine galois_rn_generator;
        
    protected:
        int max_len_encoder;
        int max_len_decoder;
        int input_size;
        int output_size;
        vector<int> hidden_sizes;

        int train_seq_count = 0;
        SP_NArray<T> train_X = nullptr;
        SP_NArray<T> train_Y = nullptr;
        int test_seq_count = 0;
        SP_NArray<T> test_X = nullptr;
        SP_NArray<T> test_Y = nullptr;
    public:
        SeqEncoderDecoder(int max_len_encoder,
            int max_len_decoder,
            int input_size,
            int output_size,
            initializer_list<int> hidden_sizes,
            int batch_size,
            int num_epoch,
            T learning_rate,
            string optimizer_name);
        SeqEncoderDecoder(const SeqEncoderDecoder& other) = delete;
        SeqEncoderDecoder& operator=(const SeqEncoderDecoder&) = delete;
        
        using OrderedModel<T>::get_params;
        using OrderedModel<T>::get_grads;
        
        void add_train_dataset(const SP_NArray<T> data, const SP_NArray<T> target);
        void add_test_dataset(const SP_NArray<T> data, const SP_NArray<T> target);
        T train_one_batch(const bool update=true);
        void fit();
    };
    template<typename T>
    default_random_engine SeqEncoderDecoder<T>::galois_rn_generator(0);

}

#endif
