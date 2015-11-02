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
        int seq_length;
        int input_size;
        int output_size;
        vector<int> hidden_sizes;

    public:
        RNN(int seq_length,
            int input_size,
            int output_size,
            initializer_list<int> hidden_sizes,
            int batch_size,
            int num_epoch,
            T learning_rate,
            string optimizer_name);
        RNN(const RNN& other) = delete;
        RNN& operator=(const RNN&) = delete;
    };

}

#endif
