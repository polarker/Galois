#include "galois/models.h"
#include "galois/dataset/chartxt.h"
#include <cstdlib>
#include <cassert>
#include <cmath>

using namespace std;
using namespace gs;

int main()
{
    using T = double;

    chartxt::Article<T> article("./data/tinyshakespeare.txt");

    size_t seq_length = 3;
    size_t input_size = article.get_num_diff_chars();
    size_t output_size = article.get_num_diff_chars();
    initializer_list<size_t> hidden_sizes {128, 128};
    size_t batch_size = 100;
    int num_epoch = 10;
    T learning_rate = 0.01;
    bool use_embedding = true;
    RNN<T> model(seq_length, input_size, output_size, hidden_sizes, batch_size, num_epoch, learning_rate, "sgd", use_embedding);

    auto params = model.get_params();
    auto grads = model.get_grads();

    model.add_train_dataset(article.get_input_sequence(), article.get_target_sequence());

    srand(time(NULL));
    for (int k = 0; k < 10; k++) {
        int idx;
        idx = rand() % params.size();
        auto p = params[idx];
        auto dp = grads[idx];

        idx = rand() % p->get_size();
        int start_idx = 0;

        auto old_pi = p->get_data()[idx];
        T delta = 1e-5;
        model.train_one_batch(start_idx, false);
        auto grad = dp->get_data()[idx];

        p->get_data()[idx] = old_pi + delta;
        auto loss1 = model.train_one_batch(start_idx, false);

        p->get_data()[idx] = old_pi - delta;
        auto loss2 = model.train_one_batch(start_idx, false);

        auto grad_ = (loss1-loss2) / (2*delta);
        auto diff = abs(grad - grad_);
        assert(diff < delta);
        printf("%dth gradient check passed\n", k);
    }
}
