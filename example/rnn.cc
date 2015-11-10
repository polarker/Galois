#include "galois/models.h"

using namespace std;
using namespace gs;

int main()
{
    using T = double;

    int seq_length = 3;
    int input_size = 100;
    int output_size = 100;
    auto hidden_sizes = {1000, 1000};
    int batch_size = 100;
    int num_epoch = 10;
    T learning_rate = 0.01;
    RNN<T> model(seq_length, input_size, output_size, hidden_sizes, batch_size, num_epoch, learning_rate, "sgd");
}