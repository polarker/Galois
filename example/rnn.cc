#include "galois/models.h"
#include "galois/dataset/chartxt.h"

using namespace std;
using namespace gs;

int main()
{
    using T = double;
    
    chartxt::Article<T> article("./data/tinyshakespeare.txt");

    int seq_length = 50;
    int input_size = article.get_num_diff_chars();
    int output_size = article.get_num_diff_chars();
    auto hidden_sizes = {128, 128};
    int batch_size = 100;
    int num_epoch = 10;
    T learning_rate = 0.01;
    RNN<T> model(seq_length, input_size, output_size, hidden_sizes, batch_size, num_epoch, learning_rate, "sgd");
    
    model.add_train_dataset(article.get_vectorized_sequence(), article.get_target_sequence());
    model.fit();
}
