#include "galois/models.h"
#include "galois/filters.h"
#include "galois/dataset/mnist.h"
#include <cstdlib>
#include <cmath>

using namespace std;
using namespace gs;

int main()
{
    using T = double;

    int batch_size = 1;
    int num_epoch = 1;
    T learning_rate = 0.01;
    MLPModel<T> model(batch_size, num_epoch, learning_rate, "sgd");

    auto l1 = make_shared<Convolution<T>>(28, 28, 1, 20, 5, 5);
    auto lm = make_shared<MaxPooling<T>>(2, 2, 2, 2);
    auto l2 = make_shared<Linear<T>>(20*12*12, 10);

    model.add_filter(l1);
    model.add_filter(make_shared<Tanh<T>>());
    model.add_filter(lm);
    model.add_filter(l2);
    model.add_filter(make_shared<CrossEntropy<T>>());
    model.compile();

    auto images = mnist::read_images<T>("./data/train-images-idx3-ubyte.gz", 1);
    auto labels = mnist::read_labels<T>("./data/train-labels-idx1-ubyte.gz", 1);
    model.add_train_dataset(images, labels);

    auto params = model.get_params();
    auto grads = model.get_grads();

    srand(time(NULL));
    for (int k = 0; k < 100; k++) {
        int idx;
        idx = rand() % params.size();
        auto p = params[idx];
        auto dp = grads[idx];

        idx = rand() % p->get_size();

        auto old_pi = p->get_data()[idx];
        T delta = 1e-6;
        model.train_one_batch(false);
        auto grad = dp->get_data()[idx];

        p->get_data()[idx] = old_pi + delta;
        auto loss1 = model.train_one_batch(false);

        p->get_data()[idx] = old_pi - delta;
        auto loss2 = model.train_one_batch(false);

        auto grad_ = (loss1-loss2) / (2*delta);
        auto diff = abs(grad - grad_);
        assert(diff < delta);
        printf("%dth gradient check passed\n", k);
    }

    return 0;
}
