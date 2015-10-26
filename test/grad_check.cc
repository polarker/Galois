#include "galois/model.h"
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
    Model<T> model(batch_size, num_epoch, learning_rate, "sgd");

    auto l1 = make_shared<Linear<T>>(28*28, 1024);
    auto l2 = make_shared<Linear<T>>(1024, 10);

    model.add_link("images", "raw_h1", l1);
    model.add_link("raw_h1", "h1", make_shared<Tanh<T>>());
    model.add_link("h1", "raw_h2", l2);
    model.add_link("raw_h2", "predicitons", make_shared<CrossEntropy<T>>());
    model.set_input_ids("images");
    model.set_output_ids("predicitons");
    model.compile();

    auto images = mnist::read_images<T>("./data/train-images-idx3-ubyte.gz", 1);
    auto labels = mnist::read_labels<T>("./data/train-labels-idx1-ubyte.gz", 1);
    model.add_train_dataset(images, labels);

    srand(time(NULL));
    for (int k = 0; k < 10; k++) {
        auto l = l1;
        if (rand() % 2 == 1) l = l2;
        auto p = l->w;
        auto dp = l->dw;
        if (rand() % 2 == 1) { p = l->b; dp = l->db; }
        int idx = rand() % p->get_size();

        auto old_pi = p->get_data()[idx];
        T delta = 1e-5;
        model.fit_one_batch(false);
        auto grad = dp->get_data()[idx];

        p->get_data()[idx] = old_pi + delta;
        auto loss1 = model.fit_one_batch(false);

        p->get_data()[idx] = old_pi - delta;
        auto loss2 = model.fit_one_batch(false);

        auto grad_ = (loss1-loss2) / (2*delta);
        auto diff = abs(grad - grad_);
        assert(diff < delta);
        printf("%dth gradient check passed\n", k);
    }

    return 0;
}
