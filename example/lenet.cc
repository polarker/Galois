#include "galois/models.h"
#include "galois/filters.h"
#include "galois/dataset/mnist.h"

using namespace std;
using namespace gs;

int main()
{
    using T = float;

    int batch_size = 64;
    int num_epoch = 20;
    T learning_rate = 0.1;
    MLPModel<T> model(batch_size, num_epoch, learning_rate, "sgd");

    model.add_filter(make_shared<Convolution<T>>(28, 28, 1, 20, 5, 5));
    model.add_filter(make_shared<Tanh<T>>());
    model.add_filter(make_shared<MaxPooling<T>>(2, 2, 2, 2));
    model.add_filter(make_shared<Convolution<T>>(12, 12, 20, 50, 5, 5));
    model.add_filter(make_shared<Tanh<T>>());
    model.add_filter(make_shared<MaxPooling<T>>(2, 2, 2, 2));
    model.add_filter(make_shared<Linear<T>>(50*4*4, 500));
    model.add_filter(make_shared<Tanh<T>>());
    model.add_filter(make_shared<Linear<T>>(500, 10));
    model.add_filter(make_shared<CrossEntropy<T>>());

    auto train_images = mnist::read_images<T>("./data/train-images-idx3-ubyte.gz");
    auto train_labels = mnist::read_labels<T>("./data/train-labels-idx1-ubyte.gz");
    model.add_train_dataset(train_images, train_labels);
    auto test_images = mnist::read_images<T>("./data/t10k-images-idx3-ubyte.gz");
    auto test_labels = mnist::read_labels<T>("./data/t10k-labels-idx1-ubyte.gz");
    model.add_test_dataset(test_images, test_labels);

    model.fit();
}
