#include <string>
#include "galois/model.h"
#include "galois/filters.h"
#include "galois/dataset/mnist.h"

using namespace std;
using namespace gs;

int main()
{
    using T = double;

    int batch_size = 100;
    int num_epoch = 10;
    T learning_rate = 0.01;
    Model<T> model(batch_size, num_epoch, learning_rate, "sgd");

    model.add_link("images", "raw_h1", make_shared<Linear<T>>(28*28, 1024));
    model.add_link("raw_h1", "h1", make_shared<Tanh<T>>());
    model.add_link("h1", "raw_h2", make_shared<Linear<T>>(1024, 10));
    model.add_link("raw_h2", "predicitons", make_shared<CrossEntropy<T>>());
    model.set_input_ids("images");
    model.set_output_ids("predicitons");
    model.compile();

    auto images = mnist::read_images<T>("../data/train-images-idx3-ubyte.gz");
    auto labels = mnist::read_labels<T>("../data/train-labels-idx1-ubyte.gz");
    model.add_train_dataset(images, labels);

    model.fit();
}

