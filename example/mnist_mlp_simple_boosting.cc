#include "galois/models.h"
#include "galois/filters.h"
#include "galois/dataset/mnist.h"

using namespace std;
using namespace gs;

int main()
{
    using T = float;

    auto path1 = make_shared<Path<T>>();
    path1->add_filter(make_shared<Linear<T>>(28*28, 512));
    path1->add_filter(make_shared<Tanh<T>>());
    path1->add_filter(make_shared<Linear<T>>(512, 10));

    int batch_size = 10;
    int num_epoch = 5;
    T learning_rate = 0.05;
    auto train_images = mnist::read_images<T>("./data/train-images-idx3-ubyte.gz");
    auto train_labels = mnist::read_labels<T>("./data/train-labels-idx1-ubyte.gz");
    auto test_images = mnist::read_images<T>("./data/t10k-images-idx3-ubyte.gz");
    auto test_labels = mnist::read_labels<T>("./data/t10k-labels-idx1-ubyte.gz");

    Model<T> model1(batch_size, num_epoch, learning_rate, "sgd");
    model1.add_link("images", "output", path1);
    model1.add_link("output", "predicitons", make_shared<CrossEntropy<T>>());
    model1.add_input_ids("images");
    model1.add_output_ids("predicitons");
    model1.add_train_dataset(train_images, train_labels);
    model1.add_test_dataset(test_images, test_labels);
    model1.fit();

    auto path2 = path1->clone();
    path1->fix_params();
    Model<T> model2(batch_size, 10, 0.01, "sgd");
    model2.add_link("images", "output", path1);
    model2.add_link("images", "output", path2);
    model2.add_link("output", "predicitons", make_shared<CrossEntropy<T>>());
    model2.add_input_ids("images");
    model2.add_output_ids("predicitons");
    model2.add_train_dataset(train_images, train_labels);
    model2.add_test_dataset(test_images, test_labels);
    model2.fit();
}
