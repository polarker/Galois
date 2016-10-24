#include "galois/models/rnn.h"
#include "galois/gfilters/path.h"
#include "galois/filters.h"

namespace gs
{

    string generate_id(string tag, int i) {
        return tag + "[" + to_string(i) + "]";
    }

    string generate_id(string tag, int i, int j) {
        return tag + "[" + to_string(i) + "," + to_string(j) + "]";
    }

    template<typename T>
    RNN<T>::RNN(int _max_len,
                int _input_size,
                int _output_size,
                initializer_list<int> _hidden_sizes,
                int _batch_size,
                int _num_epoch,
                T _learning_rate,
                string _optimizer_name,
                bool _use_embedding)
            : Model<T>(_batch_size, _num_epoch, _learning_rate, _optimizer_name)
            , max_len(_max_len)
            , input_size(_input_size)
            , output_size(_output_size)
            , hidden_sizes(_hidden_sizes)
            , use_embedding(_use_embedding) {
        auto h2hraw = vector<SP_Filter<T>>();
        for (auto hsize : hidden_sizes) {
            h2hraw.push_back(make_shared<Linear<T>>(hsize, hsize));
        }
        auto x2hraw = vector<SP_Filter<T>>();
        for (int i = 0; i < hidden_sizes.size(); i++) {
            if (i == 0) {
                if (use_embedding) {
                    x2hraw.push_back(make_shared<Embedding<T>>(input_size, hidden_sizes[i]));
                } else {
                    x2hraw.push_back(make_shared<Linear<T>>(input_size, hidden_sizes[i]));
                }
            } else {
                x2hraw.push_back(make_shared<Linear<T>>(hidden_sizes[i-1], hidden_sizes[i]));
            }
        }
        auto h2yraw = make_shared<Linear<T>>(hidden_sizes.back(), output_size);
        for (int i = 0; i < max_len; i++) {
            for (int j = 0; j < hidden_sizes.size(); j++) {
                string hraw = generate_id("hraw", i, j);
                string left_h = generate_id("h", i-1, j);
                string down_h;
                if (j == 0) {
                    down_h = generate_id("x", i);
                } else {
                    down_h = generate_id("h", i, j-1);
                }
                if (i > 0) {
                    this->add_link(left_h, hraw, h2hraw[j]->share());
                }
                this->add_link(down_h, hraw, x2hraw[j]->share());
                string h = generate_id("h", i, j);
                this->add_link(hraw, h, make_shared<Tanh<T>>());
            }
            string yraw = generate_id("yraw", i);
            string down_h = generate_id("h", i, hidden_sizes.size()-1);
            this->add_link(down_h, yraw, h2yraw->share());
            string y = generate_id("y", i);
            this->add_link(yraw, y, make_shared<CrossEntropy<T>>());
        }

        auto x_ids = vector<string>();
        auto y_ids = vector<string>();
        for (int i = 0; i < max_len; i++) {
            x_ids.push_back(generate_id("x", i));
            y_ids.push_back(generate_id("y", i));
        }
        this->add_input_ids(x_ids);
        this->add_output_ids(y_ids);

        this->compile();
    }

    template<typename T>
    void RNN<T>::add_train_dataset(const SP_NArray<T> data, const SP_NArray<T> target) {
        auto data_dims = data->get_dims();
        auto target_dims = target->get_dims();
        CHECK(data_dims[0] == target_dims[0], "length of data and target must match");

        CHECK(train_X==nullptr && train_Y==nullptr, "dataset should not be set before");
        train_seq_len = data_dims[0];
        train_X = data;
        train_Y = target;
    }

    template<typename T>
    void RNN<T>::add_test_dataset(const SP_NArray<T> data, const SP_NArray<T> target) {
        auto data_dims = data->get_dims();
        auto target_dims = target->get_dims();
        CHECK(data_dims[0] == target_dims[0], "length of data and target must match");
        CHECK(data_dims.size() == 2 && target_dims.size() == 1 && data_dims[1] == input_size, "sizes must match");

        CHECK(test_X==nullptr && test_Y==nullptr, "dataset should not be set before");
        test_seq_len = data_dims[0];
        test_X = data;
        test_Y = target;
    }

    template<typename T>
    T RNN<T>::train_one_batch(const int start_from, bool update) {
        this->net.reopaque();
        for (int i = 0; i < this->input_signals.size(); i++) {
            this->input_signals[i]->reopaque();
            this->input_signals[i]->get_data()->copy_from(start_from+i, this->batch_size, train_X);
        }
        for (int i = 0; i < this->output_signals.size(); i++) {
            this->output_signals[i]->reopaque();
            this->output_signals[i]->get_target()->copy_from(start_from+i, this->batch_size, train_Y);
        }

        this->net.forward();
        this->net.backward();
        if (update) {
            this->optimizer->update();
        }

        T loss = 0;
        for (auto output_signal : this->output_signals) {
            loss += *output_signal->get_loss();
        }
        return loss;
    }

    // test dataset is not support for the moment
    template<typename T>
    void RNN<T>::fit() {
        printf("Start training\n");

        for (int k = 1; k < this->num_epoch+1; k++) {
            printf("Epoch: %2d", k);
            auto start = chrono::system_clock::now();
            T loss = 0;

            int len = train_seq_len - max_len + 1 - this->batch_size + 1;
            for (int i = 0; i < len; i += this->batch_size) {
                loss += train_one_batch(i);
                if (i % 10000 == 0) {
                    cout << " > " << i << endl;
                }
            }
            loss /= T(len);

            auto end = chrono::system_clock::now();
            chrono::duration<double> eplased_time = end - start;
            printf(", time: %.2fs", eplased_time.count());
            printf(", loss: %.6f", loss);
            printf("\n");
        }
    }

    template class RNN<float>;
    template class RNN<double>;

}