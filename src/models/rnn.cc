#include "galois/models/rnn.h"
#include "galois/path.h"
#include "galois/filters.h"

namespace gs
{

    template<typename T>
    SP_Filter<T> linear_tanh(int in_size, int out_size) {
        auto p = make_shared<Path<T>>();
        p->add_filter(Linear<T>(in_size, out_size));
        p->add_filter(Tanh<T>());
        return p;
    }

    template<typename T>
    SP_Filter<T> linear_entropy(int in_size, int out_size) {
        auto p = make_shared<Path<T>>();
        p->add_filter(Linear<T>(in_size, out_size));
        p->add_filter(CrossEntropy<T>());
        return p;
    }

    string generate_id(string tag, int i) {
        return tag + "[" + to_string(i) + "]";
    }

    string generate_id(string tag, int i, int j) {
        return tag + "[" + to_string(i) + "," + to_string(j) + "]";
    }

    template<typename T>
    RNN<T>::RNN(int _seq_length,
                int _input_size,
                int _output_size,
                initializer_list<int> _hidden_sizes,
                int _batch_size,
                int _num_epoch,
                T _learning_rate,
                string optimizer_name)
            : seq_length(_seq_length)
            , input_size(_input_size)
            , output_size(_output_size)
            , hidden_sizes(_hidden_sizes) {
        this->batch_size = _batch_size;
        this->num_epoch = _num_epoch;
        this->learning_rate = _learning_rate;
        if (optimizer_name == "sgd") {
            this->optimizer = make_shared<SGD_Optimizer<T>>(this->learning_rate);
        } else {
            throw(optimizer_name + " is not implemented");
        }

        auto h2h = vector<SP_Filter<T>>();
        for (auto hsize : hidden_sizes) {
            h2h.push_back(linear_tanh<T>(hsize, hsize));
        }
        auto x2h = vector<SP_Filter<T>>();
        for (int i = 0; i < hidden_sizes.size(); i++) {
            if (i == 0) {
                x2h.push_back(linear_tanh<T>(input_size, hidden_sizes[i]));
            } else {
                x2h.push_back(linear_tanh<T>(hidden_sizes[i-1], hidden_sizes[i]));
            }
        }
        auto h2y = linear_entropy<T>(hidden_sizes[hidden_sizes.size()-1], output_size);
        for (int i = 0; i < seq_length; i++) {
            for (int j = 0; j < hidden_sizes.size(); j++) {
                string h = generate_id("h", i, j);
                string left_h = generate_id("h", i-1, j);
                string down_h;
                if (j == 0) {
                    down_h = generate_id("x", i);
                } else {
                    down_h = generate_id("h", i, j-1);
                }
                if (i > 0) {
                    add_link(left_h, h, h2h[j].share());
                }
                add_link(down_h, h, x2h[j].share());
            }
            string y = generate_id("y", i);
            string down_h = generate_id("h", i, hidden_sizes.size()-1);
            add_link(down_h, y, h2y.share());
        }

        auto x_ids = vector<string>();
        auto y_ids = vector<string>();
        for (int i = 0; i < seq_length; i++) {
            x_ids.push_back(generate_id("x", i));
            y_ids.push_back(generate_id("y", i));
        }
        this->set_input_ids(x_ids);
        this->set_output_ids(y_ids);

        this->compile();
    }

}