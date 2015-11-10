#include "galois/models/rnn.h"
#include "galois/path.h"
#include "galois/filters.h"

namespace gs
{

    template<typename T>
    SP_Filter<T> linear_tanh(int in_size, int out_size) {
        auto p = make_shared<Path<T>>();
        p->add_filter(make_shared<Linear<T>>(in_size, out_size));
        p->add_filter(make_shared<Tanh<T>>());
        return p;
    }

    template<typename T>
    SP_Filter<T> linear_entropy(int in_size, int out_size) {
        auto p = make_shared<Path<T>>();
        p->add_filter(make_shared<Linear<T>>(in_size, out_size));
        p->add_filter(make_shared<CrossEntropy<T>>());
        return p;
    }

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
                string optimizer_name)
            : Model<T>(_batch_size, _num_epoch, _learning_rate, optimizer_name)
            , max_len(_max_len)
            , input_size(_input_size)
            , output_size(_output_size)
            , hidden_sizes(_hidden_sizes) {
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
        for (int i = 0; i < max_len; i++) {
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
                    this->add_link(left_h, h, h2h[j]->share());
                }
                this->add_link(down_h, h, x2h[j]->share());
            }
            string y = generate_id("y", i);
            string down_h = generate_id("h", i, hidden_sizes.size()-1);
            this->add_link(down_h, y, h2y->share());
        }

        auto x_ids = vector<string>();
        auto y_ids = vector<string>();
        for (int i = 0; i < max_len; i++) {
            x_ids.push_back(generate_id("x", i));
            y_ids.push_back(generate_id("y", i));
        }
        this->set_input_ids(x_ids);
        this->set_output_ids(y_ids);

        this->compile();

        for (auto idx : this->net.fp_order) {
            auto t = this->net.links[idx];
            auto in_id = get<0>(t)[0];
            auto out_id = get<1>(t)[0];
            cout << in_id << " -> " << out_id << endl;
        }
        for (auto idx : this->net.bp_order) {
            auto t = this->net.links[idx];
            auto in_id = get<0>(t)[0];
            auto out_id = get<1>(t)[0];
            cout << in_id << " <- " << out_id << endl;
        }
    }
    
    template<typename T>
    void RNN<T>::add_train_dataset(const SP_NArray<T> data, const SP_NArray<T> target) {
        auto data_dims = data->get_dims();
        auto target_dims = target->get_dims();
        CHECK(data_dims[0] == target_dims[0], "length of data and target must match");
        CHECK(data_dims.size() == 2 && target_dims.size() == 1 && data_dims[1] == input_size, "sizes must match");
        
        CHECK(X==nullptr && Y==nullptr, "dataset should not be set before");
        seq_len = data_dims[0];
        X = data;
        Y = target;
    }
    
    template<typename T>
    T RNN<T>::fit_one_batch(const int start_from, bool update) {
        this->net.reopaque();
        for (int i = 0; i < this->input_signals.size(); i++) {
            this->input_signals[i]->reopaque();
            this->input_signals[i]->get_data()->copy_from(start_from+i, this->batch_size, X);
        }
        for (int i = 0; i < this->output_signals.size(); i++) {
            this->output_signals[i]->reopaque();
            this->output_signals[i]->get_target()->copy_from(start_from+i, this->batch_size, Y);
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
    
    template<typename T>
    void RNN<T>::fit() {
        for (int k = 1; k < this->num_epoch+1; k++) {
            printf("Epoch: %2d", k);
            auto start = chrono::system_clock::now();
            T loss = 0;
            
            int len = seq_len - max_len + 1 - this->batch_size + 1;
            for (int i = 0; i < len; i += this->batch_size) {
                loss += fit_one_batch(i);
                if (i % 1000 == 0) {
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