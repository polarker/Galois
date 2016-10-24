#include "galois/models/seq_encoder_decoder.h"
#include "galois/gfilters/path.h"
#include "galois/filters.h"

namespace gs
{

    string seq_generate_id(string tag, int i) {
        return tag + "[" + to_string(i) + "]";
    }

    string seq_generate_id(string tag, int i, int j) {
        return tag + "[" + to_string(i) + "," + to_string(j) + "]";
    }

    template<typename T>
    SeqEncoderDecoder<T>::SeqEncoderDecoder(int _max_len_encoder,
                int _max_len_decoder,
                int _input_size,
                int _output_size,
                initializer_list<int> _hidden_sizes,
                int _batch_size,
                int _num_epoch,
                T _learning_rate,
                string _optimizer_name)
            : OrderedModel<T>(_batch_size, _num_epoch, _learning_rate, _optimizer_name)
            , max_len_encoder(_max_len_encoder)
            , max_len_decoder(_max_len_decoder)
            , input_size(_input_size)
            , output_size(_output_size)
            , hidden_sizes(_hidden_sizes) {
        auto h2hraw_encoder = vector<SP_Filter<T>>();
        for (auto hsize : hidden_sizes) {
            h2hraw_encoder.push_back(make_shared<Linear<T>>(hsize, hsize));
        }
        auto h2hraw_decoder = vector<SP_Filter<T>>();
        for (auto hsize : hidden_sizes) {
            h2hraw_decoder.push_back(make_shared<Linear<T>>(hsize, hsize));
        }
        auto x2hraw_encoder = vector<SP_Filter<T>>();
        for (int i = 0; i < hidden_sizes.size(); i++) {
            if (i == 0) {
                    x2hraw_encoder.push_back(make_shared<Embedding<T>>(input_size, hidden_sizes[i]));
            } else {
                x2hraw_encoder.push_back(make_shared<Linear<T>>(hidden_sizes[i-1], hidden_sizes[i]));
            }
        }
        auto x2hraw_decoder = vector<SP_Filter<T>>();
        for (int i = 0; i < hidden_sizes.size(); i++) {
            if (i == 0) {
                    x2hraw_decoder.push_back(make_shared<Embedding<T>>(output_size, hidden_sizes[i]));
            } else {
                x2hraw_decoder.push_back(make_shared<Linear<T>>(hidden_sizes[i-1], hidden_sizes[i]));
            }
        }
        auto h2yraw_decoder = make_shared<Linear<T>>(hidden_sizes.back(), output_size);

        for (int i = 0; i < max_len_encoder; i++) {
            for (int j = 0; j < hidden_sizes.size(); j++) {
                string hraw = seq_generate_id("hraw_encoder", i, j);
                string left_h = seq_generate_id("h_encoder", i-1, j);
                if (i > 0) {
                    this->add_link(left_h, hraw, h2hraw_encoder[j]->share());
                }
            }
            for (int j = 0; j < hidden_sizes.size(); j++) {
                string hraw = seq_generate_id("hraw_encoder", i, j);
                string down_h;
                if (j == 0) {
                    down_h = seq_generate_id("x_encoder", i);
                } else {
                    down_h = seq_generate_id("h_encoder", i, j-1);
                }
                this->add_link(down_h, hraw, x2hraw_encoder[j]->share());
                string h = seq_generate_id("h_encoder", i, j);
                this->add_link(hraw, h, make_shared<Tanh<T>>());
            }
        }
        for (int i = 0; i < max_len_decoder; i++) {
            for (int j = 0; j < hidden_sizes.size(); j++) {
                string hraw = seq_generate_id("hraw_decoder", i, j);
                string left_h;
                if (i == 0) {
                    left_h = seq_generate_id("h_encoder", max_len_encoder-1, j);
                } else {
                    left_h = seq_generate_id("h_decoder", i-1, j);
                }
                this->add_link(left_h, hraw, h2hraw_decoder[j]->share());
            }
            for (int j = 0; j < hidden_sizes.size(); j++) {
                string hraw = seq_generate_id("hraw_decoder", i, j);
                string down_h;
                if (j == 0) {
                    down_h = seq_generate_id("x_decoder", i);
                } else {
                    down_h = seq_generate_id("h_decoder", i, j-1);
                }
                this->add_link(down_h, hraw, x2hraw_decoder[j]->share());
                string h = seq_generate_id("h_decoder", i, j);
                this->add_link(hraw, h, make_shared<Tanh<T>>());
            }
            string yraw = seq_generate_id("yraw_decoder", i);
            string down_h = seq_generate_id("h_decoder", i, hidden_sizes.size()-1);
            this->add_link(down_h, yraw, h2yraw_decoder->share());
            string y = seq_generate_id("y_decoder", i);
            this->add_link(yraw, y, make_shared<CrossEntropy<T>>());
        }

        auto x_ids = vector<string>();
        auto y_ids = vector<string>();
        for (int i = 0; i < max_len_encoder; i++) {
            x_ids.push_back(seq_generate_id("x_encoder", i));
        }
        for (int i = 0; i < max_len_decoder; i++) {
            x_ids.push_back(seq_generate_id("x_decoder", i));
            y_ids.push_back(seq_generate_id("y_decoder", i));
        }
        this->add_input_ids(x_ids);
        this->add_output_ids(y_ids);

        this->compile();
    }

    template<typename T>
    void SeqEncoderDecoder<T>::add_train_dataset(const SP_NArray<T> data, const SP_NArray<T> target) {
        auto data_dims = data->get_dims();
        auto target_dims = target->get_dims();
        CHECK(data_dims[0] == target_dims[0], "length of data and target must match");
        CHECK(data_dims.size() == 2 && target_dims.size() == 2, "both should be an array of sentences");
        CHECK(data_dims[1] == max_len_encoder && target_dims[1] == max_len_decoder, "these should match");

        CHECK(train_X==nullptr && train_Y==nullptr, "dataset should not be set before");
        train_seq_count = data_dims[0];
        train_X = data;
        train_Y = target;
    }

    template<typename T>
    void SeqEncoderDecoder<T>::add_test_dataset(const SP_NArray<T> data, const SP_NArray<T> target) {
        auto data_dims = data->get_dims();
        auto target_dims = target->get_dims();
        CHECK(data_dims[0] == target_dims[0], "length of data and target must match");
        CHECK(data_dims.size() == 2 && target_dims.size() == 2, "both should be an array of sentences");
        CHECK(data_dims[1] == max_len_encoder && target_dims[1] == max_len_decoder, "these should match");

        CHECK(test_X==nullptr && test_Y==nullptr, "dataset should not be set before");
        test_seq_count = data_dims[0];
        test_X = data;
        test_Y = target;
    }

    template<typename T>
    T SeqEncoderDecoder<T>::train_one_batch(bool update) {
        uniform_int_distribution<> distribution(0, train_seq_count-1);
        vector<int> batch_ids(this->batch_size);
        for (int i = 0; i < this->batch_size; i++) {
            batch_ids[i] = distribution(galois_rn_generator);
        }

        this->net.reopaque();

        for (int i = 0; i < max_len_encoder; i++) {
            this->input_signals[i]->get_data()->copy_from(batch_ids, i, train_X);
        }
        this->input_signals[max_len_encoder]->get_data()->fill(0); // <EOS> characters

//        this->net.forward();
        int num1 = hidden_sizes.size()*2 + (max_len_encoder-1)*hidden_sizes.size()*3;
        for (int i = 0; i < num1; i++) {
            this->net.forward(i);
        }
        for (int i = 0; i < max_len_decoder; i++) {
            if (i > 0) {
                auto input_data = this->input_signals[max_len_encoder+i]->get_data();
                auto prev_output_data = this->output_signals[i-1]->get_data();
                input_data->copy_from(prev_output_data);
            }
            int num2 = num1+i*(hidden_sizes.size()*3 + 2);
            for (int j = num2; j < hidden_sizes.size()*3 + 2; j++) {
                this->net.forward(i);
            }
        }
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
    void SeqEncoderDecoder<T>::fit() {
        printf("Start training\n");

        for (int k = 1; k < this->num_epoch+1; k++) {
            printf("Epoch: %2d", k);
            auto start = chrono::system_clock::now();
            T loss = 0;

            int len = train_seq_count;
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

    template class SeqEncoderDecoder<float>;
    template class SeqEncoderDecoder<double>;

}