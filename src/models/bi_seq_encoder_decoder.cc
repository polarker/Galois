#include "galois/models/bi_seq_encoder_decoder.h"
#include "galois/gfilters/path.h"
#include "galois/filters.h"

#include <chrono>

namespace gs
{

    string bi_seq_generate_id(string tag, int i) {
        return tag + "[" + to_string(i) + "]";
    }

    string bi_seq_generate_id(string tag, int i, int j) {
        return tag + "[" + to_string(i) + "," + to_string(j) + "]";
    }

    template<typename T>
    class Encoder : public OrderedNet<T>
    {
    public:
        Encoder(const Encoder& other) = delete;
        Encoder& operator=(const Encoder&) = delete;
        Encoder(size_t max_len, size_t input_size, vector<size_t> hidden_sizes) {
            auto h2hraw = vector<SP_Filter<T>>();
            for (auto hsize : hidden_sizes) {
                h2hraw.push_back(make_shared<Linear<T>>(hsize, hsize));
            }
            auto x2hraw = vector<SP_Filter<T>>();
            for (size_t i = 0; i < hidden_sizes.size(); i++) {
                if (i == 0) {
                    x2hraw.push_back(make_shared<Embedding<T>>(input_size, hidden_sizes[i]));
                } else {
                    x2hraw.push_back(make_shared<Linear<T>>(hidden_sizes[i-1], hidden_sizes[i]));
                }
            }
            for (size_t i = 0; i < max_len; i++) {
                for (size_t j = 0; j < hidden_sizes.size(); j++) {
                    string hraw = bi_seq_generate_id("hraw", i, j);
                    string left_h = bi_seq_generate_id("h", i-1, j);
                    if (i > 0) {
                        BaseNet<T>::add_link(left_h, hraw, h2hraw[j]->share());
                    }
                }
                for (size_t j = 0; j < hidden_sizes.size(); j++) {
                    string hraw = bi_seq_generate_id("hraw", i, j);
                    string down_h;
                    if (j == 0) {
                        down_h = bi_seq_generate_id("x", i);
                    } else {
                        down_h = bi_seq_generate_id("h", i, j-1);
                    }
                    BaseNet<T>::add_link(down_h, hraw, x2hraw[j]->share());
                    string h = bi_seq_generate_id("h", i, j);
                    BaseNet<T>::add_link(hraw, h, make_shared<Tanh<T>>());
                }
            }
            auto x_ids = vector<string>();
            auto y_ids = vector<string>();
            for (size_t i = 0; i < max_len; i++) {
                x_ids.push_back(bi_seq_generate_id("x", i));
            }
            for (size_t j = 0; j < hidden_sizes.size(); j++) {
                y_ids.push_back(bi_seq_generate_id("h", max_len-1, j));
            }
            this->add_input_ids(x_ids);
            this->add_output_ids(y_ids);

            this->fix_net();
        }

        using OrderedNet<T>::forward;
        using OrderedNet<T>::backward;
    };

    template<typename T>
    class Decoder : public OrderedNet<T>
    {
    private:
        size_t max_len;
        size_t num_hidden_layer;

        SP_Signal<T> initial_input_signal;

    public:
        Decoder(const Decoder& other) = delete;
        Decoder& operator=(const Decoder&) = delete;
        Decoder(size_t max_len, size_t input_size, vector<size_t> hidden_sizes)
                : max_len(max_len)
                , num_hidden_layer(hidden_sizes.size()) {
            auto h2hraw = vector<SP_Filter<T>>();
            for (auto hsize : hidden_sizes) {
                h2hraw.push_back(make_shared<Linear<T>>(hsize, hsize));
            }
            auto x2hraw = vector<SP_Filter<T>>();
            for (size_t i = 0; i < hidden_sizes.size(); i++) {
                if (i == 0) {
                    x2hraw.push_back(make_shared<Embedding<T>>(input_size, hidden_sizes[i]));
                } else {
                    x2hraw.push_back(make_shared<Linear<T>>(hidden_sizes[i-1], hidden_sizes[i]));
                }
            }
            auto h2yraw = make_shared<Linear<T>>(hidden_sizes.back(), input_size);

            for (size_t i = 0; i < max_len; i++) {
                for (size_t j = 0; j < hidden_sizes.size(); j++) {
                    string hraw = bi_seq_generate_id("hraw", i, j);
                    string left_h = bi_seq_generate_id("h", i-1, j);
                    BaseNet<T>::add_link(left_h, hraw, h2hraw[j]->share());
                }
                for (size_t j = 0; j < hidden_sizes.size(); j++) {
                    string hraw = bi_seq_generate_id("hraw", i, j);
                    string down_h;
                    if (j == 0) {
                        down_h = bi_seq_generate_id("x", i);
                    } else {
                        down_h = bi_seq_generate_id("h", i, j-1);
                    }
                    BaseNet<T>::add_link(down_h, hraw, x2hraw[j]->share());
                    string h = bi_seq_generate_id("h", i, j);
                    BaseNet<T>::add_link(hraw, h, make_shared<Tanh<T>>());
                }
                string yraw = bi_seq_generate_id("yraw", i);
                string down_h = bi_seq_generate_id("h", i, hidden_sizes.size()-1);
                BaseNet<T>::add_link(down_h, yraw, h2yraw->share());
                string y = bi_seq_generate_id("y", i);
                BaseNet<T>::add_link(yraw, y, make_shared<CrossEntropy<T>>());
            }

            auto x_ids = vector<string>();
            auto y_ids = vector<string>();
            for (size_t i = 0; i < max_len; i++) {
                x_ids.push_back(bi_seq_generate_id("x", i));
                y_ids.push_back(bi_seq_generate_id("y", i));
            }
            for (size_t i = 0; i < hidden_sizes.size(); i++) {
                x_ids.push_back(bi_seq_generate_id("h", -1, i));
            }
            this->add_input_ids(x_ids);
            this->add_output_ids(y_ids);

            this->fix_net();

            initial_input_signal = make_shared<Signal<T>>(InputSignal);
        }

        SP_Filter<T> share() override {
            throw "to be implemented";
        }

        void install_signals(const vector<SP_Signal<T>> &in_signals, const vector<SP_Signal<T>> &out_signals) override {
            cout << "virtual function installing signals" << endl;
            auto new_in_signals = vector<SP_Signal<T>>();
            new_in_signals.push_back(initial_input_signal);
            new_in_signals.insert(new_in_signals.end(), out_signals.begin(), out_signals.end()-1);
            new_in_signals.insert(new_in_signals.end(), in_signals.begin(), in_signals.end());
            OrderedNet<T>::install_signals(new_in_signals, out_signals);
        }

        void forward() override {
            initial_input_signal->get_data()->fill(0);
            OrderedNet<T>::forward();
        }
        using OrderedNet<T>::backward;
    };

    template<typename T>
    BiSeqEncoderDecoder<T>::BiSeqEncoderDecoder(
                size_t _max_len_one,
                size_t _max_len_another,
                size_t _input_size_one,
                size_t _input_size_another,
                initializer_list<size_t> _hidden_sizes,
                size_t _batch_size,
                int _num_epoch,
                T _learning_rate,
                string _optimizer_name)
            : OrderedModel<T>(_batch_size, _num_epoch, _learning_rate, _optimizer_name)
            , max_len_one(_max_len_one)
            , max_len_another(_max_len_another)
            , input_size_one(_input_size_one)
            , input_size_another(_input_size_another)
            , hidden_sizes(_hidden_sizes) {
        auto encoder_one = make_shared<Encoder<T>>(max_len_one, input_size_one, hidden_sizes);
        auto decoder_one = make_shared<Decoder<T>>(max_len_one, input_size_one, hidden_sizes);
        auto encoder_another = make_shared<Encoder<T>>(max_len_another, input_size_another, hidden_sizes);
        auto decoder_another = make_shared<Decoder<T>>(max_len_another, input_size_another, hidden_sizes);

        auto x_ids_one = vector<string>();
        auto x_ids_another = vector<string>();
        auto h_ids_one = vector<string>();
        auto h_ids_another = vector<string>();
        auto y_ids_one2one = vector<string>();
        auto y_ids_one2another = vector<string>();
        auto y_ids_another2one = vector<string>();
        auto y_ids_another2another = vector<string>();
        for (size_t i = 0; i < max_len_one; i++) {
            x_ids_one.push_back(bi_seq_generate_id("x_one", i));
            y_ids_one2one.push_back(bi_seq_generate_id("y_one2one", i));
            y_ids_another2one.push_back(bi_seq_generate_id("y_another2one", i));
        }
        for (size_t j = 0; j < hidden_sizes.size(); j++) {
            h_ids_one.push_back(bi_seq_generate_id("h_one", max_len_one-1, j));
            h_ids_another.push_back(bi_seq_generate_id("h_another", max_len_another-1, j));
        }
        for (size_t i = 0; i < max_len_another; i++) {
            x_ids_another.push_back(bi_seq_generate_id("x_another", i));
            y_ids_one2another.push_back(bi_seq_generate_id("y_one2another", i));
            y_ids_another2another.push_back(bi_seq_generate_id("y_another2another", i));
        }

        this->add_link(x_ids_one, h_ids_one, encoder_one);
        this->add_link(h_ids_one, y_ids_one2one, decoder_one);
        this->add_link(h_ids_one, y_ids_one2another, decoder_another);
        this->add_link(x_ids_another, h_ids_another, encoder_another->share());
        this->add_link(h_ids_another, y_ids_another2another, decoder_another->share());
        this->add_link(h_ids_another, y_ids_another2one, decoder_one->share());

        this->add_input_ids(x_ids_one);
        this->add_input_ids(x_ids_another);
        this->add_output_ids(y_ids_one2one);
        this->add_output_ids(y_ids_one2another);
        this->add_output_ids(y_ids_another2one);
        this->add_output_ids(y_ids_another2another);

        this->compile();
    }

    template<typename T>
    void BiSeqEncoderDecoder<T>::add_train_dataset(const SP_NArray<T> one, const SP_NArray<T> another) {
        auto one_dims = one->get_dims();
        auto another_dims = another->get_dims();
        CHECK(one_dims[0] == another_dims[0], "length of data and target must match");
        CHECK(one_dims.size() == 2 && another_dims.size() == 2, "both should be an array of sentences");
        CHECK(one_dims[1] == max_len_one && another_dims[1] == max_len_another, "these should match");

        CHECK(train_one==nullptr && train_another==nullptr, "dataset should not be set before");
        train_seq_count = one_dims[0];
        train_one = one;
        train_another = another;
    }

    template<typename T>
    T BiSeqEncoderDecoder<T>::train_one_batch(bool update) {
        uniform_int_distribution<> distribution(0, train_seq_count-1);
        vector<size_t> batch_ids(this->batch_size);
        for (size_t i = 0; i < this->batch_size; i++) {
            batch_ids[i] = distribution(galois_rn_generator);
        }

        this->net.reopaque();

        for (size_t i = 0; i < max_len_one; i++) {
            this->input_signals[i]->get_data()->copy_from(batch_ids, i, train_one);
        }
        for (size_t i = max_len_one; i < max_len_one+max_len_another; i++) {
            this->input_signals[i]->get_data()->copy_from(batch_ids, i, train_another);
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
    void BiSeqEncoderDecoder<T>::fit() {
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

    template class BiSeqEncoderDecoder<float>;
    template class BiSeqEncoderDecoder<double>;

}
