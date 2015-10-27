#ifndef _GALOIS_BASE_H_
#define _GALOIS_BASE_H_

# include "galois/narray.h"
# include "galois/utils.h"
# include <iostream>
# include <random>
# include <vector>
# include <set>
# include <cassert>

using namespace std;

namespace gs
{
    enum SignalType { InnerSignal, InputSignal, OutputSignal };
    
    template<typename T>
    class Signal
    {
    private:
        const SignalType type;
        
        SP_NArray<T> data = nullptr;
        // only for inner signal
        SP_NArray<T> grad = nullptr;
        
        // only for output signal
        SP_NArray<T> target = nullptr;
        shared_ptr<T> loss = nullptr;
        
        // for optimization, might be vector in the future
        SP_NArray<T> extra = nullptr;
        
    public:
        Signal() = delete;
        explicit Signal(SignalType type) : type(type) {};
        Signal(const Signal& other) = delete;
        Signal& operator=(const Signal&) = delete;
        
        bool empty() {
            return (data == nullptr) &&
                   (grad == nullptr) &&
                   (target == nullptr) &&
                   (loss == nullptr) &&
                   (extra == nullptr);
        }
        
        SignalType      get_type()      { return type;  }
        SP_NArray<T>    get_data()      { return data;  }
        SP_NArray<T>    get_grad()      { return grad;  }
        SP_NArray<T>    get_target()    { return target;}
        shared_ptr<T>   get_loss()      { return loss;  }
        SP_NArray<T>    get_extra()     { return extra; }
        
        void reopaque() {
            if (data)   { data->reopaque(); }
            if (grad)   { grad->reopaque(); }
            if (target) { target->reopaque(); }
            if (extra)  { extra->reopaque(); }
        }
        
        // set dims for data (and grad)
        void set_data_dims(int m)                        { set_data_dims({m}); }
        void set_data_dims(int m, int n)                 { set_data_dims({m,n}); }
        void set_data_dims(int m, int n, int o)          { set_data_dims({m,n,o}); }
        void set_data_dims(int m, int n, int o, int k)   { set_data_dims({m,n,o,k}); }
        void set_data_dims(initializer_list<int> nums) {
            set_data_dims(vector<int>(nums));
        }
        void set_data_dims(vector<int> nums) {
            CHECK(!data, "data should be nullptr before initialization");
            data = make_shared<NArray<T>>(nums);
            if (type == InnerSignal) {
                CHECK(!grad, "grad should be nullptr before initialization");
                grad = make_shared<NArray<T>>(nums);
            }
        }
        vector<int> get_data_dims() {
            CHECK(data, "data should be non-empty");
            return data->get_dims();
        }
        // set dims for target
        void set_target_dims(int m)                         { set_target_dims({m}); }
        void set_target_dims(int m, int n)                  { set_target_dims({m,n}); }
        void set_target_dims(int m, int n, int o)           { set_target_dims({m,n,o}); }
        void set_target_dims(int m, int n, int o, int k)    { set_target_dims({m,n,o,k}); }
        void set_target_dims(initializer_list<int> nums) {
            set_target_dims(vector<int>(nums));
        }
        void set_target_dims(vector<int> nums) {
            CHECK(type == OutputSignal, "only OutputSignal could set target");
            CHECK(!target, "target should be nullptr before initialization");
            target = make_shared<NArray<T>>(nums);
        }
        vector<int> get_target_dims() {
            CHECK(target, "target should be non-empty");
            return target->get_dims();
        }
        // initialize loss
        void initialize_loss() {
            CHECK(type == OutputSignal, "only OutputSignal could set loss");
            CHECK(!loss, "loss should be nullptr before initialization");
            loss = make_shared<T>(0);
        }
        // set dims for extra
        void set_extra_dims(int m)                         { set_extra_dims({m}); }
        void set_extra_dims(int m, int n)                  { set_extra_dims({m,n}); }
        void set_extra_dims(int m, int n, int o)           { set_extra_dims({m,n,o}); }
        void set_extra_dims(int m, int n, int o, int k)    { set_extra_dims({m,n,o,k}); }
        void set_extra_dims(initializer_list<int> nums) {
            set_extra_dims(vector<int>(nums));
        }
        void set_extra_dims(vector<int> nums) {
            CHECK(!extra, "extra should be nullptr before initialization");
            extra = make_shared<NArray<T>>(nums);
        }
        vector<int> get_extra_dims() {
            CHECK(extra, "extra should be non-empty");
            return extra->get_dims();
        }
    };
    template<typename T>
    using SP_Signal = shared_ptr<Signal<T>>;
    
    
    // Filter does forward/backward propagation
    template<typename T>
    class Filter
    {
    public:
        virtual void forward() = 0;
        virtual void backward() = 0;
        virtual void install_signals(const vector<SP_Signal<T>> &in_signals, const vector<SP_Signal<T>> &out_signals) = 0;
        virtual void set_dims(int batch_size) = 0;
        virtual void reopaque() = 0; // each filter only opaques its inner signal
        //virtual const Filter *Share() const;
        //virtual const Filter *Clone() const;
    };
    template<typename T>
    using SP_Filter = shared_ptr<Filter<T>>;
    
    template<typename T>
    class BFilter : public Filter<T> {
    public:
        void reopaque() override {}
    };
    
    template<typename T>
    class PFilter : public Filter<T> {
    public:
        virtual vector<SP_NArray<T>> get_params() = 0;
        virtual vector<SP_NArray<T>> get_grads() = 0;
    };
    template<typename T>
    using SP_PFilter = shared_ptr<PFilter<T>>;
    
    template<typename T>
    class GFilter : public Filter<T>
    {
    public:
        virtual set<SP_PFilter<T>> get_pfilters() = 0;
    };
    
}

#endif
