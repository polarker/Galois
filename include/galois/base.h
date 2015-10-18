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
    
    template<typename T>
    class Signal
    {
    public:
        SP_NArray<T> data = nullptr;
        
    public:
        Signal() {};
        Signal(const Signal& other) = delete;
        Signal& operator=(const Signal&) = delete;
        void set_dims(int m)                        { set_dims({m}); }
        void set_dims(int m, int n)                 { set_dims({m,n}); }
        void set_dims(int m, int n, int o)          { set_dims({m,n,o}); }
        void set_dims(int m, int n, int o, int k)   { set_dims({m,n,o,k}); }
        void set_dims(initializer_list<int> nums) {
            set_dims(vector<int>(nums));
        }
        virtual void set_dims(vector<int> nums) = 0;
        vector<int> get_dims() { assert(data); return data->get_dims(); }
        bool empty() { return data == nullptr; }
    };
    template<typename T>
    using SP_Signal = shared_ptr<Signal<T>>;

    template<typename T>
    class InnerSignal : public Signal<T>
    {
    public:
        SP_NArray<T> grad = nullptr;

    public:
        void set_dims(vector<int> nums) override {
            this->data = make_shared<NArray<T>>(nums);
            this->grad = make_shared<NArray<T>>(nums);
        }
    };
    template<typename T>
    using SP_InnerSignal = shared_ptr<InnerSignal<T>>;


    template<typename T>
    class InputSignal : public Signal<T>
    {
    public:
        void set_dims(vector<int> nums) {
            this->data = make_shared<NArray<T>>(nums);
        }
    };
    template<typename T>
    using SP_InputSignal = shared_ptr<InputSignal<T>>;


    template<typename T>
    class OutputSignal : public Signal<T>
    {
    public:
        SP_NArray<T> target_data = nullptr;
        T loss = 0;
        SP_NArray<T> extra_data = nullptr; // output signals need extra for optimization sometimes.

    public:
        void set_dims(vector<int> nums) {
            this->data = make_shared<NArray<T>>(nums);
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
            target_data = make_shared<NArray<T>>(nums);
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
            extra_data = make_shared<NArray<T>>(nums);
        }
    };
    template<typename T>
    using SP_OutputSignal = shared_ptr<OutputSignal<T>>;
    
    template<typename T>
    ostream& operator<<(std::ostream &strm, const SP_Signal<T> signal) {
        return strm << "{"
                    << signal->data->get_dims()
                    << "}";
    }
    template<typename T>
    ostream& operator<<(std::ostream &strm, const SP_InputSignal<T> signal) {
        return strm << "{"
        << signal->data->get_dims()
        << "}";
    }
    template<typename T>
    ostream& operator<<(std::ostream &strm, const SP_OutputSignal<T> signal) {
        return strm << "{"
        << signal->data->get_dims()
        << ","
        << signal->target_data->get_dims()
        << ","
        << signal->extra_data->get_dims()
        << "}";
    }
    
    
    
    // Filter does forward/backward propagation
    template<typename T>
    class Filter
    {
    public:
        virtual void forward(const vector<SP_Signal<T>> &in_signals, const vector<SP_Signal<T>> &out_signals) = 0;
        virtual void backward(const vector<SP_Signal<T>> &in_signals, const vector<SP_Signal<T>> &out_signals) = 0;
        virtual void set_dims(const vector<SP_Signal<T>> &in_signals,
                              const vector<SP_Signal<T>> &out_signals,
                              int batch_size) = 0;
        //virtual const Filter *Share() const;
        //virtual const Filter *Clone() const;
    };
    template<typename T>
    using SP_Filter = shared_ptr<Filter<T>>;
    
    template<typename T>
    class BFilter : public Filter<T> {};
    
    template<typename T>
    class PFilter : public Filter<T> {
    public:
        bool opaque = true;
    };
    template<typename T>
    using SP_PFilter = shared_ptr<PFilter<T>>;
    
    template<typename T>
    class GFilter : public Filter<T>
    {
    public:
        virtual set<SP_PFilter<T>> get_pfilters() = 0;
    };
    
    template<typename T>
    class TestBFilter : public BFilter<T>
    {
    public:
        void forward(const vector<SP_Signal<T>> in_signals, const vector<SP_Signal<T>> out_signals) override {
            cout << "forward" << endl;
        }
        void backward(const vector<SP_Signal<T>> in_signals, const vector<SP_Signal<T>> out_signals) override {
            cout << "backward" << endl;
        }
        void set_dims(const vector<SP_Signal<T>> in_signals,
                      const vector<SP_Signal<T>> out_signals,
                      int batch_size) {}
    };
    
    template<typename T>
    class TestPFilter : public PFilter<T>
    {
    public:
        void forward(const vector<SP_Signal<T>> in_signals, const vector<SP_Signal<T>> out_signals) override {
            cout << "forward" << endl;
        }
        void backward(const vector<SP_Signal<T>> in_signals, const vector<SP_Signal<T>> out_signals) override {
            cout << "backward" << endl;
        }
        void set_dims(const vector<SP_Signal<T>> in_signals,
                      const vector<SP_Signal<T>> out_signals,
                      int batch_size) override {}
    };
    
}

#endif
