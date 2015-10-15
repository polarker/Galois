#ifndef _GALOIS_BASE_H_
#define _GALOIS_BASE_H_

# include "galois/narray.h"
# include "galois/utils.h"
# include <iostream>
# include <vector>
# include <set>
using namespace std;

namespace gs
{
    
//    template<typename T>
//    using NArray = vector<T>;
    
//    enum SignalType { InnerSignal, InputSignal, OutputSignal };
    
    template<typename T>
    class Signal
    {
    public:
        vector<int> dims = {};
        SP_NArray<T> data = nullptr;

        bool opaque = true;
        
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
            for (auto m : nums) {
                this->dims.push_back(m);
            }
            this->data = make_shared<NArray<T>>(nums);
            grad = make_shared<NArray<T>>(nums);
        }
    };
    template<typename T>
    using SP_InnerSignal = shared_ptr<InnerSignal<T>>;


    template<typename T>
    class InputSignal : public Signal<T>
    {
    public:
        void set_dims(vector<int> nums) {
            for (auto m : nums) {
                this->dims.push_back(m);
            }
            this->data = make_shared<NArray<T>>(nums);
        }
    };
    template<typename T>
    using SP_InputSignal = shared_ptr<InputSignal<T>>;


    template<typename T>
    class OutputSignal : public Signal<T>
    {
    public:
        SP_NArray<T> target = nullptr;
        T loss = 0;

    public:
        void set_dims(vector<int> nums) {
            for (auto m : nums) {
                this->dims.push_back(m);
            }
            this->data = make_shared<NArray<T>>(nums);
        }
        // set dims for target
        void set_target_dims(int m)                         { set_dims({m}); }
        void set_target_dims(int m, int n)                  { set_dims({m,n}); }
        void set_target_dims(int m, int n, int o)           { set_dims({m,n,o}); }
        void set_target_dims(int m, int n, int o, int k)    { set_dims({m,n,o,k}); }
        void set_target_dims(initializer_list<int> nums) {
            set_target_dims(vector<int>(nums));
        }
        void set_target_dims(vector<int> nums) {
            target = make_shared<NArray<T>>(nums);
        }
    };
    template<typename T>
    using SP_OutputSignal = shared_ptr<OutputSignal<T>>;
    
    template<typename T>
    ostream& operator<<(std::ostream &strm, const SP_Signal<T> signal) {
        return strm << "{"
                    << signal->dims
                    << ","
                    << signal->data->get_dims()
//                    << ","
//                    << signal->grad->get_dims()
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
    class PFilter : public Filter<T> {};
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
//            inputs->opaque = true;
//            outputs->opaque = true;
            cout << "forward" << endl;
        }
        void backward(const vector<SP_Signal<T>> in_signals, const vector<SP_Signal<T>> out_signals) override {
//            inputs->opaque = true;
//            outputs->opaque = true;
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
//            inputs->opaque = true;
//            outputs->opaque = true;
            cout << "forward" << endl;
        }
        void backward(const vector<SP_Signal<T>> in_signals, const vector<SP_Signal<T>> out_signals) override {
//            inputs->opaque = true;
//            outputs->opaque = true;
            cout << "backward" << endl;
        }
        void set_dims(const vector<SP_Signal<T>> in_signals,
                      const vector<SP_Signal<T>> out_signals,
                      int batch_size) override {}
    };
    
}

#endif
