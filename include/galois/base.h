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
    
    enum SignalType { InnerSignal, InputSignal, OutputSignal };
    
    template<typename T>
    class Signal
    {
    public:
        SignalType type;
        
        vector<int> dims = {};
        UP_NArray<T> data = nullptr;
        UP_NArray<T> grad = nullptr;
        
        UP_NArray<T> target = nullptr; // only use this for output signal
        T loss = 0;    // only use this for output signal
        
        bool opaque;
        
        Signal(SignalType t=InnerSignal) : type{t} {}
        Signal(const Signal& other) = delete;
        Signal& operator=(const Signal&) = delete;
        
        // set dims for data and grad
        void set_dims(int m)                        { set_dims({m}); }
        void set_dims(int m, int n)                 { set_dims({m,n}); }
        void set_dims(int m, int n, int o)          { set_dims({m,n,o}); }
        void set_dims(int m, int n, int o, int k)   { set_dims({m,n,o,k}); }
        void set_dims(initializer_list<int> nums) {
            set_dims(vector<int>(nums));
        }
        void set_dims(vector<int> nums) {
            for (auto m : nums) {
                dims.push_back(m);
            }
            data = make_unique<NArray<T>>(nums);
            grad = make_unique<NArray<T>>(nums);
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
            target = make_unique<NArray<T>>(nums);
        }
    };
    template<typename T>
    using SP_Signal = shared_ptr<Signal<T>>;
    
    // Filter does forward/backward propagation
    template<typename T>
    class Filter
    {
    public:
        virtual void Forward(SP_Signal<T> inputs, SP_Signal<T> outputs) = 0;
        virtual void Backward(SP_Signal<T> inputs, SP_Signal<T> outputs) = 0;
        virtual void set_dims(const vector<SP_Signal<T>> in_signals,
                              const vector<SP_Signal<T>> out_signals,
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
        void Forward(SP_Signal<T> inputs, SP_Signal<T> outputs) {
            inputs->opaque = true;
            outputs->opaque = true;
            cout << "forward" << endl;
        }
        void Backward(SP_Signal<T> inputs, SP_Signal<T> outputs) {
            inputs->opaque = true;
            outputs->opaque = true;
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
        void Forward(SP_Signal<T> inputs, SP_Signal<T> outputs) override {
            inputs->opaque = true;
            outputs->opaque = true;
            cout << "forward" << endl;
        }
        void Backward(SP_Signal<T> inputs, SP_Signal<T> outputs) override {
            inputs->opaque = true;
            outputs->opaque = true;
            cout << "backward" << endl;
        }
        void set_dims(const vector<SP_Signal<T>> in_signals,
                      const vector<SP_Signal<T>> out_signals,
                      int batch_size) override {}
    };
    
}

#endif
