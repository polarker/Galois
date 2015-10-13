#ifndef _GALOIS_BASE_H_
#define _GALOIS_BASE_H_

# include <iostream>
# include <vector>
# include <set>
using namespace std;

namespace gs
{
    
    template<typename T>
    using NArray = vector<T>;
    
    enum SignalType { InnerSignal, InputSignal, OutputSignal };
    
    template<typename T>
    class Signal
    {
    public:
        SignalType type;
        
        vector<int> dims;
        NArray<T> *data;
        NArray<T> *grad;
        
        NArray<T> *target; // only use this for output signal
        T loss;    // only use this for output signal
        
        bool opaque;
        
        Signal(SignalType t=InnerSignal)
                : type{t}
                , dims{}
                , data{nullptr}
                , grad{nullptr}
                , target{nullptr}
                , loss{0}
                , opaque{true} {}
        Signal(const Signal& other) = delete;
        Signal& operator=(const Signal&) = delete;
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
