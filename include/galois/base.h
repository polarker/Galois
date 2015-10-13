#ifndef _GALOIS_BASE_H_
#define _GALOIS_BASE_H_

# include <iostream>
# include <vector>
# include <set>
using namespace std;

namespace gs
{
    
    typedef double GArray;
    
    enum SignalType { InnerSignal, InputSignal, OutputSignal };
    
    class Signal
    {
    public:
        SignalType type;
        
        vector<int> dims;
        GArray *data;
        GArray *grad;
        
        GArray *target; // only use this for output signal
        double loss;    // only use this for output signal
        
        bool opaque;
        
        Signal() : type{InnerSignal},
                   dims{},
                   data{nullptr},
                   grad{nullptr},
                   target{nullptr},
                   loss{0},
                   opaque{true} {}
        Signal(const Signal& other) = delete;
        Signal& operator=(const Signal&) = delete;
    };
    
    // Filter does forward/backward propagation
    class Filter
    {
    public:
        virtual void Forward(shared_ptr<Signal> inputs, shared_ptr<Signal> outputs) = 0;
        virtual void Backward(shared_ptr<Signal> inputs, shared_ptr<Signal> outputs) = 0;
        virtual void set_dims(const vector<shared_ptr<Signal>> in_signals,
                              const vector<shared_ptr<Signal>> out_signals,
                              int batch_size) = 0;
        //virtual const Filter *Share() const;
        //virtual const Filter *Clone() const;
    };
    
    class BFilter : public Filter {};
    
    class PFilter : public Filter {};
    
    class GFilter : public Filter
    {
    public:
        virtual set<shared_ptr<PFilter>> get_pfilters() = 0;
    };
    
    class TestBFilter : public BFilter
    {
    public:
        void Forward(shared_ptr<Signal> inputs, shared_ptr<Signal> outputs) {
            inputs->opaque = true;
            outputs->opaque = true;
            cout << "forward" << endl;
        }
        void Backward(shared_ptr<Signal> inputs, shared_ptr<Signal> outputs) {
            inputs->opaque = true;
            outputs->opaque = true;
            cout << "backward" << endl;
        }
        void set_dims(const vector<shared_ptr<Signal>> in_signals,
                      const vector<shared_ptr<Signal>> out_signals,
                      int batch_size) {}
    };
    
    class TestPFilter : public PFilter
    {
    public:
        void Forward(shared_ptr<Signal> inputs, shared_ptr<Signal> outputs) override {
            inputs->opaque = true;
            outputs->opaque = true;
            cout << "forward" << endl;
        }
        void Backward(shared_ptr<Signal> inputs, shared_ptr<Signal> outputs) override {
            inputs->opaque = true;
            outputs->opaque = true;
            cout << "backward" << endl;
        }
        void set_dims(const vector<shared_ptr<Signal>> in_signals,
                      const vector<shared_ptr<Signal>> out_signals,
                      int batch_size) override {}
    };
    
}

#endif
