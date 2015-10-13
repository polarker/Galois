#ifndef _GALOIS_BASE_H_
#define _GALOIS_BASE_H_

# include <iostream>
# include <set>
using namespace std;

namespace gs
{
    
    typedef double GArray;
    
    enum SignalType { InputSignal, InnerSignal, OutputSignal };
    
    class Signal
    {
    public:
        SignalType type;
        
        tuple<int, int> dims;
        GArray *data;
        GArray *grad;
        
        GArray *target; // only use this for output signal
        double loss;    // only use this for output signal
        
        bool opaque;
    };
    
    // Filter does forward/backward propagation
    class Filter
    {
    public:
        virtual void Forward(Signal *inputs, Signal *outputs) const = 0;
        virtual void Backward(Signal *inputs, Signal *outputs) const = 0;
        //virtual const Filter *Share() const;
        //virtual const Filter *Clone() const;
    };
    
    class BFilter : public Filter {};
    
    class PFilter : public Filter {};
    
    class GFilter : public Filter
    {
    public:
        virtual set<PFilter*> get_pfilters() = 0;
    };
    
    class TestBFilter : public BFilter
    {
    public:
        void Forward(Signal *inputs, Signal *outputs) const {
            inputs->opaque = true;
            outputs->opaque = true;
            cout << "forward" << endl;
        }
        void Backward(Signal *inputs, Signal *outputs) const {
            inputs->opaque = true;
            outputs->opaque = true;
            cout << "backward" << endl;
        }
    };
    
    class TestPFilter : public PFilter
    {
    public:
        void Forward(Signal *inputs, Signal *outputs) const {
            inputs->opaque = true;
            outputs->opaque = true;
            cout << "forward" << endl;
        }
        void Backward(Signal *inputs, Signal *outputs) const {
            inputs->opaque = true;
            outputs->opaque = true;
            cout << "backward" << endl;
        }
    };
    
}

#endif
