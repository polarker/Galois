#ifndef _GALOIS_OPTIMIZER_H_
#define _GALOIS_OPTIMIZER_H_

#include "galois/base.h"
#include "galois/utils.h"

namespace gs
{
    
    template<typename T>
    class Optimizer
    {
    protected:
        T lrate;
        vector<SP_NArray<T>>    params = {};
        vector<SP_NArray<T>>    grads = {};
    
    public:
        virtual void update() = 0;
        virtual void compile(vector<SP_NArray<T>> params, vector<SP_NArray<T>> grads) = 0;
    };
    template<typename T>
    using SP_Optimizer = shared_ptr<Optimizer<T>>;
    
    template<typename T>
    class SGD_Optimizer : public Optimizer<T>
    {
    public:
        SGD_Optimizer(T lr) { this->lrate = lr; }
        
        void compile(vector<SP_NArray<T>> params, vector<SP_NArray<T>> grads) override {
            CHECK(this->params.empty() && this->grads.empty(), "params and grads should not be set before");
            CHECK(params.size() == grads.size(), "params and grads should have equal size");
            for (int i = 0; i < params.size(); i++) {
                CHECK(params[i]->get_dims() == grads[i]->get_dims(), "param and grad should have the same dimensions");
            }
            this->params.insert(this->params.end(), params.begin(), params.end());
            this->grads.insert(this->grads.end(), grads.begin(), grads.end());
            cout << "The number of params: " << this->params.size() << endl;
            cout << "The number of grads: " << this->grads.size() << endl;
        }
        
        void update() override {
            for (int i = 0; i < this->params.size(); i++) {
                auto param = this->params[i];
                auto grad = this->grads[i];
                CHECK(!param->opaque(), "param should not be opaque");
                T lrate = this->lrate;
                MAP(param, [=](T x){return -lrate*x;}, grad);
            }
        }
        
    };
    
}

#endif
