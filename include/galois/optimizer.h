#include "galois/base.h"

namespace gs
{
    
    template<typename T>
    class Optimizer
    {
    protected:
        T lrate;
        vector<SP_PFilter<T>>   pfilters;
        vector<SP_NArray<T>>    params;
        vector<SP_NArray<T>>    grads;
    
    public:
        virtual void update() = 0;
        virtual void compile(vector<SP_PFilter<T>> &pfs) = 0;
    };
    template<typename T>
    using SP_Optimizer = shared_ptr<Optimizer<T>>;
    
    template<typename T>
    class SGD_Optimizer : public Optimizer<T>
    {
    public:
        SGD_Optimizer(T lr) { this->lrate = lr; }
        
        void compile(vector<SP_PFilter<T>> &pfs) override {
            this->pfilters = pfs;
            this->params = vector<SP_NArray<T>>{};
            this->grads  = vector<SP_NArray<T>>{};
            for (auto pfilter : this->pfilters) {
                auto tmp_params = pfilter->get_params();
                auto tmp_grads = pfilter->get_grads();
                CHECK(tmp_params.size() == tmp_grads.size(), "numbers of params and grads should be equal");
                for (int i = 0; i < tmp_params.size(); i++) {
                    auto param = tmp_params[i];
                    auto grad = tmp_grads[i];
                    CHECK(param->get_dims() == grad->get_dims(), "param and grad should have the same dimensions");
                    this->params.push_back(param);
                    this->grads.push_back(grad);
                }
            }
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