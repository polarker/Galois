#ifndef _GALOIS_ORDEREDNET_H_
#define _GALOIS_ORDEREDNET_H_

#include "galois/base.h"
#include "galois/gfilters/base_net.h"

using namespace std;

namespace gs
{

    template<typename T>
    class OrderedNet : public BaseNet<T>
    {
    public:
        OrderedNet() {}
        OrderedNet(const OrderedNet& other) = delete;
        OrderedNet& operator=(const OrderedNet&) = delete;

        void add_link(const vector<string>&, const vector<string>&, SP_Filter<T>) override;
        void add_input_ids(const string);
        void add_input_ids(const initializer_list<string>);
        void add_input_ids(const vector<string>&);
        void add_output_ids(const string);
        void add_output_ids(const initializer_list<string>);
        void add_output_ids(const vector<string>&);

        void fix_net();

        // in order to share a net, methods above should be called and methods below should not be called
        SP_Filter<T> share() override;

        // propagation could just apply to the subnet that contains only those links with index not less than idx
        using BaseNet<T>::forward;
        using BaseNet<T>::backward;
        void forward(int idx);
        void backward(int idx);
    };

}

#endif
