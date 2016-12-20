#ifndef _GALOIS_PATH_H_
#define _GALOIS_PATH_H_

#include "galois/base.h"
#include <iostream>
#include <string>
#include <vector>
#include <set>

using namespace std;

namespace gs
{

    template<typename T>
    class Path : public GFilter<T>
    {
    private:
        vector<SP_Filter<T>> links;
        set<SP_PFilter<T>> pfilters;

        vector<SP_Signal<T>> inner_signals;

    public:
        Path() {}
        Path(const Path& other) = delete;
        Path& operator=(const Path&) = delete;

        void add_filter(SP_Filter<T> filter);

        SP_Filter<T> share() override;
        SP_Filter<T> clone() override;
        set<SP_PFilter<T>> get_pfilters() override;

        void install_signals(const vector<SP_Signal<T>>& in_signals, const vector<SP_Signal<T>>& out_signals) override;
        void set_dims(size_t batch_size) override;
        void reopaque() override;

        void forward() override;
        void backward() override;
    };

}

#endif
