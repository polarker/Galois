#ifndef _GALOIS_UTILS_H_
#define _GALOIS_UTILS_H_

#include <algorithm>

namespace gs {
    
    template <typename T>
    const bool Contains( std::vector<T>& Vec, const T& Element )
    {
        if (std::find(Vec.begin(), Vec.end(), Element) != Vec.end())
            return true;
        
        return false;
    }
    
}

#endif
