//
// Created by robin on 26.01.2026.
//

#ifndef PAREX_TYPES_H
#define PAREX_TYPES_H

#include <thrust/device_vector.h>

#include "core/definitions.h"

template<typename T>
using dVec = thrust::device_vector<T>;

template<typename T>
void inspect(dVec<T> vec, NodeIx n) {
    std::vector<T> host_vec(n);
    thrust::copy(vec.begin(), vec.begin() + n, host_vec.begin());
    for (const T& t : host_vec) {
        std::cout << t << "  ";
    }
    std::cout << std::endl;
}

#define PRINT_DVEC(name, v) std::cout << name << ": "; inspect(v, v.size());

#endif //PAREX_TYPES_H
