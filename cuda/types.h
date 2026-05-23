//
// Created by robin on 26.01.2026.
//

#ifndef PAREX_TYPES_H
#define PAREX_TYPES_H

#include <thrust/device_vector.h>

#include "core/definitions.h"

template<typename T>
using dVec = thrust::device_vector<T>;


#endif //PAREX_TYPES_H
