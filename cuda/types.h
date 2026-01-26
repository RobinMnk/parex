//
// Created by robin on 26.01.2026.
//

#ifndef PAREX_TYPES_H
#define PAREX_TYPES_H

#include "core/definitions.h"

struct __align__(16) NodeData {
    NodeIx nix;
    NodeIx label;
    EdgeIx activeDegree;
    NodeIx rangeStart; // can be removed
};


#endif //PAREX_TYPES_H
