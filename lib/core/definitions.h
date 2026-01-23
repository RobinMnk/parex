//
// Created by robin on 13.03.2025.
//

#ifndef RCUT_DEFINITIONS_H
#define RCUT_DEFINITIONS_H

#include <cassert>
#include <vector>

#ifdef DEBUG
#include <iostream>
#define INFO(msg) (std::cout << "[INFO] " << msg << std::endl)
#define LOG(msg) (std::cout << "[LOG] " << msg << std::endl)
#define WARN(msg) (std::cout << "[WARNING] " << msg << "!" << std::endl)
#else
#include <iostream>
#define INFO(msg) ((void)0)
#define LOG(msg) ((void)0)
#define WARN(msg) (std::cout << "[WARNING] " << msg << "!" << std::endl)
#endif

using NodeIx        = unsigned int;
using EdgeIx        = unsigned int;
using EdgeWeight    = unsigned int;
using frac_t        = float;

const int threads = 256;

struct AllSweepCuts {
    std::vector<NodeIx> clusterIds;
    std::vector<NodeIx> offsets;
    std::vector<float> sparsities;
};


#endif //RCUT_DEFINITIONS_H
