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

constexpr int threads = 256;
inline unsigned int randSeed{6582};

constexpr frac_t rw_stay = 0.1;
inline frac_t rw_threshold = 1e-4;
inline float sc_threshold = 0.3;

constexpr NodeIx INVALID_EDGE = static_cast<NodeIx>(-1);

struct SweepCutData {
    int64_t clusterId;
    float sparsity;
    NodeIx offset;
};

struct NodeData {
    NodeIx nix;
    EdgeIx activeDegree;
    int64_t label;  // a negative label represents inactive cluster

    float rwValue;
    EdgeIx prefixVolume;
    int32_t prefixEdgeDiff;
    NodeIx offsetInCluster;
};


struct AllSweepCuts {
    std::vector<int> clusterIds;
    std::vector<SweepCutData> cuts;
//    std::vector<PrefixValues> prefixSums;
};

struct FinalPartition {
    std::vector<int> clusterIds;
    int numClusters;
};


#endif //RCUT_DEFINITIONS_H
