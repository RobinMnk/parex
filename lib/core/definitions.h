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
constexpr int WARP = 32;
constexpr int warpsPerBlock = threads / WARP;

constexpr int WARPMASK = WARP-1;
constexpr unsigned int BASE_SUBWARP_MASK = (WARP == 32) ? 0xFFFFFFFFU : ((1U << WARP) - 1U);


inline unsigned int getGridSize(unsigned int n) {
    return (n + threads - 1) / threads;
}

inline unsigned int getGridSizeWarpParallel(unsigned int n) {
    return (n + warpsPerBlock - 1) / warpsPerBlock;
}

inline unsigned int randSeed{6582};

constexpr double rw_stay = 0.1;
inline frac_t rw_threshold = 1e-4;
inline float sc_threshold = 0.3;

constexpr NodeIx INVALID_EDGE = static_cast<NodeIx>(-1);


using label_t = int32_t;

struct SweepCutData {
    label_t clusterId;
    float sparsity;
    NodeIx offset;
};

struct LabeledNode {
    NodeIx nix;
    label_t clusterId;
};

struct NodeData {
    NodeIx nix;
    // EdgeIx activeDegree;
    // int64_t label;  // a negative label represents inactive cluster

    // float rwValue;
    EdgeIx prefixVolume;
    int32_t prefixEdgeDiff;
    NodeIx offsetInCluster;
};

struct ClusterData {
    float rwSum;
    float maxPotential;
    float minPotential;
    NodeIx totalElements;
};


// struct NodeData {
//     NodeIx nix;
//     EdgeIx activeDegree;
//     int64_t label;  // a negative label represents inactive cluster
//
//     // float rwValue;
//     EdgeIx prefixVolume;
//     int32_t prefixEdgeDiff;
//     NodeIx offsetInCluster;
// };

struct AllSweepCuts {
    std::vector<label_t> clusterIds;
    std::vector<SweepCutData> cuts;
//    std::vector<PrefixValues> prefixSums;
};

struct FinalPartition {
    std::vector<label_t> clusterIds;
    int numClusters;
};


#endif //RCUT_DEFINITIONS_H
