//
// Created by robin on 19.01.2026.
//

#ifndef PAREX_DEVRANDOMWALK_H
#define PAREX_DEVRANDOMWALK_H

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/functional.h>

#include <curand_kernel.h>

#include <cub/cub.cuh>
#include <cassert>
#include <thrust/detail/binary_search.inl>

#include "devGraph.h"


struct NormalDistributionFunctor {
    unsigned int base_seed;

    explicit NormalDistributionFunctor(unsigned int s) : base_seed(s) {}

    __host__ __device__
    unsigned int hash_idx(unsigned int x) const {
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = (x >> 16) ^ x;
        return x;
    }

    __host__ __device__
    frac_t operator()(const NodeIx idx) const {
        unsigned int thread_seed = hash_idx(base_seed ^ idx);

        thrust::default_random_engine rng(thread_seed);
        thrust::normal_distribution<frac_t> dist;

        return dist(rng);
    }

//     __host__ __device__
//     frac_t operator()(const NodeIx idx) const {
//         thrust::default_random_engine rng(base_seed);
//         thrust::normal_distribution<frac_t> dist;
//         rng.discard(idx);
//         return dist(rng);
// //        return static_cast<frac_t>(idx) / 4096;
//     }
};


__device__
inline uint64_t packKey(int label, float v) {
    uint32_t i = __float_as_uint(v);

    uint32_t mask = (i & 0x80000000) ? 0xffffffff : 0x80000000;
    uint32_t ordered = i ^ mask;

    uint32_t orderedLabel = static_cast<uint32_t>(label) ^ 0x80000000;

    // Shift label to high bits, place ordered float in low bits
    return (static_cast<uint64_t>(orderedLabel) << 32) | ordered;
}

// __global__
// void lazyRandomWalkKernel(
//         NodeIx numNodes,
//         const NodeIx* __restrict__ neighbors,
//         const EdgeIx* __restrict__ ranges,
//         NodeData* __restrict__ nodeData,
//         const EdgeIx* __restrict__ activeDegrees,
//         const frac_t* __restrict__ old_dist,
//         frac_t* __restrict__ dist,
//         uint64_t* __restrict__ packedKeys
// ) {
//     NodeIx i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= numNodes) return;
//
//     // early exit for inactive nodes
//     const NodeData data = nodeData[i];
//     if(data.label < 0) {
//         packedKeys[i] = packKey(data.label, 0);
//         return;
//     }; // label < 0 considered inactive
//
//     assert(data.nix == i && "nix mismatch!");
//
//     frac_t incoming_sum = 0.0f;
//     const EdgeIx rangeStart = ranges[i];
//     const EdgeIx rangeEnd = ranges[i] + data.degree;
//
//     for (NodeIx j = rangeStart; j < rangeEnd; ++j) {
//         const NodeIx neighbor = __ldg(&neighbors[j]);
//         const NodeData nbData = nodeData[neighbor];
//         const EdgeIx nbDeg = __ldg(&activeDegrees[neighbor]);
//
//         if(nbData.label == data.label && nbDeg != 0) {
//             incoming_sum += (__ldg(&old_dist[neighbor]) / static_cast<frac_t>(nbDeg));
//         }
//     }
//
//     const frac_t nodeVal = (incoming_sum * (1.0f - rw_stay)) + (old_dist[i] * rw_stay);
//
//     dist[i] = nodeVal;
//     nodeData[i].activeDegree = activeDegrees[i];
//
//     // FOR SWEEP CUT:
//     // Already Pack label and rw value into packedKeys for sorting!!
//     packedKeys[i] = packKey(data.label, nodeVal);
// }


struct LabelExtractorRW {
    __host__ __device__
    inline int operator()(const NodeData& data) const {
        return data.label;
    }
};

struct ClusterData {
    float rwSum;
    float maxPotential;
    float minPotential;
    NodeIx totalElements;
};

struct ClusterDataExtractorRW {
    __device__
    inline ClusterData operator()(const NodeData& nodeData) const {
        const float rwValue = nodeData.rwValue;
        return {rwValue, rwValue, rwValue, 1};
    }
};


struct ClusterDataReduceOp {
    __host__ __device__
    ClusterData operator()(const ClusterData& a, const ClusterData& b) const {
        return {
            a.rwSum + b.rwSum,
            a.maxPotential > b.maxPotential ? a.maxPotential : b.maxPotential,
            a.minPotential < b.minPotential ? a.minPotential : b.minPotential,
            a.totalElements + b.totalElements
        };
    }
};


struct WalkEdgeLogic {
    const NodeIx* __restrict__ neighbors;
    const EdgeIx* __restrict__ activeDegrees;
    const double* __restrict__ dist;

    __device__ __forceinline__
    double operator()(EdgeIx edgeIdx) const {
        const NodeIx tgtNode = __ldg(&neighbors[edgeIdx]);
        if (tgtNode == INVALID_EDGE) {
            // inactive edge
            return 0.0;
        }

        const EdgeIx nbDeg = __ldg(&activeDegrees[tgtNode]);

        if (nbDeg > 0) {
            return __ldg(&dist[tgtNode]) / static_cast<float>(nbDeg);
        }
        return 0.0f;
    }
};

__global__
void finalizeRandomWalk(
    NodeIx numNodes,
    const EdgeIx* __restrict__ activeDegrees,
    const double* __restrict__ incoming_sums,
    const int* __restrict__ labels,
    double* __restrict__ dist,
    NodeData* __restrict__ nodeData,
    uint64_t* __restrict__ packedKeys
) {
    NodeIx i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) return;

    const int label = __ldg(&labels[i]);
    if (label < 0) {
        packedKeys[i] = packKey(label, 0);
        return;
    }

    const double nodeVal = (incoming_sums[i] * (1.0f - rw_stay)) + (dist[i] * rw_stay);
    const float nodeValF = static_cast<float>(nodeVal);

    dist[i] = nodeVal;
    nodeData[i].activeDegree = activeDegrees[i];
    nodeData[i].label = label;
    nodeData[i].nix = i;
    nodeData[i].rwValue = nodeValF;
    packedKeys[i] = packKey(label, nodeValF);
}



class RandomWalkManager {
    thrust::device_vector<double> dist;
    thrust::device_vector<double> incomingSums;

    NodeIx numNodes;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    // int maxLabel{0};

public:
    explicit RandomWalkManager(NodeIx n) : dist(n), incomingSums(n), numNodes(n) {
        initRandomWalk(randSeed);
        prepare_cub();
    }

    ~RandomWalkManager() {
        if (d_temp_storage) cudaFree(d_temp_storage);
    }

    auto& getValues() {
        return dist;
    }

    void stepFast(GraphManager& gm,
          cub::DoubleBuffer<NodeData>& partition,
          cub::DoubleBuffer<uint64_t>& packedKeys,
          thrust::device_vector<int>& nodeLabels,
          thrust::device_vector<EdgeIx>& activeDegrees
    ) {
        const EdgeIx* rangesPtr = thrust::raw_pointer_cast(gm.getRanges().data());
        const NodeIx* neighborsPtr = thrust::raw_pointer_cast(gm.getNeighbors().data());
        const EdgeIx* activeDegsPtr = thrust::raw_pointer_cast(activeDegrees.data());
        double* distPtr = thrust::raw_pointer_cast(dist.data());
        double* incomingSumsPtr = thrust::raw_pointer_cast(incomingSums.data());
        const int* labelsPtr = thrust::raw_pointer_cast(nodeLabels.data());

        thrust::counting_iterator<EdgeIx> edgeIndexIter(0);
        WalkEdgeLogic functor{neighborsPtr, activeDegsPtr, distPtr};

        auto transIter = thrust::make_transform_iterator(edgeIndexIter, functor);

        // perform the reduction
        cub::DeviceSegmentedReduce::Sum(
            d_temp_storage, temp_storage_bytes,
            transIter, incomingSumsPtr,
            static_cast<int>(numNodes), rangesPtr, rangesPtr + 1
        );

        // Finalize
        int gridSize = (numNodes + threads - 1) / threads;

        finalizeRandomWalk<<<gridSize, threads, 0, nullptr>>>(
            numNodes,
            activeDegsPtr,
            incomingSumsPtr,
            labelsPtr,
            distPtr,
            partition.Current(),
            packedKeys.Current()
        );
    }



    // void step(GraphManager& gm,
    //       cub::DoubleBuffer<NodeData>& partition,
    //       cub::DoubleBuffer<uint64_t>& packedKeys,
    //       thrust::device_vector<EdgeIx>& activeDegrees,
    //       cudaStream_t stream = nullptr
    // ) {
    //
    //     // BEFORE:
    //
    //     // recenter clusters -> find potential -> deactivate if necessary!!
    //
    //     old_dist.swap(dist);
    //
    //     unsigned int blocksPerGrid = (numNodes + threads - 1) / threads;
    //
    //     lazyRandomWalkKernel<<<blocksPerGrid, threads, 0, stream>>>(
    //             numNodes,
    //             thrust::raw_pointer_cast(gm.getNeighbors().data()),
    //             thrust::raw_pointer_cast(gm.getRanges().data()),
    //             partition.Current(),
    //             thrust::raw_pointer_cast(activeDegrees.data()),
    //             thrust::raw_pointer_cast(old_dist.data()),
    //             thrust::raw_pointer_cast(dist.data()),
    //             packedKeys.Current()
    //     );
    // }

    // int getMaxLabel() const {
    //     return maxLabel;
    // }

    [[nodiscard]] const thrust::device_vector<double>& randomWalkValues() const {
        return dist;
    }

    std::vector<frac_t> valuesToCPU() {
        std::vector<frac_t> rwVals(numNodes);
        thrust::copy(dist.begin(), dist.end(), rwVals.begin());
        return rwVals;
    }

private:
    void prepare_cub() {
        WalkEdgeLogic dryRunFunctor{nullptr, nullptr, nullptr};
        auto dryRunIter = thrust::make_transform_iterator(thrust::make_counting_iterator<EdgeIx>(0), dryRunFunctor);

        EdgeIx* nullOffsets = static_cast<EdgeIx*>(nullptr);
        frac_t* nullOutput  = static_cast<frac_t*>(nullptr);

        // Dry run to get temp_storage_bytes
        cudaError_t err = cub::DeviceSegmentedReduce::Sum(
            nullptr, temp_storage_bytes,
            dryRunIter, nullOutput,
            static_cast<int>(numNodes), nullOffsets, nullOffsets
        );

        if (err != cudaSuccess) {
            printf("CUB Error: %s\n", cudaGetErrorString(err));
        }

        cudaMalloc(&d_temp_storage, temp_storage_bytes);
    }

public:
    void initRandomWalk(unsigned int s) {
        thrust::transform(thrust::make_counting_iterator<NodeIx>(0),
                          thrust::make_counting_iterator(numNodes),
                          dist.begin(),
                          NormalDistributionFunctor(s)
        );
    }
};

#endif //PAREX_DEVRANDOMWALK_H
