//
// Created by robin on 19.01.2026.
//

#ifndef PAREX_DEVRANDOMWALK_H
#define PAREX_DEVRANDOMWALK_H

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>

#include <curand_kernel.h>

#include <cub/cub.cuh>
#include <cassert>
#include <thrust/detail/binary_search.inl>

#include "devGraph.h"


struct NormalDistributionFunctor {
    unsigned int base_seed{10};
    const NodeIx* degrees{nullptr};

    explicit NormalDistributionFunctor(unsigned int s, const NodeIx* degs) : base_seed(s), degrees(degs) {}

    __device__
    unsigned int hash_idx(unsigned int x) const {
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = (x >> 16) ^ x;
        return x;
    }

    __device__
    frac_t operator()(const NodeIx idx) const {
        const unsigned int thread_seed = hash_idx(base_seed ^ idx);
        const EdgeIx deg = degrees[idx];
        auto x = sqrt(static_cast<double>(deg));

        thrust::default_random_engine rng(thread_seed);
        thrust::normal_distribution<frac_t> dist;
        thrust::normal_distribution<>::param_type stdev(0.0, x);

        return dist(rng, stdev);
    }
};

__device__
inline uint64_t packKey(const uint32_t label, const float v) {
    const uint32_t i = __float_as_uint(v);

    const uint32_t mask = (i & 0x80000000) ? 0xffffffff : 0x80000000;
    const uint32_t ordered = i ^ mask;

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


__global__
void fusedRandomWalk_WarpParallel(
    NodeIx numNodes,
    const LabeledNode* __restrict__ activeNodes,
    const NodeIx* __restrict__ neighbors,
    const EdgeIx* __restrict__ ranges,
    const EdgeIx* __restrict__ degrees,
    const EdgeIx* __restrict__ clusterDegrees,
    const double* __restrict__ dist,
    double* __restrict__ new_dist,
    uint64_t* __restrict__ packedKeys
) {
    const size_t warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP;
    const size_t lane   = threadIdx.x & 31;
    if (warpId >= numNodes) return;

    NodeIx nix = 0;
    int64_t myLabel = 0;
    EdgeIx start = 0, degree = 0;

    // Only Lane 0 issues the memory requests to the uniform addresses
    if (lane == 0) {
        const LabeledNode& lNode = activeNodes[warpId];
        nix     = lNode.nix;
        myLabel = lNode.clusterId;
        start   = __ldg(&ranges[nix]);
        degree  = __ldg(&degrees[nix]);
    }

    // Broadcast the uniform values to all lanes in 1 clock cycle
    nix     = __shfl_sync(0xffffffff, nix, 0);
    myLabel = __shfl_sync(0xffffffff, myLabel, 0);
    start   = __shfl_sync(0xffffffff, start, 0);
    degree  = __shfl_sync(0xffffffff, degree, 0);

    const EdgeIx end = start + degree;

    if (myLabel < 0) {
        // These are the nodes that were just deactivated in the last iteration
        packedKeys[warpId] = packKey(myLabel, 0);
        return;
    }

    double local_sum = 0.0;
    for (EdgeIx j = start + lane; j < end; j += WARP) {
        const NodeIx nb = __ldg(&neighbors[j]);

        if (nb == INVALID_EDGE) continue; // this skips edges between different clusters

        const EdgeIx nbDeg = __ldg(&clusterDegrees[nb]);
        if (nbDeg <= 0) {
            printf("ERROR: degree should never be 0!!\t[nix = %d] edge from %d [eix=%d]\n", nb, nix, j);
            continue;
        }
        local_sum += __ldg(&dist[nb]) / static_cast<double>(nbDeg);
    }

    // 3. Warp Reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    // 4. Finalize and Write-back (Only Lane 0)
    if (lane == 0) {
        const double oldVal = dist[nix];
        const double nodeVal = (local_sum * (1.0 - rw_stay)) + (oldVal * rw_stay);

        // Update all data structures in one pass
        new_dist[nix] = nodeVal;
        // nodeData[warpId].activeDegree = __ldg(&activeDegrees[nix]);
        // nodeData[warpId].label = myLabel;
        // nodeData[warpId].nix = nix;
        // nodeData[warpId].rwValue = nodeValF;

        // if (myLabel > 4294967295) {
        //     printf("ERROR: label-overflow! We have %lld\n", myLabel);
        // }

        // Prepare for Sweep Cut sort
        // TODO: needs that myLabel fits in a uint32_t, i.e. is less than 4.294.967.295
        packedKeys[warpId] = packKey(myLabel, static_cast<float>(nodeVal));
    }
}


// struct LabelExtractorRW {
//     __host__ __device__
//     int64_t operator()(const NodeData& data) const {
//         return data.label;
//     }
// };


// struct ClusterDataExtractorRW {
//     __device__
//     inline ClusterData operator()(const NodeData& nodeData) const {
//         const float rwValue = nodeData.rwValue;
//         return {rwValue, rwValue, rwValue, 1};
//     }
// };


//
// struct WalkEdgeLogic {
//     const NodeIx* __restrict__ neighbors;
//     const EdgeIx* __restrict__ activeDegrees;
//     const double* __restrict__ dist;
//
//     __device__ __forceinline__
//     double operator()(EdgeIx edgeIdx) const {
//         const NodeIx tgtNode = __ldg(&neighbors[edgeIdx]);
//         if (tgtNode == INVALID_EDGE) {
//             // inactive edge
//             return 0.0;
//         }
//
//         const EdgeIx nbDeg = __ldg(&activeDegrees[tgtNode]);
//
//         if (nbDeg > 0) {
//             return __ldg(&dist[tgtNode]) / static_cast<float>(nbDeg);
//         }
//         return 0.0f;
//     }
// };
//
// __global__
// void finalizeRandomWalk(
//     NodeIx numNodes,
//     const EdgeIx* __restrict__ activeDegrees,
//     const double* __restrict__ incoming_sums,
//     const int* __restrict__ labels,
//     double* __restrict__ dist,
//     NodeData* __restrict__ nodeData,
//     uint64_t* __restrict__ packedKeys
// ) {
//     NodeIx i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= numNodes) return;
//
//     const int label = __ldg(&labels[i]);
//     if (label < 0) {
//         nodeData[i].nix = i;
//         packedKeys[i] = packKey(label, 0);
//         return;
//     }
//
//     const double nodeVal = (incoming_sums[i] * (1.0f - rw_stay)) + (dist[i] * rw_stay);
//     const float nodeValF = static_cast<float>(nodeVal);
//
//     dist[i] = nodeVal;
//     nodeData[i].activeDegree = activeDegrees[i];
//     nodeData[i].label = label;
//     nodeData[i].nix = i;
//     nodeData[i].rwValue = nodeValF;
//     packedKeys[i] = packKey(label, nodeValF);
// }



class RandomWalkManager {
    thrust::device_vector<double> dist1;
    thrust::device_vector<double> dist2;
    cub::DoubleBuffer<double> dist;

    NodeIx numNodes;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    // int maxLabel{0};

public:
    explicit RandomWalkManager(GraphManager& gm) :
        dist1(gm.n), dist2(gm.n),
        dist(thrust::raw_pointer_cast(dist1.data()),
               thrust::raw_pointer_cast(dist2.data())),
        numNodes(gm.n)
    {
        initRandomWalk(gm, randSeed);
        // prepare_cub();
    }

    ~RandomWalkManager() {
        if (d_temp_storage) cudaFree(d_temp_storage);
    }

    double* getValues() {
        return dist.Current();
    }

    void stepFast(
        GraphManager& gm,
        thrust::device_vector<LabeledNode>& nodes,
        cub::DoubleBuffer<uint64_t>& packedKeys,
        thrust::device_vector<EdgeIx>& allInternalDegrees,
        NodeIx numActiveNodes
    ) {
        const size_t gridSize = getGridSize(numActiveNodes);

        fusedRandomWalk_WarpParallel<<<gridSize, threads>>>(
            numActiveNodes,
            thrust::raw_pointer_cast(nodes.data()),
            thrust::raw_pointer_cast(gm.getNeighbors().data()),
            thrust::raw_pointer_cast(gm.getRanges().data()),
            thrust::raw_pointer_cast(gm.getDegrees().data()),
            thrust::raw_pointer_cast(allInternalDegrees.data()),
            dist.Current(),
            dist.Alternate(),
            packedKeys.Current()
        );
        // fflush(stdout);

        dist.selector ^= 1;
    }



    // void stepFast(GraphManager& gm,
    //       cub::DoubleBuffer<NodeData>& partition,
    //       cub::DoubleBuffer<uint64_t>& packedKeys,
    //       thrust::device_vector<int>& nodeLabels,
    //       thrust::device_vector<EdgeIx>& activeDegrees
    // ) {
    //     const EdgeIx* rangesPtr = thrust::raw_pointer_cast(gm.getRanges().data());
    //     const NodeIx* neighborsPtr = thrust::raw_pointer_cast(gm.getNeighbors().data());
    //     const EdgeIx* activeDegsPtr = thrust::raw_pointer_cast(activeDegrees.data());
    //     double* distPtr = thrust::raw_pointer_cast(dist.data());
    //     double* incomingSumsPtr = thrust::raw_pointer_cast(incomingSums.data());
    //     const int* labelsPtr = thrust::raw_pointer_cast(nodeLabels.data());
    //
    //     thrust::counting_iterator<EdgeIx> edgeIndexIter(0);
    //     WalkEdgeLogic functor{neighborsPtr, activeDegsPtr, distPtr};
    //
    //     auto transIter = thrust::make_transform_iterator(edgeIndexIter, functor);
    //
    //     // perform the reduction
    //     cub::DeviceSegmentedReduce::Sum(
    //         d_temp_storage, temp_storage_bytes,
    //         transIter, incomingSumsPtr,
    //         static_cast<int>(numNodes), rangesPtr, rangesPtr + 1
    //     );
    //
    //     // Finalize
    //     int gridSize = (numNodes + threads - 1) / threads;
    //
    //     finalizeRandomWalk<<<gridSize, threads, 0, nullptr>>>(
    //         numNodes,
    //         activeDegsPtr,
    //         incomingSumsPtr,
    //         labelsPtr,
    //         distPtr,
    //         partition.Current(),
    //         packedKeys.Current()
    //     );
    //
    //     // cudaDeviceSynchronize();
    //     //
    //     // printf("Partition after finalize : %p\n", partition.Current());
    //     //
    //     // std::vector<NodeData> nodes(numNodes);
    //     // thrust::device_ptr<NodeData> dev_ptr(partition.Current());
    //     // thrust::copy(dev_ptr, dev_ptr + numNodes, nodes.begin());
    //     // for (NodeIx i = 0; i < numNodes; i++) {
    //     //     printf("%d: nix = %d, label: %d\n", i, nodes[i].nix, nodes[i].label);
    //     // }
    // }



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

    std::vector<frac_t> valuesToCPU() {
        std::vector<frac_t> rwVals(numNodes);
        auto dev_ptr_start = thrust::device_pointer_cast(dist.Current());
        auto dev_ptr_end   = dev_ptr_start + numNodes;
        thrust::copy(dev_ptr_start, dev_ptr_end, rwVals.begin());
        return rwVals;
    }

private:
    // void prepare_cub() {
    //     WalkEdgeLogic dryRunFunctor{nullptr, nullptr, nullptr};
    //     auto dryRunIter = thrust::make_transform_iterator(thrust::make_counting_iterator<EdgeIx>(0), dryRunFunctor);
    //
    //     EdgeIx* nullOffsets = static_cast<EdgeIx*>(nullptr);
    //     frac_t* nullOutput  = static_cast<frac_t*>(nullptr);
    //
    //     // Dry run to get temp_storage_bytes
    //     cudaError_t err = cub::DeviceSegmentedReduce::Sum(
    //         nullptr, temp_storage_bytes,
    //         dryRunIter, nullOutput,
    //         static_cast<int>(numNodes), nullOffsets, nullOffsets
    //     );
    //
    //     if (err != cudaSuccess) {
    //         printf("CUB Error: %s\n", cudaGetErrorString(err));
    //     }
    //
    //     cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // }

    void initRandomWalk(GraphManager& gm, unsigned int s) {
        const NodeIx* degreesPtr = thrust::raw_pointer_cast(gm.getDegrees().data());

        thrust::transform(
            thrust::device,
            thrust::make_counting_iterator<NodeIx>(0),
              thrust::make_counting_iterator(numNodes),
              dist.Current(),
              NormalDistributionFunctor(s, degreesPtr)
        );
    }
};

#endif //PAREX_DEVRANDOMWALK_H
