//
// Created by robin on 19.01.2026.
//

#ifndef PAREX_DEVRANDOMWALK_H
#define PAREX_DEVRANDOMWALK_H

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/iterator/permutation_iterator.h>

#include <curand_kernel.h>

#include <cub/cub.cuh>
#include <cassert>

#include "devGraph.h"


const unsigned int seed{6};
const frac_t rw_stay = 0.1;

struct NormalDistributionFunctor {
    unsigned int base_seed;

    explicit NormalDistributionFunctor(unsigned int s) : base_seed(s) {}

    __host__ __device__
    frac_t operator()(const NodeIx idx) const {
        thrust::default_random_engine rng(base_seed);
        thrust::normal_distribution<frac_t> dist;
        rng.discard(idx);
        return dist(rng);
//        return static_cast<frac_t>(idx) / 4096;
    }
};


__device__
inline uint64_t packKey(NodeIx label, float v) {
    uint32_t i = __float_as_uint(v);

    uint32_t mask = (i & 0x80000000) ? 0xffffffff : 0x80000000;
    uint32_t ordered = i ^ mask;

    // Shift label to high bits, place ordered float in low bits
    return (static_cast<uint64_t>(label) << 32) | ordered;
}

__global__
void lazyRandomWalkKernel(
        NodeIx numNodes,
        const NodeIx* __restrict__ neighbors,
        const NodeData* __restrict__ nodeData,
        const EdgeIx* __restrict__ activeDegrees,
        const frac_t* __restrict__ old_dist,
        frac_t* __restrict__ dist,
        uint64_t* __restrict__ packedKeys,
        frac_t stay_weight
) {
    NodeIx i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) return;

    // early exit for inactive nodes
    const NodeData data = nodeData[i];
//    if(data.label < 0) return; // label < 0 considered inactive, but label is currently unsigned

    assert(data.nix == i && "nix mismatch!");

    frac_t incoming_sum = 0.0f;
    const EdgeIx rangeEnd = data.rangeStart + data.degree;

    for (NodeIx j = data.rangeStart; j < rangeEnd; ++j) {
        const NodeIx neighbor = __ldg(&neighbors[j]);
        const NodeData nbData = nodeData[neighbor];
        const EdgeIx nbDeg = __ldg(&activeDegrees[neighbor]);

        if(nbData.label == data.label && nbDeg != 0) {
            // inactive nodes have deg == 0
            incoming_sum += (__ldg(&old_dist[neighbor]) / static_cast<frac_t>(nbDeg));
        }
    }

    const frac_t nodeVal = (incoming_sum * (1.0f - stay_weight)) + (old_dist[i] * stay_weight);

    dist[i] = nodeVal;

    // FOR SWEEP CUT:
    // Already Pack label and rw value into packedKeys for sorting!!
    packedKeys[i] = packKey(data.label, nodeVal);
}

class RandomWalkManager {
    thrust::device_vector<frac_t> dist;
    thrust::device_vector<frac_t> old_dist;
    thrust::device_vector<frac_t> node_val;

    NodeIx numNodes;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

public:
    explicit RandomWalkManager(NodeIx n) : dist(n), old_dist(n), node_val(n), numNodes(n) {
        initRandomWalk(seed);
//        prepare_cub(gr, static_cast<int>(n));
    }

    ~RandomWalkManager() {
        if (d_temp_storage) cudaFree(d_temp_storage);
    }

    void step(GraphManager& gm,
          cub::DoubleBuffer<NodeData> partition,
          cub::DoubleBuffer<uint64_t> packedKeys,
          thrust::device_vector<EdgeIx>& activeDegrees,
          cudaStream_t stream = nullptr
    ) {
        old_dist.swap(dist);

        unsigned int blocksPerGrid = (numNodes + threads - 1) / threads;

        lazyRandomWalkKernel<<<blocksPerGrid, threads, 0, stream>>>(
                numNodes,
                thrust::raw_pointer_cast(gm.getNeighbors().data()),
                partition.Current(),
                thrust::raw_pointer_cast(activeDegrees.data()),
                thrust::raw_pointer_cast(old_dist.data()),
                thrust::raw_pointer_cast(dist.data()),
                packedKeys.Current(),
                rw_stay
        );
    }


//    void newStep(GraphManager& gm,
//              cub::DoubleBuffer<NodeData> partition,
//              cub::DoubleBuffer<uint64_t> packedKeys,
//              cudaStream_t stream = nullptr
//    ) {
//
//        thrust::transform(thrust::cuda::par.on(stream),
//                          old_dist.begin(), old_dist.end(),
//                          gr.active_degrees,
//                          node_val.begin(),
//            [move_weight] __device__ (frac_t d, EdgeIx deg) {
//                return (deg > 0) ? (d / static_cast<frac_t>(deg)) * move_weight : 0.0f;
//            }
//        );
//
//    }



//    void stepSlow(GraphManager& gm, cudaStream_t stream = nullptr) {
//        old_dist.swap(dist);
//
//        const frac_t stay_weight = rw_stay;
//        const frac_t move_weight = 1.0f - rw_stay;
//
//        // node_val = (old_dist / degree) * move_weight
//        thrust::transform(thrust::cuda::par.on(stream),
//                          old_dist.begin(), old_dist.end(),
//                          gr.active_degrees,
//                          node_val.begin(),
//            [move_weight] __device__ (frac_t d, EdgeIx deg) {
//                return (deg > 0) ? (d / static_cast<frac_t>(deg)) * move_weight : 0.0f;
//            }
//        );
//
//        frac_t* raw_node_vals = thrust::raw_pointer_cast(node_val.data());
//        frac_t* raw_dist_out = thrust::raw_pointer_cast(dist.data());
//        NodeIx* raw_ranges = thrust::raw_pointer_cast(gm.getRanges().data());
//
//        auto v_mapped_iter = thrust::make_permutation_iterator(raw_node_vals, gr.neighbors);
//
//        // dist = sum_neighbors node_val
//        cub::DeviceSegmentedReduce::Sum(
//                d_temp_storage,
//                temp_storage_bytes,
//                v_mapped_iter,
//                raw_dist_out,
//                static_cast<int>(numNodes),
//                raw_ranges,
//                raw_ranges + 1,
//                stream
//        );
//
//        // dist += (1 - rw_stay) * old_dist
//        thrust::transform(thrust::cuda::par.on(stream),
//                          dist.begin(), dist.end(),
//                          old_dist.begin(),
//                          dist.begin(),
//            [stay_weight] __device__ (frac_t summed_neighbors, frac_t self_old) {
//                return summed_neighbors + (self_old * stay_weight);
//            }
//        );
//    }

    [[nodiscard]] const thrust::device_vector<frac_t>& randomWalkValues() const {
        return dist;
    }

    std::vector<frac_t> valuesToCPU() {
        std::vector<frac_t> rwVals(numNodes);
        thrust::copy(dist.begin(), dist.end(), rwVals.begin());
        return rwVals;
    }

private:
//    void prepare_cub(DevGraph gr, int n) {
//        frac_t* raw_node_vals = thrust::raw_pointer_cast(node_val.data());
//        frac_t* raw_dist_out = thrust::raw_pointer_cast(dist.data());
//
//        thrust::permutation_iterator<frac_t*, const NodeIx*> v_mapped_iter = thrust::make_permutation_iterator(raw_node_vals, gr.neighbors);
//
//        cudaError_t err = cub::DeviceSegmentedReduce::Sum(
//                nullptr,
//                temp_storage_bytes,
//                v_mapped_iter,
//                raw_dist_out,
//                n,
//                gr.ranges,
//                gr.ranges + 1
//        );
//
//        if (err != cudaSuccess) {
//            printf("CUB Error: %s\n", cudaGetErrorString(err));
//        }
//
//        std::cout << "allocating " << temp_storage_bytes << "B for storage" << std::endl;
//
//        cudaMalloc(&d_temp_storage, temp_storage_bytes);
//    }

    void initRandomWalk(unsigned int s) {
        thrust::transform(thrust::make_counting_iterator<NodeIx>(0),
                          thrust::make_counting_iterator(numNodes),
                          dist.begin(),
                          NormalDistributionFunctor(s)
        );
    }
};

#endif //PAREX_DEVRANDOMWALK_H
