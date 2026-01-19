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


#include "devGraph.h"


const unsigned int seed{0};
const frac_t rw_stay = 0.1;

struct NormalDistributionFunctor {
    unsigned int base_seed;

    // Use a constructor to pass a real-time seed
    explicit NormalDistributionFunctor(unsigned int s) : base_seed(s) {}

    __host__ __device__
    frac_t operator()(const NodeIx idx) const {
        thrust::default_random_engine rng(base_seed);
        thrust::normal_distribution<frac_t> dist;
        rng.discard(idx);
        return dist(rng);
    }
};

__global__
void lazyRandomWalkKernel(
        NodeIx numNodes,
        const NodeIx* __restrict__ ranges,
        const NodeIx* __restrict__ neighbors,
        const EdgeIx* __restrict__ degrees,
        const frac_t* __restrict__ old_dist,
        frac_t* __restrict__ dist,
        frac_t stay_weight,
        frac_t move_weight)
{
    NodeIx i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) return;

    // Prefetch ranges to registers
    const NodeIx start = ranges[i];
    const NodeIx end   = ranges[i+1];

    frac_t incoming_sum = 0.0f;

    // The #pragma unroll hint tells the compiler to optimize the loop
    // for small segments, which are common in many graphs.
#pragma unroll 4
    for (NodeIx j = start; j < end; ++j) {
        const NodeIx neighbor = neighbors[j];
        const EdgeIx deg = degrees[neighbor];
        incoming_sum += (__ldg(&old_dist[neighbor]) / static_cast<frac_t>(deg));
    }

    dist[i] = (incoming_sum * move_weight) + (old_dist[i] * stay_weight);
}

class RandomWalkManager {
    thrust::device_vector<frac_t> dist;
    thrust::device_vector<frac_t> old_dist;
    thrust::device_vector<frac_t> node_val;

    NodeIx numNodes;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

public:
    explicit RandomWalkManager(DevGraph gr, NodeIx n) : dist(n), old_dist(n), node_val(n), numNodes(n) {
        initRandomWalk(seed);
        prepare_cub(gr, static_cast<int>(n));
    }

    ~RandomWalkManager() {
        if (d_temp_storage) cudaFree(d_temp_storage);
    }

    void step(const DevGraph gr, cudaStream_t stream = nullptr) {
        old_dist.swap(dist);

        int threadsPerBlock = 256;
        unsigned int blocksPerGrid = (numNodes + threadsPerBlock - 1) / threadsPerBlock;

        lazyRandomWalkKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                numNodes,
                gr.ranges,
                gr.neighbors,
                gr.active_degrees,
                thrust::raw_pointer_cast(old_dist.data()),
                thrust::raw_pointer_cast(dist.data()),
                rw_stay,
                1.0f - rw_stay
        );
    }

    void stepSlow(DevGraph gr, cudaStream_t stream = nullptr) {
        old_dist.swap(dist);

        const frac_t stay_weight = rw_stay;
        const frac_t move_weight = 1.0 - rw_stay;

        // node_val = (old_dist / degree) * move_weight
        thrust::transform(thrust::cuda::par.on(stream),
                          old_dist.begin(), old_dist.end(),
                          gr.active_degrees,
                          node_val.begin(),
            [move_weight] __device__ (frac_t d, EdgeIx deg) {
                return (deg > 0) ? (d / static_cast<frac_t>(deg)) * move_weight : 0.0f;
            }
        );

        frac_t* raw_node_vals = thrust::raw_pointer_cast(node_val.data());
        frac_t* raw_dist_out = thrust::raw_pointer_cast(dist.data());

        auto v_mapped_iter = thrust::make_permutation_iterator(raw_node_vals, gr.neighbors);

        // dist = sum_neighbors node_val
        cub::DeviceSegmentedReduce::Sum(
                d_temp_storage,
                temp_storage_bytes,
                v_mapped_iter,
                raw_dist_out,
                static_cast<int>(numNodes),
                gr.ranges,
                gr.ranges + 1,
                stream
        );

        // dist += (1 - rw_stay) * old_dist
        thrust::transform(thrust::cuda::par.on(stream),
                          dist.begin(), dist.end(),
                          old_dist.begin(),
                          dist.begin(),
            [stay_weight] __device__ (frac_t summed_neighbors, frac_t self_old) {
                return summed_neighbors + (self_old * stay_weight);
            }
        );
    }

    std::vector<frac_t> readRandomWalkValues() {
        std::vector<frac_t> rwVals(numNodes);
        thrust::copy(dist.begin(), dist.end(), rwVals.begin());
        return rwVals;
    }

private:
    void prepare_cub(DevGraph gr, int n) {
        frac_t* raw_node_vals = thrust::raw_pointer_cast(node_val.data());
        frac_t* raw_dist_out = thrust::raw_pointer_cast(dist.data());

        thrust::permutation_iterator<frac_t*, NodeIx*> v_mapped_iter = thrust::make_permutation_iterator(raw_node_vals, gr.neighbors);

        cudaError_t err = cub::DeviceSegmentedReduce::Sum(
                nullptr,
                temp_storage_bytes,
                v_mapped_iter,
                raw_dist_out,
                n,
                gr.ranges,
                gr.ranges + 1
        );

        if (err != cudaSuccess) {
            printf("CUB Error: %s\n", cudaGetErrorString(err));
        }

        std::cout << "allocating " << temp_storage_bytes << "B for storage" << std::endl;

        cudaMalloc(&d_temp_storage, temp_storage_bytes);
    }

    void initRandomWalk(unsigned int s) {
        thrust::transform(thrust::make_counting_iterator<NodeIx>(0),
                          thrust::make_counting_iterator(numNodes),
                          dist.begin(),
                          NormalDistributionFunctor(s)
        );
    }
};

#endif //PAREX_DEVRANDOMWALK_H
