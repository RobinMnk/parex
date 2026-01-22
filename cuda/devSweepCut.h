//
// Created by robin on 21.01.2026.
//

#ifndef PAREX_DEVSWEEPCUT_H
#define PAREX_DEVSWEEPCUT_H

#include "devGraph.h"

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/inner_product.h>


__global__
void nodeDiffKernel(
        NodeIx numNodes,
        const NodeIx* __restrict__ ranges,
        const NodeIx* __restrict__ neighbors,
        const NodeIx* __restrict__ labels,
        const frac_t* __restrict__ values,
        frac_t* __restrict__ contributions
) {
    const NodeIx i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    if (i >= numNodes) return;

    const NodeIx start = ranges[i];
    const NodeIx end = ranges[i+1];
    const NodeIx myLabel = labels[i];
    const frac_t myVal = values[i];

    frac_t localSum = 0.0f;

    // Warp-parallel loop over neighbors
    for (NodeIx j = start + (threadIdx.x % warpSize); j < end; j += warpSize) {
        const NodeIx neighbor = neighbors[j];

        // Load label first to potentially avoid second load
        if (labels[neighbor] == myLabel) {
            const frac_t otherVal = values[neighbor];
            localSum += (otherVal < myVal) ? -1.0f : 1.0f;
        }
    }

    // Parallel reduction within the warp
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        localSum += __shfl_down_sync(0xFFFFFFFF, localSum, offset);
    }

    // Lane 0 writes the final result
    if ((threadIdx.x % warpSize) == 0) {
        contributions[i] = localSum;
    }
}

__global__
void nodeDiffKernel_Sparse(
        NodeIx numNodes,
        const NodeIx* __restrict__ ranges,
        const NodeIx* __restrict__ neighbors,
        const NodeIx* __restrict__ labels,
        const frac_t* __restrict__ values,
        frac_t* __restrict__ contributions
) {
    NodeIx i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) return;

    const NodeIx start = __ldg(&ranges[i]);
    const NodeIx end   = __ldg(&ranges[i+1]);

    if (start == end) return; // Equivalent to degree == 0

    const NodeIx myLabel = __ldg(&labels[i]);
    const frac_t myVal   = __ldg(&values[i]);

    frac_t nodeContribution = 0.0f;

#pragma unroll 4
    for (NodeIx j = start; j < end; ++j) {
        const NodeIx neighbor = __ldg(&neighbors[j]);

        if (__ldg(&labels[neighbor]) == myLabel) {
            // only consider edges that stay within the same cluster
            const frac_t otherVal = __ldg(&values[neighbor]);
            // if otherVal < myVal -> then the edge goes "left" in the sweep cut
            nodeContribution += (otherVal < myVal) ? -1.0f : 1.0f; // or the edge weight if weighted
        }
    }

    contributions[i] = nodeContribution;
}

// Helper to convert float to a sortable integer (for Radix Sort)
__host__ __device__
uint32_t floatToOrderedInt(float v) {
    uint32_t i = *((uint32_t*)&v);
    uint32_t mask = (i >> 31 != 0) ? 0xffffffff : 0x80000000;
    return i ^ mask;
}

// Custom operator to find the minimum ratio and keep the associated index
struct ArgMinOp {
    typedef thrust::tuple<float, NodeIx> ValueIndex;
    __host__ __device__ ValueIndex operator()(const ValueIndex& a, const ValueIndex& b) const {
        // Return the tuple with the smaller ratio
        return (thrust::get<0>(a) < thrust::get<0>(b)) ? a : b;
    }
};

class SweepCutManager {
    thrust::device_vector<frac_t> nodeContributions;

    // Buffers
    thrust::device_vector<uint64_t> d_packed_keys;
    thrust::device_vector<NodeIx> d_indices;
    thrust::device_vector<NodeIx> d_sorted_labels;
    thrust::device_vector<float> d_prefix_weights;
    thrust::device_vector<float> d_sweepCuts;

    // Output
    thrust::device_vector<NodeIx> d_unique_labels;
    thrust::device_vector<NodeIx> d_min_indices;
    thrust::device_vector<float> d_min_ratios;

    void prepareNodeContributions(const DevGraph gr, const thrust::device_vector<frac_t>& values, cudaStream_t stream = nullptr) {
        unsigned int blocksPerGrid = (gr.numNodes + threads - 1) / threads;

        nodeDiffKernel_Sparse<<<blocksPerGrid, threads, 0, stream>>>(
                gr.numNodes,
                gr.ranges,
                gr.neighbors,
                gr.labels,
                thrust::raw_pointer_cast(values.data()),
                thrust::raw_pointer_cast(nodeContributions.data())
        );
    }

    void solve(GraphManager& gm, const thrust::device_vector<frac_t>& values);

public:
    explicit SweepCutManager(NodeIx n) : nodeContributions(n), d_packed_keys(n), d_indices(n), d_sorted_labels(n), d_prefix_weights(n), d_sweepCuts(n),
                                         d_unique_labels(n), d_min_indices(n), d_min_ratios(n)  {}

    void compute(GraphManager& gm, const thrust::device_vector<frac_t>& values, cudaStream_t stream = nullptr) {
        prepareNodeContributions(gm.getView(), values, stream);
        cudaStreamSynchronize(stream);
        solve(gm, values);
    }

    AllSweepCuts resultToCPU(NodeIx numClusters) {
        std::vector<NodeIx> clusterIds;
        std::vector<NodeIx> offsets;
        std::vector<float> sparsities;
        thrust::copy(d_unique_labels.begin(), d_unique_labels.begin() + numClusters, clusterIds.begin());
        thrust::copy(d_min_indices.begin(), d_min_indices.begin() + numClusters, offsets.begin());
        thrust::copy(d_min_ratios.begin(), d_min_ratios.begin() + numClusters, sparsities.begin());
        return { clusterIds, offsets, sparsities };
    }
};

void SweepCutManager::solve(GraphManager& gm, const thrust::device_vector<frac_t>& values) {

    // 1. Prepare Sorting Keys: Pack (Label, Value) into uint64
    // High 32 bits = Label, Low 32 bits = Sortable Float
//    thrust::device_vector<uint64_t> d_packed_keys(N);
//    thrust::device_vector<NodeIx> d_indices(N);

    thrust::sequence(d_indices.begin(), d_indices.end());

    thrust::transform(gm.getLabels().begin(), gm.getLabels().end(), values.begin(), d_packed_keys.begin(),
                      [] __device__ (NodeIx l, float v) {
                          return ((uint64_t)l << 32) | (uint64_t)floatToOrderedInt(v);
                      });

    // 2. Radix Sort by Packed Key
    // We sort the packed keys and the associated data (weight, dividend, original index)
    auto begin_data = thrust::make_zip_iterator(thrust::make_tuple(
            nodeContributions.begin(), gm.getActiveDegrees().begin(), d_indices.begin())
    );

    thrust::sort_by_key(d_packed_keys.begin(), d_packed_keys.end(), begin_data);

    // After sorting, d_packed_keys contains sorted labels in the high bits
//    thrust::device_vector<NodeIx> d_sorted_labels(N);
    thrust::transform(d_packed_keys.begin(), d_packed_keys.end(), d_sorted_labels.begin(),
                      [] __device__ (uint64_t key) { return (NodeIx)(key >> 32); });

    // 3. Segmented Prefix Sum of Weights
//    thrust::device_vector<float> d_prefix_weights(N);
    thrust::inclusive_scan_by_key(d_sorted_labels.begin(), d_sorted_labels.end(), nodeContributions.begin(), d_prefix_weights.begin());

    // 4. Calculate Ratios and keep original indices
    // We store these in a zip iterator for reduction
//    thrust::device_vector<float> d_ratios(N);
    thrust::transform(d_prefix_weights.begin(), d_prefix_weights.end(), gm.getActiveDegrees().begin(), d_sweepCuts.begin(),thrust::divides<float>());

    // 5. Segmented Reduction to find Min Ratio per Label
    // Each label can have multiple entries; we want the min ratio and its index
//    int num_labels = thrust::inner_product(d_sorted_labels.begin(), d_sorted_labels.end() - 1,
//                                           d_sorted_labels.begin() + 1, 1,
//                                           thrust::plus<int>(), thrust::not_equal_to<NodeIx>());


    auto ratio_index_begin = thrust::make_zip_iterator(thrust::make_tuple(d_sweepCuts.begin(), d_indices.begin()));
    auto output_begin = thrust::make_zip_iterator(thrust::make_tuple(d_min_ratios.begin(), d_min_indices.begin()));

    thrust::reduce_by_key(
            d_sorted_labels.begin(), d_sorted_labels.end(), // Keys
            ratio_index_begin,                             // Values (Ratio, Index)
            d_unique_labels.begin(),                       // Output Keys
            output_begin,                                  // Output Values
            thrust::equal_to<NodeIx>(),              // Key Binary Predicate
            ArgMinOp()                                     // Reduction Operator
    );

    // Result is now in d_unique_labels, d_min_ratios, and d_min_indices
}





#endif //PAREX_DEVSWEEPCUT_H
