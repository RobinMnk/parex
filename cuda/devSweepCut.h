//
// Created by robin on 21.01.2026.
//

#ifndef PAREX_DEVSWEEPCUT_H
#define PAREX_DEVSWEEPCUT_H

#include "devGraph.h"
#include "devPartition.h"

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/iterator/zip_iterator.h>

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
        const NodeIx* __restrict__ neighbors,
        const NodeData* __restrict__ nodeData,
        const frac_t* __restrict__ values,
        frac_t* __restrict__ contributions
) {
    NodeIx i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) return;

    // early exit for inactive nodes
    const NodeData data = nodeData[i];
    if(data.activeDegree == 0) return;

    const frac_t myVal = __ldg(&values[i]);

    frac_t nodeContribution = 0.0f;
    const EdgeIx rangeEnd = data.rangeStart + data.activeDegree;

    for (NodeIx j = data.rangeStart; j < rangeEnd; ++j) {
        const NodeIx neighbor = __ldg(&neighbors[j]);
        const NodeData nbData = nodeData[neighbor];         // this has to fetch 32 Bytes, even though we only need 4

        if (nbData.label == data.label) {
            // only consider edges that stay within the same cluster
            const frac_t otherVal = __ldg(&values[neighbor]);
            // if otherVal < myVal -> then the edge goes "left" in the sweep cut
            nodeContribution += (otherVal < myVal) ? -1.0f : 1.0f; // or the edge weight if weighted
        }
    }

    contributions[i] = nodeContribution;
}
// This need all random walk values done -> has to be computed after!


struct PrefixValues {
    EdgeIx edgeDiff;
    EdgeIx volume;

    __device__ PrefixValues operator+(const PrefixValues& other) const {
        return {edgeDiff + other.edgeDiff, volume + other.volume};
    }
};

struct SweepCutData {
    float sparsity;
    int offset;
};


class SweepCutManager {
    thrust::device_vector<frac_t> nodeContributions;

    // Buffers
    thrust::device_vector<uint64_t> packedKeysIn;
    thrust::device_vector<uint64_t> packedKeysOut;
    cub::DoubleBuffer<uint64_t> packedKeys;


    thrust::device_vector<PrefixValues> prefixSums;


    // CUB Buffers
    size_t temp_storage_bytes = 0;
    void *d_temp_storage = nullptr;


    // Output
    thrust::device_vector<NodeIx> d_unique_labels;
    thrust::device_vector<SweepCutData> sweepCuts;


    void prepareNodeContributions(
            GraphManager &gm,
            const thrust::device_vector<frac_t> &values,
            PartitionManager& pm,
            cudaStream_t stream = nullptr
    ) {
        unsigned int blocksPerGrid = (gm.n + threads - 1) / threads;

        nodeDiffKernel_Sparse<<<blocksPerGrid, threads, 0, stream>>>(
                gm.n,
                thrust::raw_pointer_cast(gm.getNeighbors().data()),
                pm.getPartition(),
                thrust::raw_pointer_cast(values.data()),
                thrust::raw_pointer_cast(nodeContributions.data())
        );
    }

    void initializeCUB(NodeIx numNodes, cub::DoubleBuffer<NodeData> partitionView) {
        cub::DeviceRadixSort::SortPairs(
            nullptr, temp_storage_bytes,
            packedKeys,
            partitionView,
            static_cast<int>(numNodes)
        );

        cudaMalloc(&d_temp_storage, temp_storage_bytes);
    }

public:
    explicit SweepCutManager(NodeIx n, cub::DoubleBuffer<NodeData> partitionView) :
        nodeContributions(n),
        packedKeysIn(n),
        packedKeysOut(n),
        packedKeys(thrust::raw_pointer_cast(packedKeysIn.data()),
                   thrust::raw_pointer_cast(packedKeysOut.data())),
        prefixSums(n),
        d_unique_labels(n),
        sweepCuts(n)
    {
        initializeCUB(n, partitionView);
    }


    void solve(GraphManager& gm, PartitionManager& pm);

    void compute(
        GraphManager& gm,
        const thrust::device_vector<frac_t> &values,
        PartitionManager& pm,
        cudaStream_t stream = nullptr
     ) {
        prepareNodeContributions(gm, values, pm, stream);
        cudaStreamSynchronize(stream);
        solve(gm, pm);
    }

    AllSweepCuts resultToCPU(NodeIx numClusters) {
        std::vector<NodeIx> clusterIds(numClusters);
        std::vector<NodeIx> offsets(numClusters);
        std::vector<float> sparsities(numClusters);
        thrust::copy(d_unique_labels.begin(), d_unique_labels.begin() + numClusters, clusterIds.begin());
//        thrust::copy(d_min_indices.begin(), d_min_indices.begin() + numClusters, offsets.begin());
//        thrust::copy(d_min_ratios.begin(), d_min_ratios.begin() + numClusters, sparsities.begin());
        return {clusterIds, offsets, sparsities};
    }

    uint64_t* getKeyBuffer() {
        return packedKeys.Current();
    }

};


struct LabelExtractor {
    __device__
    inline NodeIx operator()(uint64_t k) const {
        return static_cast<NodeIx>(k >> 32);
    }
};

struct ArgMinOp {
    __host__ __device__
    SweepCutData operator()(const SweepCutData& a, const SweepCutData& b) const {
        return (a.sparsity < b.sparsity) ? a : b;
    }
};

void SweepCutManager::solve(GraphManager& gm, PartitionManager& pm) {

    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        packedKeys,
        pm.getPartitionView(),
        static_cast<int>(gm.n)
    );

    NodeData* sortedData = pm.getPartition();
    uint64_t* sortedKeys = packedKeys.Current();

    /**
     * - Extract edgeDiff and activeDegree from every node
     * - Perform separate prefix sum on both
     * - save outcome to prefixSums
     */
    thrust::inclusive_scan_by_key(
        thrust::device,
        // Keys
        thrust::make_transform_iterator(sortedKeys, LabelExtractor()),
        thrust::make_transform_iterator(sortedKeys + gm.n, LabelExtractor()),
        // Values (Input)
        thrust::make_transform_iterator(sortedData, [] __device__(const NodeData& n) -> PrefixValues {
            return {n.edgeDiff, n.activeDegree};
        }),
        // Output
        prefixSums.begin(),
        thrust::equal_to<NodeIx>(),
        thrust::plus<PrefixValues>()
    );


    // 1. Get raw pointers for the lambda capture
    EdgeIx* clusterVolumesPtr       = thrust::raw_pointer_cast(pm.getVolumes().data());
    PrefixValues* prefixSumsPtr     = thrust::raw_pointer_cast(prefixSums.data());

    // 2. Define the Conductance + Indexing logic in a single iterator
    auto conductance_index_iter = thrust::make_transform_iterator(
            thrust::make_counting_iterator<int>(0),
            [clusterVolumesPtr, sortedKeys, prefixSumsPtr] __device__ (int i) -> SweepCutData {
                const uint64_t key = sortedKeys[i];
                const NodeIx label = static_cast<NodeIx>(key >> 32);
                const EdgeIx totalVol  = clusterVolumesPtr[label];

                const PrefixValues pv = prefixSumsPtr[i];
                const EdgeIx edgeDiff = pv.edgeDiff;
                const EdgeIx prefixVol = pv.volume;

                // min(vol, totalVol - vol)
                const EdgeIx denom = (prefixVol < totalVol - prefixVol) ? prefixVol : (totalVol - prefixVol);
                const float sparsity  = (denom > 0) ? (static_cast<float>(edgeDiff) / static_cast<float>(denom)) : 1e30f;

                return { sparsity , i };
            }
    );

// 3. One-pass Reduction to find the best cut per label
    thrust::reduce_by_key(
            thrust::device,
            // Keys
            thrust::make_transform_iterator(sortedKeys, LabelExtractor()),
            thrust::make_transform_iterator(sortedKeys + gm.n, LabelExtractor()),
            // Values
            conductance_index_iter,
            // Output Label
            d_unique_labels.begin(),
            // Output Values
            sweepCuts.begin(),
            thrust::equal_to<NodeIx>(),
            ArgMinOp()
    );
}



//    thrust::inclusive_scan_by_key(
//            thrust::device,
//        thrust::make_transform_iterator(sortedKeys, LabelExtractor()),
//        thrust::make_transform_iterator(sortedKeys + gm.n, LabelExtractor()),
//        thrust::make_transform_iterator(sortedData, [] __device__(const NodeData& n) { return n.edgeDiff; }),
//        prefixContributions.begin()
//    );
//
//    thrust::inclusive_scan_by_key(
//            thrust::device,
//            thrust::make_transform_iterator(sortedKeys, LabelExtractor()),
//            thrust::make_transform_iterator(sortedKeys + gm.n, LabelExtractor()),
//            thrust::make_transform_iterator(sortedData, [] __device__(const NodeData& n) { return n.activeDegree; }),
//            prefixVolumes.begin()
//    );

    // 4. Final Conductance & ArgMin
    // Use the same reduction logic from your original code,
    // but reading from d_prefix_weights and d_prefix_volumes
//    computeConductanceAndReduce(gm, sortedKeys);


//    void inspect(std::vector<EdgeIx>& pS, std::vector<EdgeIx>& s) {
//        prefixSums = pS;
//        cutVolumes = s;
//    }
//
//    void checkDegrees(GraphManager& gm) {
//        std::vector<float> comp(gm.n);
//        thrust::copy(d_sweepCuts.begin(), d_sweepCuts.end(), comp.begin());
//
//        std::vector<float> pw(gm.n);
//        thrust::copy(d_prefix_weights.begin(), d_prefix_weights.end(), pw.begin());
//
//        std::vector<float> deg(gm.n);
//        thrust::copy(d_volumes.begin(), d_volumes.end(), deg.begin());
//
//        for(int i = 0; i < 20; i++) {
//            std::cout << i << ":  CPU = " << prefixSums[i] << " / " << cutVolumes[i] << " = " << (prefixSums[i] / cutVolumes[i])<< " \tGPU = " << pw[i] << " / " << deg[i] << " = " << comp[i]  << "\n";
//        }
//        std::cout << ((sparsities == comp) ? " Ratios fine." : "ERROR: Ratios BROKEN!!!") << std::endl;
//    }
//};

//void SweepCutManager::solve(GraphManager& gm, const thrust::device_vector<frac_t>& values) {
//    thrust::sequence(d_indices.begin(), d_indices.end());
//
//    // 1. Prepare Keys
//    thrust::transform(gm.getLabels().begin(), gm.getLabels().end(), values.begin(), d_packed_keys.begin(),
//                      [] __device__ (NodeIx l, float v) {
//                          return ((uint64_t)l << 32) | (uint64_t)floatToOrderedInt(v);
//                      });
//
//    // 2. Prepare Data for ONE SINGLE SORT
//    // Copy original degrees into d_volumes BEFORE sorting
//    thrust::copy(gm.getActiveDegrees().begin(), gm.getActiveDegrees().end(), d_volumes.begin());
//
//    // Zip everything that needs to stay synchronized with the keys
//    auto begin_data = thrust::make_zip_iterator(thrust::make_tuple(
//            nodeContributions.begin(),
//            d_volumes.begin(),
//            d_indices.begin())
//    );
//
//    // SORT ONCE - This moves everything in the tuple to match the new key order
//    thrust::sort_by_key(d_packed_keys.begin(), d_packed_keys.end(), begin_data);
//
//    // 3. Extract sorted labels from the now-sorted keys
//    thrust::transform(d_packed_keys.begin(), d_packed_keys.end(), d_sorted_labels.begin(),
//                      [] __device__ (uint64_t key) { return (NodeIx)(key >> 32); });
//
//    // 4. Prefix Sums (Both are now aligned with d_sorted_labels)
//
//    // Accumulate Cut Edges (Scan nodeContributions into d_prefix_weights)
//    thrust::inclusive_scan_by_key(d_sorted_labels.begin(), d_sorted_labels.end(),
//                                  nodeContributions.begin(), d_prefix_weights.begin());
//
//    // Accumulate Volumes (Scan d_volumes in-place)
//    thrust::inclusive_scan_by_key(d_sorted_labels.begin(), d_sorted_labels.end(),
//                                  d_volumes.begin(), d_volumes.begin());
//
//    // 5. Final Conductance Calculation
//    EdgeIx* clusterVolumesPtr = thrust::raw_pointer_cast(gm.getVolumes().data());
//
//    thrust::transform(d_prefix_weights.begin(), d_prefix_weights.end(), // Prefix Cut
//                      thrust::make_zip_iterator(thrust::make_tuple(d_volumes.begin(), d_sorted_labels.begin())), // Prefix Vol + Label
//                      d_sweepCuts.begin(), // Result Ratio
//                        [clusterVolumesPtr] __device__ (float cutSize, thrust::tuple<EdgeIx, NodeIx> t) {
//                                EdgeIx prefixVol = thrust::get<0>(t);
//                                NodeIx clusterId = thrust::get<1>(t);
//                                EdgeIx totalVol  = clusterVolumesPtr[clusterId];
//
//                                // Denominator: min(vol, totalVol - vol)
//                                EdgeIx denom = (prefixVol < totalVol - prefixVol) ? prefixVol : (totalVol - prefixVol);
//
//                                return (denom > 0) ? (cutSize / (float)denom) : 1e30f;
//                            }
//    );
//
//
//    auto ratio_index_begin = thrust::make_zip_iterator(thrust::make_tuple(d_sweepCuts.begin(), d_indices.begin()));
//    auto output_begin = thrust::make_zip_iterator(thrust::make_tuple(d_min_ratios.begin(), d_min_indices.begin()));
//
//    thrust::reduce_by_key(
//            d_sorted_labels.begin(), d_sorted_labels.end(), // Keys
//            ratio_index_begin,                             // Values (Ratio, Index)
//            d_unique_labels.begin(),                       // Output Keys
//            output_begin,                                  // Output Values
//            thrust::equal_to<NodeIx>(),              // Key Binary Predicate
//            ArgMinOp()                                     // Reduction Operator
//    );
//}



#endif //PAREX_DEVSWEEPCUT_H
