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

#include <cassert>

__global__
void nodeDiffKernel(
        NodeIx numNodes,
        const NodeIx* __restrict__ ranges,
        const NodeIx* __restrict__ neighbors,
        const int* __restrict__ labels,
        const frac_t* __restrict__ values,
        frac_t* __restrict__ contributions
) {
    const NodeIx i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    if (i >= numNodes) return;

    const NodeIx start = ranges[i];
    const NodeIx end = ranges[i+1];
    const int myLabel = labels[i];
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
        NodeData* __restrict__ nodeData,
        const frac_t* __restrict__ values
) {
    NodeIx i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) return;

    // early exit for inactive nodes
    const NodeData data = nodeData[i];

    assert(data.nix == i && "nix mismatch!");

    const frac_t myVal = __ldg(&values[i]);

    int nodeContribution = 0;
    const EdgeIx rangeEnd = data.rangeStart + data.degree;

    for (NodeIx j = data.rangeStart; j < rangeEnd; ++j) {
        const NodeIx neighbor = __ldg(&neighbors[j]);
        const NodeData nbData = nodeData[neighbor];

        assert(nbData.nix == neighbor && "neighbor nix mismatch!");

        // only consider edges that stay within the same cluster
        if (nbData.label == data.label) {
            const frac_t otherVal = __ldg(&values[neighbor]);

            bool isBefore;
            if (otherVal != myVal) {
                isBefore = (otherVal < myVal);
            } else {
                // matching the stable sort
                isBefore = (neighbor < i);
            }

            nodeContribution += (isBefore) ? -1 : 1;
        }
    }

    // Initialize data fields for the sweep cut
    nodeData[i].prefixEdgeDiff = nodeContribution;
    nodeData[i].prefixVolume = data.activeDegree;
    nodeData[i].offsetInCluster = 1;
}
// This needs all random walk values done -> has to be computed after!


class SweepCutManager {
    NodeIx numNodes;
    int numActiveClusters;

    // Buffers
    thrust::device_vector<uint64_t> packedKeysIn;
    thrust::device_vector<uint64_t> packedKeysOut;
    cub::DoubleBuffer<uint64_t> packedKeys;

    // CUB Buffers
    size_t temp_storage_bytes = 0;
    void *d_temp_storage = nullptr;


    // Output
    thrust::device_vector<int> d_unique_labels;
    thrust::device_vector<SweepCutData> sweepCuts;

    void initializeCUB(NodeIx n, PartitionManager& pm) {
        cub::DeviceRadixSort::SortPairs(
            nullptr, temp_storage_bytes,
            packedKeys,
            pm.getPartitionView(),
            static_cast<int>(n)
        );

        cudaMalloc(&d_temp_storage, temp_storage_bytes);
    }

public:
    explicit SweepCutManager(NodeIx n, PartitionManager& pm) :
        numNodes(n),
        packedKeysIn(n),
        packedKeysOut(n),
        packedKeys(thrust::raw_pointer_cast(packedKeysIn.data()),
                   thrust::raw_pointer_cast(packedKeysOut.data())),
        d_unique_labels(n),
        sweepCuts(n)
    {
        initializeCUB(n, pm);
    }

    thrust::device_vector<SweepCutData>& getSweepCuts() {
        return sweepCuts;
    }

    int getNumActiveClusters() const {
        return numActiveClusters;
    }

    auto& getLabels() const {
        return d_unique_labels;
    }

    void compute(
        GraphManager& gm,
        PartitionManager& pm,
        const thrust::device_vector<frac_t> &values
    );

    AllSweepCuts resultToCPU(NodeIx numClusters) {
        std::vector<int> clusterIds(numClusters);
        std::vector<SweepCutData> cuts(numClusters);
        thrust::copy(d_unique_labels.begin(), d_unique_labels.begin() + numClusters, clusterIds.begin());
        thrust::copy(sweepCuts.begin(), sweepCuts.begin() + numClusters, cuts.begin());
        return {clusterIds, cuts};
    }

    cub::DoubleBuffer<uint64_t>& getKeyBuffer() {
        return packedKeys;
    }
};


struct LabelExtractor {
    __host__ __device__
    inline NodeIx operator()(uint64_t k) const {
        return static_cast<NodeIx>(k >> 32);
    }
};

struct DegreeExtractor {
    __device__ __forceinline__
    EdgeIx operator()(const NodeData& nd) const {
        return nd.activeDegree;
    }
};

struct ArgMinOp {
    __host__ __device__
    SweepCutData operator()(const SweepCutData& a, const SweepCutData& b) const {
        return (a.sparsity < b.sparsity) ? a : b;
    }
};

struct NodeDataScanOp {
    __host__ __device__
    NodeData operator()(const NodeData& a, const NodeData& b) const {
        NodeData res = b;
        res.prefixEdgeDiff += a.prefixEdgeDiff;
        res.prefixVolume += a.prefixVolume;
        res.offsetInCluster += a.offsetInCluster;
        return res;
    }
};

struct ReduceOp {
    const EdgeIx* clusterVolumesPtr;

    explicit ReduceOp(const EdgeIx* volumesPtr) : clusterVolumesPtr(volumesPtr) {}

    __host__ __device__
    SweepCutData operator()(const NodeData& nodeData) const {
        const EdgeIx totalVol  = clusterVolumesPtr[nodeData.label];

        const EdgeIx edgeDiff = nodeData.prefixEdgeDiff;
        const EdgeIx prefixVol = nodeData.prefixVolume;

        // min(vol, totalVol - vol)
        const EdgeIx denom = (prefixVol < totalVol - prefixVol) ? prefixVol : (totalVol - prefixVol);
        const float sparsity  = (denom > 0) ? (static_cast<float>(edgeDiff) / static_cast<float>(denom)) : 2.0f;

        return { nodeData.label, sparsity , nodeData.offsetInCluster };
    }
};

void SweepCutManager::compute(GraphManager& gm, PartitionManager& pm, const thrust::device_vector<frac_t> &values) {

    /**
     *  STEP 0: Prepare Data
     */

    auto& partition = pm.getPartitionView();

    unsigned int blocksPerGrid = (gm.n + threads - 1) / threads;

    nodeDiffKernel_Sparse<<<blocksPerGrid, threads, 0, nullptr>>>(
            gm.n,
            thrust::raw_pointer_cast(gm.getNeighbors().data()),
            partition.Current(),
            thrust::raw_pointer_cast(values.data())
    );

    cudaStreamSynchronize(nullptr);

    /**
     *  STEP 1: SORT
     */

    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        packedKeys,
        partition,
        static_cast<int>(gm.n)
    );

    NodeData* sortedData = partition.Current();
    uint64_t* sortedKeys = packedKeys.Current();

    auto label_iter = thrust::make_transform_iterator(sortedKeys, LabelExtractor());

    /**
     *  STEP 2: SCAN
     * - Perform separate prefix sum on edgeDiff and volume
     * - save outcome in-place into partition
     */
    thrust::inclusive_scan_by_key(
        thrust::device,
        // Keys
        label_iter,
        label_iter + gm.n,
        // Values
        sortedData,      // Input Begin
        sortedData,      // Output Begin (In-place!)
        thrust::equal_to<int>(),
        NodeDataScanOp()
    );

    /**
     * STEP 3: REDUCE
     */

    // find internal volume of each cluster
    thrust::reduce_by_key(
            thrust::device,
            // Keys
            label_iter,
            label_iter + gm.n,
            // Values
            thrust::make_transform_iterator(sortedData, DegreeExtractor()),
            // Output Label
            thrust::make_discard_iterator(),
            // Output Values
            pm.getVolumes().begin(),
            thrust::equal_to<int>(),
            thrust::plus<>()
    );

    EdgeIx* clusterVolumesPtr = thrust::raw_pointer_cast(pm.getVolumes().data());

    // find best sweep cut for each cluster
    auto end_pair = thrust::reduce_by_key(
        thrust::device,
        // Keys
        label_iter,
        label_iter + gm.n,
        // Values
        thrust::make_transform_iterator(sortedData, ReduceOp(clusterVolumesPtr)),
        // Output Label
        d_unique_labels.begin(),
        // Output Values
        sweepCuts.begin(),
        thrust::equal_to<int>(),
        ArgMinOp()
    );

    numActiveClusters = static_cast<int>(thrust::distance(sweepCuts.begin(), end_pair.second));
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
