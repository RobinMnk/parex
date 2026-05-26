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
#include <thrust/binary_search.h>

#include <cassert>


__global__
void nodeDiffKernel_Sparse_WarpParallel(
    NodeIx numActiveNodes,
    const LabeledNode* __restrict__ nodes,
    const NodeIx* __restrict__ neighbors,
    const EdgeIx* __restrict__ ranges,
    const NodeIx* __restrict__ degrees,
    const NodeIx* __restrict__ clusterDegrees,
    const label_t* __restrict__ labels,
    const uint64_t* __restrict__ packedKeys,
    const double* __restrict__ dist,
    NodeData* __restrict__ nodeData
) {
    const unsigned int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP;
    const unsigned int lane   = threadIdx.x & 31;
    if (warpId >= numActiveNodes) return;

    NodeIx nix = 0;
    uint64_t myKey = 0;
    int64_t myLabel = 0;
    EdgeIx start = 0, degree = 0, clusterDegree = 0;

    // Only Lane 0 issues the memory requests to the uniform addresses
    if (lane == 0) {
        const LabeledNode& lNode = nodes[warpId];
        nix             = lNode.nix;
        myLabel         = lNode.clusterId;
        myKey           = __ldg(&packedKeys[warpId]);
        start           = __ldg(&ranges[nix]);
        degree          = __ldg(&degrees[nix]);
        clusterDegree    = __ldg(&clusterDegrees[nix]);
    }

    // Broadcast the uniform values to all lanes in 1 clock cycle
    nix     = __shfl_sync(0xffffffff, nix, 0);
    myLabel = __shfl_sync(0xffffffff, myLabel, 0);
    start   = __shfl_sync(0xffffffff, start, 0);
    degree  = __shfl_sync(0xffffffff, degree, 0);
    myKey   = __shfl_sync(0xffffffff, myKey, 0);

    const EdgeIx end = start + degree;

    if (myLabel < 0) {
        printf("ERROR: sweep cut should not consider inactive nodes now!!\n");
        return;
    }

    int32_t nodeContribution = 0;
    for (EdgeIx j = start + lane; j < end; j += WARP) {
        const NodeIx nb = __ldg(&neighbors[j]);

        if (nb == INVALID_EDGE) continue;

        const label_t nbLabel   = __ldg(&labels[nb]);
        const double nbRwValue  = __ldg(&dist[nb]);
        // re-computing the key from scratch is much faster than trying to find and load the value from global memory
        const uint64_t nbKey = packKey(nbLabel, nbRwValue);

        if (nbLabel != myLabel) {
            printf("ERROR: this edge should have been invalidated: %d -> %d [eix: %d]\n", nix, nb, j);
        }

        bool isBefore;
        if (nbKey != myKey) {
            isBefore = (nbKey < myKey);
        } else {
            isBefore = (nb < nix);
        }

        nodeContribution += isBefore ? -1 : 1;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        nodeContribution += __shfl_down_sync(0xffffffff, nodeContribution, offset);

    if (lane == 0) {
        nodeData[warpId].nix                = nix;
        nodeData[warpId].prefixEdgeDiff     = nodeContribution;
        nodeData[warpId].prefixVolume       = clusterDegree;
        nodeData[warpId].offsetInCluster    = 1;
        // nodeData[i].rwValue         = static_cast<float>(myVal);
    }
}


//
//
// __global__
// void nodeDiffKernel_Sparse(
//         NodeIx numNodes,
//         const NodeIx* __restrict__ neighbors,
//         const EdgeIx* __restrict__ ranges,
//         const NodeIx* __restrict__ degrees,
//         NodeData* __restrict__ nodeData,
//         const uint64_t* __restrict__ packedKeys
// ) {
//     NodeIx i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= numNodes) return;
//
//     // early exit for inactive nodes
//     const NodeData data = nodeData[i];
//
//     if (data.nix != i) {
//         printf("nix mismatch: i = %d but nix = %d\n", i, data.nix);
//     }
//
//     assert(data.nix == i && "nix mismatch!");
//
//     if (data.label < 0) return;
//
//     const uint64_t myKey = __ldg(&packedKeys[i]);
//
//     int nodeContribution = 0;
//     const EdgeIx rangeStart = ranges[i];
//     const EdgeIx rangeEnd = ranges[i] + degrees[i];
//
//
//     for (NodeIx j = rangeStart; j < rangeEnd; ++j) {
//         const NodeIx neighbor = __ldg(&neighbors[j]);
//         if (neighbor == INVALID_EDGE) {
//             // inactive edge;
//             continue;
//         }
//
//         const NodeData nbData = nodeData[neighbor];
//
//         assert(nbData.nix == neighbor && "neighbor nix mismatch!");
//
//         // printf("%d: edge %d [%d]  -->  %d [%d]\n", j, data.nix, data.label, nbData.nix, nbData.label);
//
//         // only consider edges that stay within the same cluster
//         if (nbData.label == data.label) {
//             const uint64_t otherKey = __ldg(&packedKeys[neighbor]);
//
//             bool isBefore;
//             if (otherKey != myKey) {
//                 isBefore = (otherKey < myKey);
//             } else {
//                 // matching the stable sort
//                 isBefore = (neighbor < i);
//             }
//
//             nodeContribution += (isBefore) ? -1 : 1;
//
//             // printf("  %d: Edge %d -> %d adds %d\t[running total: %d]\n", j, nodeData[i].nix, nbData.nix, isBefore ? 1 : -1, nodeContribution);
//         } else {
//             // printf("  WARN: %d:  edge %d -> %d ignored because %d.label = %d but %d.label = %d\n", j, data.nix, nbData.nix, data.nix, data.label, nbData.nix, nbData.label);
//         }
//     }
//
//     // Initialize data fields for the sweep cut
//     nodeData[i].prefixEdgeDiff = nodeContribution;
//     nodeData[i].prefixVolume = data.activeDegree;
//     nodeData[i].offsetInCluster = 1;
//     // nodeData[i].rwValue = myVal;
// }
// // This needs all random walk values done -> has to be computed after!


class SweepCutManager {
    NodeIx numNodes;

    // Buffers
    thrust::device_vector<uint64_t> packedKeysIn;
    thrust::device_vector<uint64_t> packedKeysOut;
    cub::DoubleBuffer<uint64_t> packedKeys;

    thrust::device_vector<NodeData> scNodeData1;
    thrust::device_vector<NodeData> scNodeData2;
    cub::DoubleBuffer<NodeData> scNodeData;

    // CUB Buffers
    size_t temp_storage_bytes = 0;
    void *d_temp_storage = nullptr;


    // Output
    thrust::device_vector<SweepCutData> sweepCuts, sweepCutsBuffer;

    void initializeCUB(NodeIx n) {
        cub::DeviceRadixSort::SortPairs(
            nullptr, temp_storage_bytes,
            packedKeys,
            scNodeData,
            static_cast<int>(n)
        );

        cudaMalloc(&d_temp_storage, temp_storage_bytes);
    }

public:
    NodeIx numClustersWithCut{0};


    explicit SweepCutManager(NodeIx n) :
        numNodes(n),
        packedKeysIn(n),
        packedKeysOut(n),
        packedKeys(thrust::raw_pointer_cast(packedKeysIn.data()),
                   thrust::raw_pointer_cast(packedKeysOut.data())),
        scNodeData1(n),
        scNodeData2(n),
        scNodeData(thrust::raw_pointer_cast(scNodeData1.data()),
            thrust::raw_pointer_cast(scNodeData2.data())),
        sweepCuts(n),
        sweepCutsBuffer(n)
    {
        initializeCUB(n);
    }

    thrust::device_vector<SweepCutData>& getSweepCuts() {
        return sweepCuts;
    }

    void compute(GraphManager& gm, PartitionManager& pm, const double* dist);

    cub::DoubleBuffer<uint64_t>& getKeyBuffer() {
        return packedKeys;
    }

    cub::DoubleBuffer<NodeData>& getScNodeData() {
        return scNodeData;
    }

    std::vector<NodeData> downloadPartition() {
        std::vector<NodeData> pt(numNodes);
        thrust::device_ptr<NodeData> dev_ptr(scNodeData.Current());
        thrust::copy(dev_ptr, dev_ptr + numNodes, pt.begin());
        return pt;
    }

    void print(const double*);
};


// struct LabelExtractor {
//     __host__ __device__
//     inline int operator()(uint64_t k) const {
//         uint32_t orderedLabel = static_cast<uint32_t>(k >> 32);
//         return static_cast<int>(orderedLabel ^ 0x80000000);
//     }
// };

// struct DegreeExtractor {
//     const EdgeIx* clusterDegrees;
//
//     __device__ __forceinline__
//     EdgeIx operator()(const NodeIx nix) const {
//         return clusterDegrees[nix]; // TODO which one?
//         // return nd.degree;
//     }
// };

__host__ __device__
inline label_t extractLabel(const uint64_t key) {
    const auto orderedLabel = static_cast<int32_t>(key >> 32);
    return orderedLabel ^ 0x80000000;
}

struct LabelExtractor {
    __host__ __device__
    label_t operator()(uint64_t key) const {
        return extractLabel(key);
    }
};

struct ClusterDegreeExtractorOp {
    const EdgeIx* clusterDegreesPtr;

    __host__ __device__
    ClusterDegreeExtractorOp(const EdgeIx* ptr) : clusterDegreesPtr(ptr) {}

    __device__
    EdgeIx operator()(const LabeledNode lNode) const {
        return clusterDegreesPtr[lNode.nix];
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

struct SweepCutTransform {
    const NodeData* nodeDataPtr;
    const EdgeIx* clusterVolumesPtr;
    const uint64_t* packedKeysPtr;
    size_t numInactive;

    __host__ __device__
    SweepCutData operator()(const size_t idx) const {
        const NodeData nodeData = nodeDataPtr[idx];
        const EdgeIx totalVol  = clusterVolumesPtr[idx];
        const label_t label = extractLabel(packedKeysPtr[numInactive + idx]);

        const EdgeIx edgeDiff = nodeData.prefixEdgeDiff;
        const EdgeIx prefixVol = nodeData.prefixVolume;

        const EdgeIx denom = (prefixVol >= totalVol) ? 0 :
                             (prefixVol < totalVol - prefixVol) ? prefixVol : (totalVol - prefixVol);

        const float sparsity = (denom > 0) ? (static_cast<float>(edgeDiff) / static_cast<float>(denom)) : 2.0f;

        return { label, sparsity , nodeData.offsetInCluster };
    }
};

void SweepCutManager::print(const double* dist) {
    std::vector<NodeData> nodes(numNodes);
    thrust::device_ptr<NodeData> dev_ptr(scNodeData.Current());
    thrust::copy(dev_ptr, dev_ptr + numNodes, nodes.begin());

    std::vector<frac_t> rwVals(numNodes);
    auto dev_ptr_start = thrust::device_pointer_cast(dist);
    auto dev_ptr_end   = dev_ptr_start + numNodes;
    thrust::copy(dev_ptr_start, dev_ptr_end, rwVals.begin());

    for (NodeIx i = 0; i < numNodes; i++) {
        printf("Node %d has offset %d, prefixSum = %d and prefixVol = %d with rwValue = %f\n", nodes[i].nix, nodes[i].offsetInCluster, nodes[i].prefixEdgeDiff, nodes[i].prefixVolume, rwVals[i]);
    }
    fflush(stdout);
}

inline void SweepCutManager::compute(GraphManager& gm, PartitionManager& pm, const double* dist) {

    /**
     *  STEP 0: Prepare Data
     */

    const size_t numBlocks = getGridSize(gm.n);

    nodeDiffKernel_Sparse_WarpParallel<<<numBlocks, threads>>>(
        pm.numActiveNodes,
        thrust::raw_pointer_cast(pm.getActiveNodes().data()),
        thrust::raw_pointer_cast(gm.getNeighbors().data()),
        thrust::raw_pointer_cast(gm.getRanges().data()),
        thrust::raw_pointer_cast(gm.getDegrees().data()),
        thrust::raw_pointer_cast(pm.getAllInternalDegrees().data()),
         thrust::raw_pointer_cast(pm.getAllLabels().data()),
        packedKeys.Current(),
        dist,
        scNodeData.Current()
    );


    /**
     *  STEP 1: SORT
     */

    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        packedKeys,
        scNodeData,
        static_cast<int>(pm.numActiveNodes)
    );

    // this includes nodes that just became inactive in the last iteration. Those have not been removed yet (intentionally, for performance)
    auto allLabels = thrust::make_transform_iterator(packedKeys.Current(), LabelExtractor());

    // this now contains only labels of active nodes
    auto activeLabels = thrust::lower_bound(
        thrust::device,
        allLabels,
        allLabels + pm.numActiveNodes,
        0 // We are looking for the first element >= 0
    );

    // the number of elements deactivated in the last round
    size_t inactiveElements = activeLabels - allLabels;
    // the number of remaining active nodes
    pm.numActiveNodes -= inactiveElements;

    auto activeNodeData = scNodeData.Current() + inactiveElements;

    const auto inputs_begin = thrust::make_zip_iterator(thrust::make_tuple(
        activeNodeData,
        activeLabels
    ));

    thrust::transform(
        inputs_begin, inputs_begin + pm.numActiveNodes,
        pm.getActiveNodes().begin(),
        [] __device__ (const auto& pair) {
            return LabeledNode{
                thrust::get<0>(pair).nix,
                thrust::get<1>(pair),
            };
        }
    );

    /**
     *  STEP 2: SCAN
     * - Perform separate prefix sum on edgeDiff and volume
     * - save outcome in-place into scNodeData
     */
    thrust::inclusive_scan_by_key(
        thrust::device,
        // Keys
        activeLabels,
        activeLabels + pm.numActiveNodes,
        // Values
        activeNodeData,      // Input Begin
        activeNodeData,      // Output Begin (In-place!)
        thrust::equal_to<label_t>(),
        NodeDataScanOp()
    );


    /**
     * STEP 3: REDUCE
     */

    EdgeIx* clusterDegreesPtr = thrust::raw_pointer_cast(pm.getAllInternalDegrees().data());
    auto clusterDegreeExtractor = thrust::make_transform_iterator(
        pm.getActiveNodes().begin(),
        ClusterDegreeExtractorOp(clusterDegreesPtr)
    );

    // auto clusterDegreeExtractor = thrust::make_transform_iterator(
    //     pm.getActiveNodes().begin(), [clusterDegreesPtr] __device__ (const LabeledNode lNode) { return clusterDegreesPtr[lNode.nix]; }
    // );

    // find cluster-internal volumes
    thrust::reduce_by_key(
        thrust::device,
        // Keys
        activeLabels,
        activeLabels + pm.numActiveNodes,
        // Values
        clusterDegreeExtractor,
        // Output Label
        pm.getUniqueActiveLabels().begin(),
        // Output Values
        pm.getVolumes().begin(),
        thrust::equal_to<label_t>(),
        thrust::plus<EdgeIx>()
    );


    // find best sweep cut for each cluster

    const EdgeIx* clusterVolumesPtr = thrust::raw_pointer_cast(pm.getVolumes().data());
    const SweepCutTransform scTransform(activeNodeData, clusterVolumesPtr, packedKeys.Current(), inactiveElements);

    const auto it = thrust::reduce_by_key(
        thrust::device,
        // Keys
        activeLabels,
        activeLabels + pm.numActiveNodes,
        // Values
        thrust::make_transform_iterator(thrust::make_counting_iterator<size_t>(0), scTransform),
        // Output Label
        pm.getUniqueActiveLabels().begin(),
        // Output Values
        sweepCutsBuffer.begin(),
        thrust::equal_to<int32_t>(),
        ArgMinOp()
    );
    pm.numActiveClusters = static_cast<size_t>(it.first - pm.getUniqueActiveLabels().begin());


    std::vector<SweepCutData> scs(pm.numActiveClusters);
    thrust::copy(sweepCutsBuffer.begin(), sweepCutsBuffer.begin() + pm.numActiveClusters, scs.begin());
    printf("Computed these sweep cuts:\n");
    for (auto sc : scs) {
        printf("Cluster: %d, sparsity = %f, offset = %d\n", sc.clusterId, sc.sparsity, sc.offset);
    }
    fflush(stdout);



    const auto threshold = sc_threshold;

    // Ignore all sweep Cuts that have a sparsity above the threshold
    auto end_iter = thrust::copy_if(
        sweepCutsBuffer.begin(), sweepCutsBuffer.begin() + pm.numActiveClusters,
        sweepCuts.begin(),
        [threshold] __device__ (const SweepCutData& scData) {
            return scData.sparsity < threshold;
        }
    );

    numClustersWithCut = static_cast<size_t>(end_iter - sweepCuts.begin());

    if (numClustersWithCut > 0) {
        std::vector<SweepCutData> scs2(numClustersWithCut);
        thrust::copy(sweepCuts.begin(), sweepCuts.begin() + numClustersWithCut, scs2.begin());
        printf("These Sweep Cuts are below the threshold:\n");
        for (auto sc : scs2) {
            printf("Cluster: %d, sparsity = %f, offset = %d\n", sc.clusterId, sc.sparsity, sc.offset);
        }
        fflush(stdout);
    }
}





    //
    // EdgeIx* clusterVolumesPtr = thrust::raw_pointer_cast(pm.getVolumes().data());
    // int* labelsPtr = thrust::raw_pointer_cast(pm.getActiveLabels().data());
    // const int* labelLookupPtr = thrust::raw_pointer_cast(pm.getLabelLookup().data());

    // int num = end.second - pm.getVolumes().begin();
    //
    // pm.numActiveClusters = num;
    // pm.updateLabelLookup();

    // assert(num == pm.numActiveClusters);

    // printf("\n\n");

    // std::vector<NodeData> nodes(numNodes);
    // thrust::device_ptr<NodeData> dev_ptr(partition.Current());
    // thrust::copy(dev_ptr, dev_ptr + numNodes, nodes.begin());
    // for (NodeIx i = 0; i < numNodes; i++) {
    //     printf("Node %d has label %d at offset %d, prefixSum = %d and prefixVol = %d with rwValue = %f\n", nodes[i].nix, nodes[i].label, nodes[i].offsetInCluster, nodes[i].prefixEdgeDiff, nodes[i].prefixVolume, nodes[i].rwValue);
    // }


    // std::vector<EdgeIx> clusterVolumes(num);
    // thrust::copy(pm.getVolumes().begin(), pm.getVolumes().begin() + num, clusterVolumes.begin());
    // std::vector<int> h_labels(num);
    // thrust::copy(pm.getActiveLabels().begin(), pm.getActiveLabels().begin() + num, h_labels.begin());
    // for (int i = 0; i < num; i++) {
    //     printf("%d: Cluster: %d, volume: %d\n", i, h_labels[i], clusterVolumes[i]);
    // }
    // fflush(stdout);




    // int num_unique_keys = thrust::distance(pm.getActiveLabels().begin(), end_pair.first);
    //
    // assert(num == num_unique_keys);

    // std::vector<SweepCutData> scs(pm.numActiveClusters);
    // thrust::copy(sweepCuts.begin(), sweepCuts.begin() + pm.numActiveClusters, scs.begin());
    // //
    // std::vector<int> lbl(pm.numActiveClusters);
    // thrust::copy(pm.getActiveLabels().begin(), pm.getActiveLabels().begin() + pm.numActiveClusters, lbl.begin());
    //
    //
    // for (int i = 0; i < pm.numActiveClusters; i++) {
    //     if (scs[i].clusterId != lbl[i]) {
    //         printf("ERROR: label mismatch in sweepcut! scId = %d != %d = label at index %d\n", scs[i].clusterId, lbl[i], i);
    //     }
    //     if ( i == 35764 || i == 40937 || i == 31427) {
    //         printf("\tix %d: label %d, sc = [id: %d, sps = %f, off = %d]\n", i, lbl[i], scs[i].clusterId, scs[i].sparsity, scs[i].offset);
    //     }
    // }
    // fflush(stdout);

    // printf("Computed these sweep cuts:\n");
    // for (auto sc : scs) {
    //     printf("Cluster: %d, sparsity = %f, offset = %d\n", sc.clusterId, sc.sparsity, sc.offset);
    // }
    // fflush(stdout);




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
