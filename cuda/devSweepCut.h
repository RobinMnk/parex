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

__inline__ __device__
int warpReduceSumInt(int val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__
void nodeDiffKernel_Sparse_WarpParallel(
        NodeIx numNodes,
        const NodeIx* __restrict__ neighbors,
        const EdgeIx* __restrict__ ranges,
        const NodeIx* __restrict__ degrees,
        NodeData* __restrict__ nodeData,
        const double* __restrict__ values
) {
    constexpr int WARP = 32;

    const int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    const int warpId         = globalThreadId / WARP;   // node index
    const int lane           = threadIdx.x & (WARP - 1);

    if (warpId >= numNodes) return;

    const NodeIx i = warpId;

    NodeData data{};
    if (lane == 0) {
        data = nodeData[i];
        // assert(data.nix == i && "nix mismatch!");
    }

    const int words = (sizeof(NodeData) + sizeof(int) - 1) / sizeof(int);
    int* dataWords = reinterpret_cast<int*>(&data);

    #pragma unroll
    for (int k = 0; k < words; ++k) {
        int v = (lane == 0) ? dataWords[k] : 0;
        v = __shfl_sync(0xffffffff, v, 0);
        dataWords[k] = v;
    }

    if (data.label < 0) return;

    assert(data.nix == i && "nix mismatch!");

    double myVal;
    if (lane == 0) myVal = __ldg(&values[i]);
    myVal = __shfl_sync(0xffffffff, myVal, 0);


    const EdgeIx rangeStart = __ldg(&ranges[i]);
    const EdgeIx rangeEnd   = rangeStart + degrees[i];

    int localContribution = 0;

    // const float myVal = nodeData[i].rwValue;

    for (EdgeIx j = rangeStart + lane; j < rangeEnd; j += WARP) {

        const NodeIx neighbor = __ldg(&neighbors[j]);
        // printf("skipping the %dth neighbor.\n", j);
        if (neighbor == INVALID_EDGE) continue;

        const NodeData nbData = nodeData[neighbor];
        // assert(nbData.nix == neighbor && "neighbor nix mismatch!");

        if (nbData.label == data.label) {

            const double otherVal = __ldg(&values[neighbor]);

            // const float otherVal = nbData.rwValue;

            bool isBefore;
            if (otherVal != myVal) {
                isBefore = (otherVal < myVal);
            } else {
                isBefore = (nbData.nix < i);
                printf("edge %d -> %d: %d\n", nbData.nix, i, isBefore);
            }

            // printf("  Edge %d -> %d adds %d\n", nodeData[i].nix, nbData.nix, isBefore ? 1 : -1);

            localContribution += isBefore ? -1 : 1;

            // printf("  Edge %d -> %d adds %d\t[running total: %d]\n", nodeData[i].nix, nbData.nix, isBefore ? 1 : -1, nodeContribution);

        }
    }

    int nodeContribution = warpReduceSumInt(localContribution);

    if (lane == 0) {
        nodeData[i].prefixEdgeDiff  = nodeContribution;
        nodeData[i].prefixVolume    = data.activeDegree;
        nodeData[i].offsetInCluster = 1;
        nodeData[i].rwValue         = static_cast<float>(myVal);
    }
}



__global__
void nodeDiffKernel_Sparse(
        NodeIx numNodes,
        const NodeIx* __restrict__ neighbors,
        const EdgeIx* __restrict__ ranges,
        const NodeIx* __restrict__ degrees,
        NodeData* __restrict__ nodeData,
        const uint64_t* __restrict__ packedKeys
) {
    NodeIx i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) return;

    // early exit for inactive nodes
    const NodeData data = nodeData[i];

    if (data.nix != i) {
        printf("nix mismatch: i = %d but nix = %d\n", i, data.nix);
    }

    assert(data.nix == i && "nix mismatch!");

    if (data.label < 0) return;

    const uint64_t myKey = __ldg(&packedKeys[i]);

    int nodeContribution = 0;
    const EdgeIx rangeStart = ranges[i];
    const EdgeIx rangeEnd = ranges[i] + degrees[i];

    // if (i == 9) {
    //     printf("node %d: start = %d, end = %d, deg = %d", data.nix, rangeStart, rangeEnd, data.degree);
    // }

    for (NodeIx j = rangeStart; j < rangeEnd; ++j) {
        const NodeIx neighbor = __ldg(&neighbors[j]);
        if (neighbor == INVALID_EDGE) {
            // inactive edge;
            // printf("skipping the %dth neighbor.\n", j);
            continue;
        }

        const NodeData nbData = nodeData[neighbor];

        assert(nbData.nix == neighbor && "neighbor nix mismatch!");

        // printf("%d: edge %d [%d]  -->  %d [%d]\n", j, data.nix, data.label, nbData.nix, nbData.label);

        // only consider edges that stay within the same cluster
        if (nbData.label == data.label) {
            const uint64_t otherKey = __ldg(&packedKeys[neighbor]);

            bool isBefore;
            if (otherKey != myKey) {
                isBefore = (otherKey < myKey);
            } else {
                // matching the stable sort
                isBefore = (neighbor < i);
            }

            nodeContribution += (isBefore) ? -1 : 1;

            // printf("  %d: Edge %d -> %d adds %d\t[running total: %d]\n", j, nodeData[i].nix, nbData.nix, isBefore ? 1 : -1, nodeContribution);
        } else {
            // printf("  WARN: %d:  edge %d -> %d ignored because %d.label = %d but %d.label = %d\n", j, data.nix, nbData.nix, data.nix, data.label, nbData.nix, nbData.label);
        }
    }

    // Initialize data fields for the sweep cut
    nodeData[i].prefixEdgeDiff = nodeContribution;
    nodeData[i].prefixVolume = data.activeDegree;
    nodeData[i].offsetInCluster = 1;
    // nodeData[i].rwValue = myVal;
}
// This needs all random walk values done -> has to be computed after!


class SweepCutManager {
    NodeIx numNodes;

    // Buffers
    thrust::device_vector<uint64_t> packedKeysIn;
    thrust::device_vector<uint64_t> packedKeysOut;
    cub::DoubleBuffer<uint64_t> packedKeys;

    // CUB Buffers
    // size_t temp_storage_bytes = 0;
    // void *d_temp_storage = nullptr;


    // Output
    thrust::device_vector<SweepCutData> sweepCuts;

    // void initializeCUB(NodeIx n, PartitionManager& pm) {
    //     cub::DeviceRadixSort::SortPairs(
    //         nullptr, temp_storage_bytes,
    //         packedKeys,
    //         pm.getPartitionView(),
    //         static_cast<int>(n)
    //     );
    //
    //     cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // }

public:
    explicit SweepCutManager(NodeIx n) :
        numNodes(n),
        packedKeysIn(n),
        packedKeysOut(n),
        packedKeys(thrust::raw_pointer_cast(packedKeysIn.data()),
                   thrust::raw_pointer_cast(packedKeysOut.data())),
        sweepCuts(n)
    { }

    thrust::device_vector<SweepCutData>& getSweepCuts() {
        return sweepCuts;
    }

    void compute(
        GraphManager& gm,
        PartitionManager& pm,
        const thrust::device_vector<double> &values
    );

    cub::DoubleBuffer<uint64_t>& getKeyBuffer() {
        return packedKeys;
    }
};


// struct LabelExtractor {
//     __host__ __device__
//     inline int operator()(uint64_t k) const {
//         uint32_t orderedLabel = static_cast<uint32_t>(k >> 32);
//         return static_cast<int>(orderedLabel ^ 0x80000000);
//     }
// };

struct DegreeExtractor {
    __device__ __forceinline__
    EdgeIx operator()(const NodeData& nd) const {
        return nd.activeDegree; // TODO which one?
        // return nd.degree;
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
    const int* labelsPtr;
    const EdgeIx* clusterVolumesPtr;
    const int* labelLookupPtr;
    const int numClusters;

    explicit ReduceOp(const int* labels, const EdgeIx* volumesPtr, const int* lookupPtr, const int num) : labelsPtr(labels), clusterVolumesPtr(volumesPtr), labelLookupPtr(lookupPtr), numClusters{num} {}

    __host__ __device__
    SweepCutData operator()(const NodeData& nodeData) const {

        // const int* it = thrust::lower_bound(
        //     thrust::seq,
        //     labelsPtr,
        //     labelsPtr + numClusters,
        //     nodeData.label
        // );
        // int correspondingIndex = static_cast<int>(it - labelsPtr);
        //
        // int comp = labelLookupPtr[nodeData.label];
        // if (comp != correspondingIndex) {
        //     printf("ERROR: the lookup failed!! %d != %d\n", comp, correspondingIndex);
        // }
        // correspondingIndex = comp;


        int correspondingIndex = labelLookupPtr[nodeData.label];

        if (labelsPtr[correspondingIndex] != nodeData.label) {
            // for (int i = 0; i < numClusters; ++i) {
            //     printf("%d: cluster label: %d\n", i, labelsPtr[correspondingIndex]);
            // }

            printf("ERROR: Label-data mismatch!!\tlabelsPtr[correspondingIndex] = %d, correspondingIndex=%d, nodeData.label = %lld, nix=%d\n", labelsPtr[correspondingIndex], correspondingIndex, nodeData.label, nodeData.nix);
        }

        const EdgeIx totalVol  = clusterVolumesPtr[correspondingIndex];

        const EdgeIx edgeDiff = nodeData.prefixEdgeDiff;
        const EdgeIx prefixVol = nodeData.prefixVolume;

        // 1. Check for underflow to prevent massive denom values
        const EdgeIx denom = (prefixVol >= totalVol) ? 0 :
                             (prefixVol < totalVol - prefixVol) ? prefixVol : (totalVol - prefixVol);

        // 2. Use double for the intermediate calculation if your edge counts are high
        const float sparsity = (denom > 0) ?
            (static_cast<float>(edgeDiff) / static_cast<float>(denom)) : 2.0f;

        //
        // // min(vol, totalVol - vol)
        // const EdgeIx denom = (prefixVol < totalVol - prefixVol) ? prefixVol : (totalVol - prefixVol);
        // const float sparsity  = (denom > 0) ? (static_cast<float>(edgeDiff) / static_cast<float>(denom)) : 2.0f;

        // if (nodeData.label == 316) {
        //     printf("Node %d has label %d at offset %d rwValue = %f, prefixSum = %d and prefixVol = %d -> sparsity: %f, denom = %d\n", nodeData.nix, nodeData.label, nodeData.offsetInCluster, nodeData.rwValue, nodeData.prefixEdgeDiff, nodeData.prefixVolume, sparsity, denom);
        // }


        return { nodeData.label, sparsity , nodeData.offsetInCluster };
    }
};

void SweepCutManager::compute(GraphManager& gm, PartitionManager& pm, const thrust::device_vector<double> &values) {

    /**
     *  STEP 0: Prepare Data
     */

    auto& partition = pm.getPartitionView();


    // std::vector<NodeIx> nodes2(2*gm.m);
    // thrust::device_ptr<NodeIx> dev_ptr2(gm.getNeighbors().data());
    // thrust::copy(dev_ptr2, dev_ptr2 + 2*gm.m, nodes2.begin());
    // for (NodeIx i = 0; i < 2*gm.m; i++) {
    //     printf("Edge %d: nb = %d\n", i, nodes2[i]);
    // }


    constexpr int WARP = 32;
    int warpsPerBlock = threads / WARP;                 // e.g. 256 → 8
    int numBlocks     = (gm.n + warpsPerBlock - 1) / warpsPerBlock;


    nodeDiffKernel_Sparse<<<numBlocks, threads>>>(
            gm.n,
            thrust::raw_pointer_cast(gm.getNeighbors().data()),
            thrust::raw_pointer_cast(gm.getRanges().data()),
            thrust::raw_pointer_cast(gm.getDegrees().data()),
            partition.Current(),
            packedKeys.Current()
    );

    cudaStreamSynchronize(nullptr);

    /**
     *  STEP 1: SORT
     */

    auto activeLabelIter = pm.sortByKeys(packedKeys);
    NodeData* sortedData = pm.activeNodes();


    /**
     *  STEP 2: SCAN
     * - Perform separate prefix sum on edgeDiff and volume
     * - save outcome in-place into partition
     */
    thrust::inclusive_scan_by_key(
        thrust::device,
        // Keys
        activeLabelIter,
        activeLabelIter + pm.numActiveNodes,
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
    auto end = thrust::reduce_by_key(
            thrust::device,
            // Keys
            activeLabelIter,
            activeLabelIter + pm.numActiveNodes,
            // Values
            thrust::make_transform_iterator(sortedData, DegreeExtractor()),
            // Output Label
            pm.getActiveLabels().begin(),
            // Output Values
            pm.getVolumes().begin(),
            thrust::equal_to<int>(),
            thrust::plus<>()
    );

    EdgeIx* clusterVolumesPtr = thrust::raw_pointer_cast(pm.getVolumes().data());
    int* labelsPtr = thrust::raw_pointer_cast(pm.getActiveLabels().data());
    const int* labelLookupPtr = thrust::raw_pointer_cast(pm.getLabelLookup().data());

    int num = end.second - pm.getVolumes().begin();

    pm.numActiveClusters = num;
    pm.updateLabelLookup();

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



    // find best sweep cut for each cluster
    auto end_pair = thrust::reduce_by_key(
        thrust::device,
        // Keys
        activeLabelIter,
        activeLabelIter + pm.numActiveNodes,
        // Values
        thrust::make_transform_iterator(sortedData, ReduceOp(labelsPtr, clusterVolumesPtr, labelLookupPtr, num)),
        // Output Label
            pm.getActiveLabels().begin(),
        // Output Values
        sweepCuts.begin(),
        thrust::equal_to<int>(),
        ArgMinOp()
    );

    int num_unique_keys = thrust::distance(pm.getActiveLabels().begin(), end_pair.first);

    assert(num == num_unique_keys);

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
