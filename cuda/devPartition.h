//
// Created by robin on 26.01.2026.
//

#ifndef PAREX_DEVPARTITION_H
#define PAREX_DEVPARTITION_H

#include "types.h"
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include "timer.h"

struct InitFunctor {
    const NodeIx* ranges;

    explicit InitFunctor(const NodeIx* _ranges) : ranges(_ranges) {}

    __device__
    NodeData operator()(const int i) const {
        NodeIx start = ranges[i];
        NodeIx end   = ranges[i+1];

        return {
            static_cast<NodeIx>(i),
            end - start,
            0,
        };
    }
};

struct ActiveEdgeLogic {
    const NodeIx* neighbors;

    __device__
    int operator()(EdgeIx edgeIdx) const {
        return neighbors[edgeIdx] != INVALID_EDGE ? 1 : 0;
    }
};

__global__
void disableEdgesKernel(
    EdgeIx totalEdges,
    const int* __restrict__ labels,
    const NodeIx* __restrict__ nodeLookup,
    NodeIx* __restrict__ neighbors
) {
    EdgeIx edgeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (edgeIdx >= totalEdges) return;

    NodeIx tgtNode = neighbors[edgeIdx];
    if (tgtNode == INVALID_EDGE) {
        // edge already inactive
        return;
    }

    NodeIx tgtLabel = __ldg(&labels[tgtNode]);

    NodeIx srcNode = nodeLookup[edgeIdx];

    if (srcNode == INVALID_EDGE) {
        neighbors[edgeIdx] = INVALID_EDGE;
        return;
    }

    NodeIx srcLabel = labels[srcNode];

    // assert(edgeMap[revEdge] == edgeIdx);
    // assert(nodeData[srcNode].nix == srcNode);
    // assert(nodeData[tgtNode].nix == tgtNode);

    if (srcLabel != tgtLabel) {
        // inactive edges point to totalEdges
        neighbors[edgeIdx] = INVALID_EDGE;
    }
}

__global__
void disableEdgesKernel_node(
    const NodeIx numActiveNodes,
    const LabeledNode* __restrict__ nodes,
    const label_t* __restrict__ allLabels,
    NodeIx* __restrict__ neighbors,
    const EdgeIx* __restrict__ ranges,
    const EdgeIx* __restrict__ degrees,
    NodeIx* __restrict__ internalDegrees
) {
    const size_t warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP;
    const size_t lane   = threadIdx.x & 31;
    if (warpId >= numActiveNodes) return;

    NodeIx nix = 0;
    label_t myLabel = 0;
    EdgeIx start = 0, degree = 0;

    // Only Lane 0 issues the memory requests
    if (lane == 0) {
        const LabeledNode& lNode = nodes[warpId];
        nix     = lNode.nix;
        myLabel = lNode.clusterId;
        start   = __ldg(&ranges[nix]);
        degree  = __ldg(&degrees[nix]);
    }

    // Broadcast the values to all lanes
    nix     = __shfl_sync(0xffffffff, nix, 0);
    myLabel = __shfl_sync(0xffffffff, myLabel, 0);
    start   = __shfl_sync(0xffffffff, start, 0);
    degree  = __shfl_sync(0xffffffff, degree, 0);

    const EdgeIx end = start + degree;

    if (myLabel < 0) {
        // This can happen when nodes were just deactivated in this iteration, nothing to do here
        // printf("ERROR: considering inactive node in disable edges kernel!\n");
        return;
    }

    NodeIx deg = 0;
    for (EdgeIx j = start + lane; j < end; j += WARP) {
        const NodeIx nb = __ldg(&neighbors[j]);
        if (nb == INVALID_EDGE) continue;
        const label_t otherLabel = allLabels[nb];
        if (myLabel != otherLabel) {
            neighbors[j] = INVALID_EDGE;
            printf("Disabling edge %d -> %d\t[eix: %d]\n", nix, nb, j);
        } else {
            deg++;
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        deg += __shfl_down_sync(0xffffffff, deg, offset);
    }

    if (lane == 0) {
        internalDegrees[nix] = deg;
    }
}


//
// struct ClusterDataReduceOp {
//     __host__ __device__
//     ClusterData operator()(const ClusterData& a, const ClusterData& b) const {
//         return {
//             a.rwSum + b.rwSum,
//             a.maxPotential > b.maxPotential ? a.maxPotential : b.maxPotential,
//             a.minPotential < b.minPotential ? a.minPotential : b.minPotential,
//             a.totalElements + b.totalElements
//         };
//     }
// };

struct MakeLabeledNodeOp {
    __host__ __device__
    LabeledNode operator()(const NodeIx i) const {
        return LabeledNode{i, 0};
    }
};

struct RangeExtractorOp {
    const EdgeIx* rangesPtr;
    const int offset; // 0 for begin, 1 for end

    __host__ __device__
    RangeExtractorOp(const EdgeIx* ptr, int off) : rangesPtr(ptr), offset(off) {}

    __device__
    EdgeIx operator()(const LabeledNode lNode) const {
        return rangesPtr[lNode.nix + offset];
    }
};

struct LabelExtractor {
    __host__ __device__
    label_t operator()(const LabeledNode lNode) const {
        return lNode.clusterId;
    }
};


class PartitionManager {
    NodeIx numNodes;
    EdgeIx totalEdges;
    // thrust::device_vector<NodeData> partition1;
    // thrust::device_vector<NodeData> partition2;
    // cub::DoubleBuffer<NodeData> partition;

    thrust::device_vector<label_t> allLabels;
    thrust::device_vector<EdgeIx> allInternalDegrees;

    thrust::device_vector<LabeledNode> activeNodes;

    thrust::device_vector<label_t> clusterLabels;
    thrust::device_vector<EdgeIx> clusterVolumes;
    thrust::device_vector<ClusterData> clusterData;

    // thrust::device_vector<int> labelLookup;
    // thrust::device_vector<int> temp_keys;
    // uint64_t* sortedKeys;


    // CUB Buffers
    size_t tempBytesReduce = 0; //, tempBytesSort = 0;
    void *tempStorageReduce = nullptr;
    // void *tempStorageSort = nullptr;

public:
    NodeIx totalClusters{1}, numActiveClusters{1};
    NodeIx numDisabledNodes{0}, numActiveNodes;

    Timer t, t2;

    explicit PartitionManager(GraphManager& gm) :
        numNodes(gm.n),
        totalEdges(2*gm.m),
        // partition1(gm.n),
        // partition2(gm.n),
        // partition(thrust::raw_pointer_cast(partition1.data()),
        //     thrust::raw_pointer_cast(partition2.data())),
        allLabels(gm.n, 0),
        allInternalDegrees(gm.n),
        activeNodes(gm.n),
        clusterLabels(gm.n, 0),
        clusterVolumes(gm.n, 2 * gm.m),
        clusterData(gm.n),
        // labelLookup(2 * gm.n + 1, -1),
        // temp_keys(gm.n, 0),
        numActiveNodes{gm.n}
    {
        auto index_sequence_begin = thrust::make_counting_iterator<NodeIx>(0);

        thrust::transform(
            index_sequence_begin,
            index_sequence_begin + gm.n,
            activeNodes.begin(),
            MakeLabeledNodeOp()
        );


        // thrust::sequence(activeNodes.begin(), activeNodes.end()); // initialize all node indices as active [0,1,2,3,...]
        // thrust::transform(
        //     thrust::make_counting_iterator<NodeIx>(0),
        //     thrust::make_counting_iterator(gm.n),
        //     partition1.begin(),
        //     InitFunctor(thrust::raw_pointer_cast(gm.getRanges().data()))
        // );
        auto& ranges = gm.getRanges();
        thrust::transform(ranges.begin() + 1, ranges.end(), ranges.begin(), allInternalDegrees.begin(), thrust::minus<NodeIx>());


        const EdgeIx* rangesPtr = thrust::raw_pointer_cast(gm.getRanges().data());
        const NodeIx* neighbors = thrust::raw_pointer_cast(gm.getNeighbors().data());

        auto input_iter = thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                ActiveEdgeLogic{neighbors}
        );

        cub::DeviceSegmentedReduce::Sum(
                nullptr, tempBytesReduce,
                input_iter,
                thrust::raw_pointer_cast(allInternalDegrees.data()),
                static_cast<int>(numNodes),
                rangesPtr,
                rangesPtr + 1
        );

        cudaMalloc(&tempStorageReduce, tempBytesReduce);

        // cub::DeviceRadixSort::SortPairs(
        //     nullptr, tempBytesSort,
        //     keys,
        //     partition,
        //     static_cast<int>(numNodes)
        // );
        //
        // cudaMalloc(&tempStorageSort, tempBytesSort);

        t.start();
        t2.start();
    }


    // void sortByKeys(cub::DoubleBuffer<uint64_t>& keys) {
    //     cub::DeviceRadixSort::SortPairs(
    //         tempStorageSort, tempBytesSort,
    //         keys,
    //         partition,
    //         static_cast<int>(numActiveNodes)
    //     );
    //
    //     sortedKeys = keys.Current();
    //
    //     // smallestKey = extractLabel(sortByKeys[0]);
    //
    //     // auto label_iter_all_clusters = thrust::make_transform_iterator(sortedKeys, LabelExtractor());
    //     //
    //     // auto label_iter = thrust::find_if(
    //     //     thrust::device,
    //     //     label_iter_all_clusters,
    //     //     label_iter_all_clusters + numNodes,
    //     //     [] __device__ (int label) { return label >= 0; }
    //     // );
    //     //
    //     // numDisabledNodes = thrust::distance(label_iter_all_clusters, label_iter);
    //     // numActiveNodes = numNodes - numDisabledNodes;
    //     //
    //     // // printf("active Nodes: %d (clusters: %d)\t\tRound time:  %lldms\ttotal time: %lldms\n", numActiveNodes, numActiveClusters, t.timeMillis(), t2.timeMillis());
    //     // // t.start();
    //     //
    //     // return label_iter;
    // }

    // NodeData* activeNodes() {
    //     return partition.Current() + numDisabledNodes;
    // }


    thrust::device_vector<LabeledNode>& getActiveNodes() {
        return activeNodes;
    }

    thrust::device_vector<label_t>& getUniqueActiveLabels() {
        return clusterLabels;
    }


    void computeClusterData(const double* dist) {
        auto value_iter = thrust::make_transform_iterator(
            activeNodes.begin(),
            [dist] __device__ (const LabeledNode lNode) -> ClusterData {
                const auto rwValue = static_cast<float>(dist[lNode.nix]);
                return { rwValue, rwValue, rwValue, 1 };
            }
        );

        const auto extractLabel =  [] __device__ (const LabeledNode a) { return a.clusterId; };
        const auto iter_begin = thrust::make_transform_iterator(activeNodes.begin(), extractLabel);

        // auto equalClusterPred = [] __device__ (const LabeledNode a, const LabeledNode b) { return a.clusterId == b.clusterId; };

        auto reduceOp = [] __device__ (const ClusterData& a, const ClusterData& b) -> ClusterData {
            return {
                a.rwSum + b.rwSum,
                a.maxPotential > b.maxPotential ? a.maxPotential : b.maxPotential,
                a.minPotential < b.minPotential ? a.minPotential : b.minPotential,
                a.totalElements + b.totalElements
            };
        };

        auto end_iters_new = thrust::reduce_by_key(
            thrust::device,
            iter_begin,
            iter_begin + numActiveNodes,
            value_iter,
            clusterLabels.begin(),
            clusterData.begin(),
            thrust::equal_to<label_t>(),
            reduceOp
        );

        inspect(clusterLabels, numActiveClusters);

        numActiveClusters = end_iters_new.second - clusterData.begin();



        // auto active_base_ptr = partition.Current() + numDisabledNodes;
        //
        // auto label_iter = thrust::make_transform_iterator(active_base_ptr, LabelExtractorRW());
        //
        // thrust::copy(thrust::device, label_iter, label_iter + numActiveNodes, temp_keys.begin());


        // clusters were sorted by label after sweep cut, but consolidate may have split some, destroying the order
        // before the reduce, we need to sort them again
        // thrust::sort_by_key(
        //     temp_keys.begin(),
        //     temp_keys.begin() + numActiveNodes,
        //     active_base_ptr
        // );

        //
        // auto value_iter = thrust::make_transform_iterator(activeNodes(), ClusterDataExtractorRW());
        //
        // // find ClusterData for each cluster (to compute potential and average of each)
        // auto end_iters = thrust::reduce_by_key(
        //     thrust::device,
        //     label_iter,
        //     label_iter + numActiveNodes,
        //     value_iter,
        //     activeLabels.begin(),
        //     clusterSums.begin(),
        //     thrust::equal_to<int>(),
        //     ClusterDataReduceOp()
        // );

        // thrust::sort_by_key(
        //     activeLabels.begin(),
        //     activeLabels.begin() + numActiveClusters,
        //     clusterSums.begin()
        // );

        // end_iters = thrust::reduce_by_key(
        //     thrust::device,
        //     activeLabels.begin(),
        //     activeLabels.begin() + numActiveClusters,
        //     clusterSums.begin(),
        //     activeLabels.begin(),
        //     clusterSums.begin(),
        //     thrust::equal_to<int>(),
        //     ClusterDataReduceOp()
        // );
        //
        // numActiveClusters = end_iters.second - clusterSums.begin();


        // size_t maxLabel;
        // thrust::copy_n(activeLabels.begin() + numActiveClusters - 1, 1, &maxLabel);

        // std::vector<int> h_labels(numActiveClusters);
        // thrust::copy(activeLabels.begin(), activeLabels.begin() + numActiveClusters, h_labels.begin());
        // for (int i = 0; i < numActiveClusters; i++) {
        //     printf("%d: Cluster: %d\n", i, h_labels[i]);
        // }
        // fflush(stdout);

        // printf("There are %d clusters and the maximum label is %llu\n", numActiveClusters, maxLabel);
    }

    void disableEdges(GraphManager& gm) {
        NodeIx* neighborsPtr = thrust::raw_pointer_cast(gm.getNeighbors().data());
        const EdgeIx* rangesPtr = thrust::raw_pointer_cast(gm.getRanges().data());
        const EdgeIx* degreesPtr = thrust::raw_pointer_cast(gm.getDegrees().data());
        const LabeledNode* nodes = thrust::raw_pointer_cast(activeNodes.data());
        const label_t* labelsPtr = thrust::raw_pointer_cast(allLabels.data());
        NodeIx* internalDegsPtr = thrust::raw_pointer_cast(allInternalDegrees.data());

        const size_t gridSize = getGridSize(numActiveNodes);

        disableEdgesKernel_node<<<gridSize, threads>>>(
            numActiveNodes,
            nodes,
            labelsPtr,
            neighborsPtr,
            rangesPtr,
            degreesPtr,
            internalDegsPtr
        );
        fflush(stdout);
    }

    std::vector<int> getActiveEdgeMap(GraphManager& gm) const {
        std::vector<int> aem(2*gm.m);

        thrust::device_vector<int> d_aem(2*gm.m);

        const NodeIx* neighbors = thrust::raw_pointer_cast(gm.getNeighbors().data());

        auto input_iter = thrust::make_counting_iterator(0);

        thrust::transform(
            input_iter,
            input_iter + numNodes,
            d_aem.begin(),
            ActiveEdgeLogic{neighbors}
        );

        thrust::copy(d_aem.begin(), d_aem.end(), aem.begin());

        return aem;
    }

    void cutClusters(
        cub::DoubleBuffer<NodeData>& scNodeData,
        const thrust::device_vector<SweepCutData>& sweepCuts,
        size_t numClustersWithCut
    ) {
        const SweepCutData* sweepCutPtr = thrust::raw_pointer_cast(sweepCuts.data());
        const NodeData* scNodeDataPtr = scNodeData.Current();
        const label_t* uniqueLabels = thrust::raw_pointer_cast(clusterLabels.data());
        LabeledNode* labeledNodesPtr = thrust::raw_pointer_cast(activeNodes.data());
        label_t* allLabelsPtr = thrust::raw_pointer_cast(allLabels.data());

        const float sparsity_target = sc_threshold;
        const label_t maxLabel = clusterLabels.back();
        const NodeIx clusterCount = numClustersWithCut;

        inspect(clusterLabels, numActiveClusters);

        thrust::for_each_n(
            thrust::device,
            thrust::make_counting_iterator(0),
            numActiveNodes,
            [sweepCutPtr, scNodeDataPtr, labeledNodesPtr, allLabelsPtr, uniqueLabels, sparsity_target, maxLabel, clusterCount] __device__ (int idx) {
                const NodeData& scNode = scNodeDataPtr[idx];
                LabeledNode lNode = labeledNodesPtr[idx];

                if (scNode.nix != lNode.nix) {
                    printf("ERROR: scNode.nix != lNode.nix:  %d != %d\n", scNode.nix, lNode.nix);
                }

                if (lNode.clusterId < 0) {
                    // cluster already inactive - this should actually never happen now!
                    printf("ERROR: Considering inactive node in recenterAndDeactivateClusters! It has label %d\n", lNode.clusterId);
                    return;
                }

                const SweepCutData* sc = thrust::lower_bound(
                    thrust::seq,
                    sweepCutPtr,
                    sweepCutPtr + clusterCount,
                    lNode.clusterId,
                    [=] __device__ (const SweepCutData& element, label_t target) {
                        return element.clusterId < target;
                    }
                );

                // lower bound returns the first value >= lNode.clusterId. It thus might return a larger value if there is no matching entry
                if (sc == sweepCutPtr + clusterCount || lNode.clusterId != sc->clusterId) {
                    // the sparsity of this sweepCut was above the threshold so it was removed -> nothing to do
                    printf("INFO: did not find corresponding sweep cut info for node %d with label = %d\n", lNode.nix, lNode.clusterId);
                    return;
                }

                // if (lNode.clusterId != sc->clusterId) {
                //     printf("ERRORRR!!!! clusterId does not match sweep cut!\n\t%d is the clusterId, this is the scId: %d\t[sparsity = %f, offset = %d], also index = %lld and numClusters: %d\n", lNode.clusterId, scData.clusterId, scData.sparsity, scData.offset, index, clusterCount);
                // }

                if (sc->sparsity >= sparsity_target) {
                    printf("ERRORRR!!!! cluster should have been removed:\t[sparsity = %f >= = %f]\n", sc->sparsity, sparsity_target);

                }

                printf("Node %d has offset %d threshold is > %d\n", lNode.nix, scNode.offsetInCluster, sc->offset);
                if(scNode.offsetInCluster > sc->offset) { // TODO: should be > not >=
                    label_t updatedLabel = lNode.clusterId + maxLabel + 1;
                    labeledNodesPtr[idx].clusterId = updatedLabel;
                    allLabelsPtr[lNode.nix] = updatedLabel;
                    printf("Cluster %d is split into two parts -> new label for nix = %d is %d, because maxLabel = %d\n", lNode.clusterId, lNode.nix, updatedLabel, maxLabel);
                }
            }
        );
    }

    void recenterAndDeactivateClusters(double* dist, const label_t* smallestLabel) {
        // subtract average from each (active) node
        const ClusterData* clusterDataPtr = thrust::raw_pointer_cast(clusterData.data());
        const label_t* uniqueLabelsPtr = thrust::raw_pointer_cast(clusterLabels.data());
        label_t* allLabelsPtr = thrust::raw_pointer_cast(allLabels.data());

        const float walk_threshold = rw_threshold;
        const NodeIx numClusters = numActiveClusters;

        thrust::for_each_n(
            thrust::device,
            activeNodes.begin(),
            numActiveNodes,
            [clusterDataPtr, uniqueLabelsPtr, dist, allLabelsPtr, walk_threshold, smallestLabel, numClusters] __device__ (LabeledNode& lNode) {
                const label_t label = lNode.clusterId;

                if (label < 0) {
                    // cluster already inactive - this should actually never happen now!
                    // printf("node %d with label %d returns because label is negative\n", data.nix, label);
                    printf("ERROR: Considering inactive node in recenterAndDeactivateClusters! It has label %d\n", label);
                    return;
                }

                const int* it = thrust::lower_bound(
                    thrust::seq,
                    uniqueLabelsPtr,
                    uniqueLabelsPtr + numClusters,
                    label
                );
                const int index = static_cast<int>(it - uniqueLabelsPtr);

                // int correspondingSweepCutIndex = labelLookupPtr[label];

                // if (comp != correspondingSweepCutIndex) {
                //     printf("WARN: comp != correspondingSweepCutIndex \t %d != %d\n", comp, correspondingSweepCutIndex);
                // }


                if (uniqueLabelsPtr[index] != label) {
                    printf("ERROR: label mismatch!! For nix = %d:\t%d != %d\n", lNode.nix, label, uniqueLabelsPtr[index]);
                }

                const ClusterData& cd = clusterDataPtr[index];

                // if (cd.totalElements < 2) {
                //     // printf("node %d with label %d returns because numElements = %d. Note that totalClusters = %d\n", data.nix, label, cd.totalElements, numUnique);
                //     return;
                // };

                const float clusterPotential = cd.maxPotential - cd.minPotential;

                if (clusterPotential < walk_threshold || cd.totalElements < 2) {
                    // this cluster should be deactivated
                    const label_t updatedLabel = *smallestLabel - label - 1;
                    printf("Deactivating cluster: %d -> %d \t smallest: %d\n", lNode.clusterId, updatedLabel, *smallestLabel);
                    lNode.clusterId = updatedLabel;
                    allLabelsPtr[lNode.nix] = updatedLabel;
                } else {
                    const double average = cd.rwSum / static_cast<double>(cd.totalElements);
                    dist[lNode.nix] -= average;
                }
            }
        );

        // printf("After\n");
        // int x = computeClusterData(partition, uniqueLabels);
        // // assert(x-1 == maxLabel);
        // print(numUnique, uniqueLabels);
        // printf("\n");
        //
        // return numUnique;
    }


    // void updateLabelLookup() {
    //     const int* uniqueLabelsPtr = thrust::raw_pointer_cast(activeLabels.data());
    //     int* lookupPtr = thrust::raw_pointer_cast(labelLookup.data());
    //
    //     auto populate_map = [uniqueLabelsPtr, lookupPtr] __device__ (const int i) {
    //         int label = uniqueLabelsPtr[i];
    //         if (label >= 0) {
    //             lookupPtr[label] = i;
    //         }
    //     };
    //
    //     thrust::for_each(
    //         thrust::device,
    //         thrust::make_counting_iterator<NodeIx>(0),
    //         thrust::make_counting_iterator(numActiveClusters),
    //         populate_map
    //     );
    //
    //
    //     // std::vector<int> h_lookup(2*numNodes+1);
    //     // thrust::copy(labelLookup.begin(), labelLookup.end(), h_lookup.begin());
    //     // for (int i = 0; i < h_lookup.size(); ++i) {
    //     //     int index = h_lookup[i];
    //     //     if (index >= 0) {
    //     //         printf("label %d -> %d\n", i, index);
    //     //     }
    //     // }
    // }

    // auto& getLabelLookup() const  {
    //     return labelLookup;
    // }
    // auto& getActiveNodeLabels()  {
    //     return activeLabels;
    // }

    EdgeIx numCutEdges(GraphManager& gm) {
        const NodeIx* neighborsPtr = thrust::raw_pointer_cast(gm.getNeighbors().data());

        auto input_iter_begin = thrust::make_transform_iterator(
            thrust::make_counting_iterator(0),
            ActiveEdgeLogic{neighborsPtr}
        );

        // Calculate the total sum of active edges
        int total_active = thrust::reduce(
            thrust::device,
            input_iter_begin,
            input_iter_begin + (2 * gm.m),
            0,                  // Initial value
            thrust::plus<int>() // Binary operation
        );

        return gm.m - total_active / 2;
    }

    void computeActiveDegreesOLD(GraphManager& gm) {
        const EdgeIx* rangesPtr = thrust::raw_pointer_cast(gm.getRanges().data());

        const auto input_iter = thrust::make_transform_iterator(
                gm.getNeighbors().begin(),
            [] __device__ (const NodeIx nb) { return nb == INVALID_EDGE ? 0 : 1; }
        );

        const auto ranges_begin_iter = thrust::make_transform_iterator(
            activeNodes.begin(), RangeExtractorOp(rangesPtr, 0)
        );

        const auto ranges_end_iter = thrust::make_transform_iterator(
            activeNodes.begin(), RangeExtractorOp(rangesPtr, 1)
        );

        // writes output to internalDegs[lNode[i].nix]
        auto writer = thrust::make_permutation_iterator(
            allInternalDegrees.begin(),
            thrust::make_transform_iterator(
                activeNodes.begin(),LabelExtractor()
            )
        );


        // TODO: this does not work!! CUB cannot do this random jumping between segments!!
        cub::DeviceSegmentedReduce::Sum(
            tempStorageReduce,
            tempBytesReduce,
            input_iter,
            writer,
            static_cast<int>(numActiveNodes),
            ranges_begin_iter,
            ranges_end_iter
        );
    }

    FinalPartition finalizePartition() {
        thrust::device_vector<label_t> B(numNodes);

        // 1. Create a sorted list of unique elements from A
        thrust::device_vector<label_t> unique_keys = allLabels;
        thrust::sort(unique_keys.begin(), unique_keys.end());
        auto new_end = thrust::unique(unique_keys.begin(), unique_keys.end());

        int num_unique = thrust::distance(unique_keys.begin(), new_end);

        unique_keys.erase(new_end, unique_keys.end());

        // 2. Map original values in A to their index in unique_keys
        // lower_bound returns the position, which effectively becomes the ID [0, num_unique-1]
        thrust::lower_bound(unique_keys.begin(),
                            unique_keys.end(),
                            allLabels.begin(),
                            allLabels.end(),
                            B.begin());


        std::vector<label_t> labels(numNodes);
        thrust::copy(B.begin(), B.end(), labels.begin());

        return {labels, num_unique };
    }

    /**
     * Restore nix-order, i.e., invariant that partition[i].nix == i
     */
    // void scatter() {
    //     NodeData* in  = partition.Current();
    //     NodeData* out = partition.Alternate();
    //
    //     thrust::for_each_n(
    //         thrust::cuda::par.on(nullptr),
    //         thrust::make_counting_iterator<NodeIx>(0),
    //         numNodes,
    //         [in, out] __device__ (NodeIx k) {
    //             const NodeData nd = in[k];
    //             out[nd.nix] = nd;
    //         }
    //     );
    //
    //     partition.selector ^= 1;
    // }

    thrust::device_vector<EdgeIx>& getAllInternalDegrees() {
        return allInternalDegrees;
    }

    thrust::device_vector<label_t>& getAllLabels() {
        return allLabels;
    }


    // cub::DoubleBuffer<NodeData>& getPartitionView() {
    //     return partition;
    // }

    thrust::device_vector<EdgeIx>& getVolumes() {
        return clusterVolumes;
    }

    void print(int numUnique, thrust::device_vector<int>& uniqueLabels) {
        std::vector<ClusterData> cst(numUnique);
        thrust::copy(clusterData.begin(), clusterData.begin() + numUnique, cst.begin());
        std::vector<int> lbl(numUnique);
        thrust::copy(uniqueLabels.begin(), uniqueLabels.begin() + numUnique, lbl.begin());
        for (int i = 0; i < numUnique; i++) {
            const float clusterPotential = cst[i].maxPotential - cst[i].minPotential;
            const float average = cst[i].rwSum / static_cast<float>(cst[i].totalElements);
            printf("%d:  Cluster %d [average = %f, potential = %f, numElements = %d]\n", i, lbl[i], average, clusterPotential, cst[i].totalElements);
        }
        fflush(stdout);
    }


    // struct Temp {
    //     NodeIx ix;
    //     EdgeIx deg;
    // };


    // struct DegreeExtractor {
    //     __host__ __device__
    //     Temp operator()(const NodeData& n) const {
    //         return {n.nix, n.degree };
    //     }
    // };


    std::vector<EdgeIx> downloadActiveDegrees() {
        std::vector<EdgeIx> degrees(numNodes);
        thrust::copy(allInternalDegrees.begin(), allInternalDegrees.end(), degrees.begin());
        return degrees;
    }

//    std::vector<EdgeIx> downloadActiveDegrees() {
//        size_t n = partition1.size();
//        std::vector<Temp> degrees(n);
//
//        auto d_ptr = thrust::device_pointer_cast(getPartitionView().Current());
//
//        auto first = thrust::make_transform_iterator(
//                d_ptr, DegreeExtractor{}
//        );
//        auto last = first + n;
//
//        thrust::copy(first, last, degrees.begin());
//
//        std::vector<EdgeIx> result(n);
//        for(int i = 0; i < n; i++) {
//            result[degrees[i].ix] = degrees[i].deg;
//        }D
//        return result;
//    }
};



#endif //PAREX_DEVPARTITION_H
