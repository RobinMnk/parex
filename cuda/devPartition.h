//
// Created by robin on 26.01.2026.
//

#ifndef PAREX_DEVPARTITION_H
#define PAREX_DEVPARTITION_H

#include "types.h"
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>
#include "assert.h"
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


class PartitionManager {
    NodeIx numNodes;
    EdgeIx totalEdges;
    // thrust::device_vector<NodeData> partition1;
    // thrust::device_vector<NodeData> partition2;
    // cub::DoubleBuffer<NodeData> partition;

    thrust::device_vector<int64_t> nodeLabels;
    thrust::device_vector<EdgeIx> clusterDegrees;

    thrust::device_vector<LabeledNode> activeNodes;

    thrust::device_vector<int32_t> clusterLabels;
    thrust::device_vector<EdgeIx> clusterVolumes;
    thrust::device_vector<ClusterData> clusterData;

    thrust::device_vector<int> labelLookup;
    thrust::device_vector<int> temp_keys;

    uint64_t* sortedKeys;


    // CUB Buffers
    size_t tempBytesReduce = 0; //, tempBytesSort = 0;
    void *tempStorageReduce = nullptr;
    // void *tempStorageSort = nullptr;

public:
    NodeIx totalClusters{1}, numActiveClusters{1};
    NodeIx numDisabledNodes{0}, numActiveNodes;

    Timer t, t2;

    explicit PartitionManager(GraphManager& gm, cub::DoubleBuffer<uint64_t>& keys) :
        numNodes(gm.n),
        totalEdges(2*gm.m),
        // partition1(gm.n),
        // partition2(gm.n),
        // partition(thrust::raw_pointer_cast(partition1.data()),
        //     thrust::raw_pointer_cast(partition2.data())),
        nodeLabels(gm.n, 0),
        clusterDegrees(gm.n),
        activeNodes(gm.n),
        clusterLabels(gm.n, 0),
        clusterVolumes(gm.n, 2 * gm.m),
        clusterData(gm.n),
        labelLookup(2 * gm.n + 1, -1),
        temp_keys(gm.n, 0),
        numActiveNodes{gm.n}
    {
        auto index_sequence_begin = thrust::make_counting_iterator<NodeIx>(0);

        thrust::transform(
            index_sequence_begin,
            index_sequence_begin + gm.n,
            activeNodes.begin(),
            [] __device__ (const NodeIx idx) -> LabeledNode {
                return LabeledNode{
                    idx, 0
                };
            }
        );


        // thrust::sequence(activeNodes.begin(), activeNodes.end()); // initialize all node indices as active [0,1,2,3,...]
        // thrust::transform(
        //     thrust::make_counting_iterator<NodeIx>(0),
        //     thrust::make_counting_iterator(gm.n),
        //     partition1.begin(),
        //     InitFunctor(thrust::raw_pointer_cast(gm.getRanges().data()))
        // );
        auto& ranges = gm.getRanges();
        thrust::transform(ranges.begin() + 1, ranges.end(), ranges.begin(), clusterDegrees.begin(), thrust::minus<NodeIx>());


        const EdgeIx* rangesPtr = thrust::raw_pointer_cast(gm.getRanges().data());
        const NodeIx* neighbors = thrust::raw_pointer_cast(gm.getNeighbors().data());

        auto input_iter = thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                ActiveEdgeLogic{neighbors}
        );

        cub::DeviceSegmentedReduce::Sum(
                nullptr, tempBytesReduce,
                input_iter,
                thrust::raw_pointer_cast(clusterDegrees.data()),
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


    void sortByKeys(cub::DoubleBuffer<uint64_t>& keys) {
        cub::DeviceRadixSort::SortPairs(
            tempStorageSort, tempBytesSort,
            keys,
            partition,
            static_cast<int>(numActiveNodes)
        );

        sortedKeys = keys.Current();

        // smallestKey = extractLabel(sortByKeys[0]);

        // auto label_iter_all_clusters = thrust::make_transform_iterator(sortedKeys, LabelExtractor());
        //
        // auto label_iter = thrust::find_if(
        //     thrust::device,
        //     label_iter_all_clusters,
        //     label_iter_all_clusters + numNodes,
        //     [] __device__ (int label) { return label >= 0; }
        // );
        //
        // numDisabledNodes = thrust::distance(label_iter_all_clusters, label_iter);
        // numActiveNodes = numNodes - numDisabledNodes;
        //
        // // printf("active Nodes: %d (clusters: %d)\t\tRound time:  %lldms\ttotal time: %lldms\n", numActiveNodes, numActiveClusters, t.timeMillis(), t2.timeMillis());
        // // t.start();
        //
        // return label_iter;
    }

    // NodeData* activeNodes() {
    //     return partition.Current() + numDisabledNodes;
    // }


    thrust::device_vector<LabeledNode>& getActiveNodes() {
        return activeNodes;
    }

    thrust::device_vector<label_t>& getUniqueActiveLabels() {
        return clusterLabels;
    }


    /**
     *  needs Partition in label-order
     */
    void computeClusterData() {
        auto active_base_ptr = partition.Current() + numDisabledNodes;

        // TODO I should already have this somewhere
        auto label_iter = thrust::make_transform_iterator(active_base_ptr, LabelExtractorRW());

        thrust::copy(thrust::device, label_iter, label_iter + numActiveNodes, temp_keys.begin());

        thrust::sort_by_key(
            temp_keys.begin(),
            temp_keys.begin() + numActiveNodes,
            active_base_ptr
        );

        auto value_iter = thrust::make_transform_iterator(activeNodes(), ClusterDataExtractorRW());

        // find ClusterData for each cluster (to compute potential and average of each)
        auto end_iters = thrust::reduce_by_key(
            thrust::device,
            label_iter,
            label_iter + numActiveNodes,
            value_iter,
            activeLabels.begin(),
            clusterSums.begin(),
            thrust::equal_to<int>(),
            ClusterDataReduceOp()
        );

        numActiveClusters = end_iters.second - clusterSums.begin();

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


    /**
     * needs partition in nix-order
     */
    void disableEdges(GraphManager& gm) {
        const NodeIx* nodeLookupPtr = thrust::raw_pointer_cast(gm.getNodeLookup().data());
        NodeIx* neighborsPtr = thrust::raw_pointer_cast(gm.getNeighbors().data());
        const int* labelsPtr = thrust::raw_pointer_cast(nodeLabels.data());

        int gridSize = (totalEdges + threads - 1) / threads;

        disableEdgesKernel<<<gridSize, threads, 0, nullptr>>>(
            totalEdges,
            labelsPtr,
            nodeLookupPtr,
            neighborsPtr
        );
    }

    std::vector<int> getActiveEdgeMap(GraphManager& gm) {
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
        size_t numClusters
    ) {
        const SweepCutData* sweepCutPtr = thrust::raw_pointer_cast(sweepCuts.data());
        const NodeData* scNodeDataPtr = scNodeData.Current();
        LabeledNode* labeledNodesPtr = thrust::raw_pointer_cast(activeNodes.data());

        const NodeIx n = numNodes;
        const float sparsity_target = sc_threshold;

        thrust::for_each_n(
            thrust::device,
            thrust::make_counting_iterator(0),
            numActiveNodes,
            [sweepCutPtr, scNodeDataPtr, labeledNodesPtr, sparsity_target, n, numClusters] __device__ (int idx) {
                const NodeData& scNode = scNodeDataPtr[idx];
                LabeledNode& lNode = labeledNodesPtr[idx];

                if (lNode.clusterId < 0) {
                    // cluster already inactive - this should actually never happen now!
                    printf("ERROR: Considering inactive node in recenterAndDeactivateClusters! It has label %d\n", lNode.clusterId);
                    return;
                }

                const SweepCutData* sc = thrust::lower_bound(
                    thrust::seq,
                    sweepCutPtr,
                    sweepCutPtr + numClusters,
                    lNode.clusterId,
                    [=] __device__ (const SweepCutData& element, label_t target) {
                        return element.clusterId < target;
                    }
                );

                if (sc != sweepCutPtr + numClusters) {
                    // the sparsity of this sweepCut was above the threshold so it was removed -> nothing to do
                    return;
                }

                if (lNode.clusterId != sc->clusterId) {
                    printf("ERRORRR!!!! clusterId does not match sweep cut!\n\t%d is the clusterId, this is the scId: %d\t[sparsity = %f, offset = %d]\n", lNode.clusterId, sc->clusterId, sc->sparsity, sc->offset);
                }

                if (sc->sparsity >= sparsity_target) {
                    printf("ERRORRR!!!! cluster should have been removed:\t[sparsity = %f >= = %f]\n", sc->sparsity, sparsity_target);

                }

                if(scNode.offsetInCluster > sc->offset) {
                    // printf("Cluster %d is split into two parts -> new label = %d, because maxLabel = %d\n", data.label, data.label + n + 1, n);
                    lNode.clusterId = lNode.clusterId + static_cast<label_t>(n) + 1; // TODO

                    // TODO: use max label not node id!!!!
                }
            }
        );
    }

    void recenterAndDeactivateClusters(thrust::device_vector<double>& dist) {
        // subtract average from each (active) node
        const ClusterData* clusterDataPtr = thrust::raw_pointer_cast(clusterSums.data());
        const int* uniqueLabelsPtr = thrust::raw_pointer_cast(activeLabels.data());
        int* labelsPtr = thrust::raw_pointer_cast(nodeLabels.data());

        double* distPtr = thrust::raw_pointer_cast(dist.data());

        // const NodeIx numActive = numActiveClusters;


        updateLabelLookup();
        int* labelLookupPtr = thrust::raw_pointer_cast(labelLookup.data());

        const float walk_threshold = rw_threshold;
        const uint64_t* smallestKey = sortedKeys;

        thrust::for_each_n(
            thrust::device,
            activeNodes(),
            numActiveNodes,
            [clusterDataPtr, uniqueLabelsPtr, distPtr, labelsPtr, labelLookupPtr, walk_threshold, smallestKey] __device__ (NodeData& data) {
                const int64_t label = data.label;

                if (data.label < 0) {
                    // cluster already inactive - this should actually never happen now!
                    // printf("node %d with label %d returns because label is negative\n", data.nix, label);
                    printf("ERROR: Considering inactive node in recenterAndDeactivateClusters! It has label %lld\n", data.label);
                    return;
                }

                // const int* it = thrust::lower_bound(
                //     thrust::seq,
                //     uniqueLabelsPtr,
                //     uniqueLabelsPtr + numActive,
                //     label
                // );
                // int correspondingSweepCutIndex = static_cast<int>(it - uniqueLabelsPtr);

                int correspondingSweepCutIndex = labelLookupPtr[label];

                // if (comp != correspondingSweepCutIndex) {
                //     printf("WARN: comp != correspondingSweepCutIndex \t %d != %d\n", comp, correspondingSweepCutIndex);
                // }


                if (uniqueLabelsPtr[correspondingSweepCutIndex] != label) {
                    printf("ERROR: label mismatch!! For nix = %d:\t%lld != %d\n", data.nix, label, uniqueLabelsPtr[correspondingSweepCutIndex]);
                }

                const ClusterData cd = clusterDataPtr[correspondingSweepCutIndex];

                // if (cd.totalElements < 2) {
                //     // printf("node %d with label %d returns because numElements = %d. Note that totalClusters = %d\n", data.nix, label, cd.totalElements, numUnique);
                //     return;
                // };

                const float clusterPotential = cd.maxPotential - cd.minPotential;

                if (clusterPotential < walk_threshold || cd.totalElements < 2) {
                    // this cluster should be deactivated
                    int smallestLabel = extractLabel(smallestKey[0]);
                    // int smallestLabel = uniqueLabelsPtr[0]; // TODO: safe?

                    // printf("Deactivating cluster: %d -> %d \t smallest: %d\n", label, smallestLabel - data.label - 1, smallestLabel);
                    int64_t updatedLabel = smallestLabel - data.label - 1;
                    data.label = updatedLabel;
                    labelsPtr[data.nix] = updatedLabel;
                } else {
                    data.label = correspondingSweepCutIndex;
                    labelsPtr[data.nix] = correspondingSweepCutIndex;

                    const double average = cd.rwSum / static_cast<double>(cd.totalElements);
                    distPtr[data.nix] -= average; // TODO: write this diff to any unused field within NodeData! (save random write)
                    data.rwValue -= average; // TODO: this line is just for debugging
                }

                // printf("moved node %d from cluster %d to cluster %d\n", data.nix, label, data.label);
                // printf("node %d with label %d, old rwValue = %f and offset = %d loaded this cluster data (%d): [average = %f, potential = %f, numElements = %d]\n", data.nix, label, rw, data.offsetInCluster, correspondingSweepCutIndex, average, cd.maxPotential - cd.minPotential, cd.totalElements);

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


    void updateLabelLookup() {
        const int* uniqueLabelsPtr = thrust::raw_pointer_cast(activeLabels.data());
        int* lookupPtr = thrust::raw_pointer_cast(labelLookup.data());

        auto populate_map = [uniqueLabelsPtr, lookupPtr] __device__ (const int i) {
            int label = uniqueLabelsPtr[i];
            if (label >= 0) {
                lookupPtr[label] = i;
            }
        };

        thrust::for_each(
            thrust::device,
            thrust::make_counting_iterator<NodeIx>(0),
            thrust::make_counting_iterator(numActiveClusters),
            populate_map
        );


        // std::vector<int> h_lookup(2*numNodes+1);
        // thrust::copy(labelLookup.begin(), labelLookup.end(), h_lookup.begin());
        // for (int i = 0; i < h_lookup.size(); ++i) {
        //     int index = h_lookup[i];
        //     if (index >= 0) {
        //         printf("label %d -> %d\n", i, index);
        //     }
        // }
    }

    auto& getLabelLookup() const  {
        return labelLookup;
    }
    auto& getActiveNodeLabels()  {
        return activeLabels;
    }

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


    /**
     * needs partition in nix-order
     * i.e., invariant partition[i].nix == i
     */
    void computeActiveDegrees(GraphManager& gm) {
        const EdgeIx* rangesPtr = thrust::raw_pointer_cast(gm.getRanges().data());
        const NodeIx* neighborsPtr = thrust::raw_pointer_cast(gm.getNeighbors().data());

        auto input_iter = thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                ActiveEdgeLogic{neighborsPtr}
        );

        cub::DeviceSegmentedReduce::Sum(
            tempStorageReduce,
            tempBytesReduce,
            input_iter,
            thrust::raw_pointer_cast(clusterDegrees.data()),
            static_cast<int>(numNodes),
            rangesPtr,
            rangesPtr + 1,
            nullptr
        );
    }

    FinalPartition finalizePartition() {
        thrust::device_vector<int> B(numNodes);

        // 1. Create a sorted list of unique elements from A
        thrust::device_vector<int> unique_keys = nodeLabels;
        thrust::sort(unique_keys.begin(), unique_keys.end());
        auto new_end = thrust::unique(unique_keys.begin(), unique_keys.end());

        int num_unique = thrust::distance(unique_keys.begin(), new_end);

        unique_keys.erase(new_end, unique_keys.end());

        // 2. Map original values in A to their index in unique_keys
        // lower_bound returns the position, which effectively becomes the ID [0, num_unique-1]
        thrust::lower_bound(unique_keys.begin(),
                            unique_keys.end(),
                            nodeLabels.begin(),
                            nodeLabels.end(),
                            B.begin());


        std::vector<int> labels(numNodes);
        thrust::copy(B.begin(), B.end(), labels.begin());

        return {labels, num_unique };
    }

    /**
     * Restore nix-order, i.e., invariant that partition[i].nix == i
     */
    void scatter() {
        NodeData* in  = partition.Current();
        NodeData* out = partition.Alternate();

        thrust::for_each_n(
            thrust::cuda::par.on(nullptr),
            thrust::make_counting_iterator<NodeIx>(0),
            numNodes,
            [in, out] __device__ (NodeIx k) {
                const NodeData nd = in[k];
                out[nd.nix] = nd;
            }
        );

        partition.selector ^= 1;
    }

    thrust::device_vector<EdgeIx>& getClusterDegrees() {
        return clusterDegrees;
    }


    cub::DoubleBuffer<NodeData>& getPartitionView() {
        return partition;
    }

    thrust::device_vector<EdgeIx>& getVolumes() {
        return clusterVolumes;
    }

    void print(int numUnique, thrust::device_vector<int>& uniqueLabels) {
        std::vector<ClusterData> cst(numUnique);
        thrust::copy(clusterSums.begin(), clusterSums.begin() + numUnique, cst.begin());
        std::vector<int> lbl(numUnique);
        thrust::copy(uniqueLabels.begin(), uniqueLabels.begin() + numUnique, lbl.begin());
        for (int i = 0; i < numUnique; i++) {
            const float clusterPotential = cst[i].maxPotential - cst[i].minPotential;
            const float average = cst[i].rwSum / static_cast<float>(cst[i].totalElements);
            printf("%d:  Cluster %d [average = %f, potential = %f, numElements = %d]\n", i, lbl[i], average, clusterPotential, cst[i].totalElements);
        }
        fflush(stdout);
    }


    struct Temp {
        NodeIx ix;
        EdgeIx deg;
    };


    // struct DegreeExtractor {
    //     __host__ __device__
    //     Temp operator()(const NodeData& n) const {
    //         return {n.nix, n.degree };
    //     }
    // };


    std::vector<EdgeIx> downloadActiveDegrees() {
        std::vector<EdgeIx> degrees(numNodes);
        thrust::copy(clusterDegrees.begin(), clusterDegrees.end(), degrees.begin());
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

    std::vector<NodeData> downloadPartition() {
        std::vector<NodeData> pt(numNodes);
        thrust::device_ptr<NodeData> dev_ptr(partition.Current());
        thrust::copy(dev_ptr, dev_ptr + numNodes, pt.begin());
        return pt;
    }

};



#endif //PAREX_DEVPARTITION_H
