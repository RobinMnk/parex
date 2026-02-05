//
// Created by robin on 26.01.2026.
//

#ifndef PAREX_DEVPARTITION_H
#define PAREX_DEVPARTITION_H

#include "types.h"
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>
#include "assert.h"

struct InitFunctor {
    const NodeIx* ranges;

    explicit InitFunctor(const NodeIx* _ranges) : ranges(_ranges) {}

    __device__
    NodeData operator()(const int i) const {
        NodeIx start = ranges[i];
        NodeIx end   = ranges[i+1];

        return {
            static_cast<NodeIx>(i),
            0,
            end - start,
            end - start,
        };
    }
};

struct ActiveEdgeLogic {
    const NodeIx* neighbors;

    __device__
    int operator()(EdgeIx edgeIdx) const {
        return neighbors[edgeIdx] != INVALID_EDGE ? 1 : 0;
        // EdgeIx revEdge = edgeMap[edgeIdx];
        // NodeIx srcNode = neighbors[revEdge];
        // NodeIx srcLabel = nodes[srcNode].label;
        //
        // assert(edgeMap[revEdge] == edgeIdx);
        // assert(nodes[srcNode].nix == srcNode);
        //
        // NodeIx tgtNode = neighbors[edgeIdx];
        // NodeIx tgtLabel = __ldg(&nodes[tgtNode].label);
        //
        // assert(nodes[tgtNode].nix == tgtNode);
        //
        // return (srcLabel == tgtLabel) ? 1 : 0;
    }
};

__global__
void disableEdgesKernel(
    EdgeIx totalEdges,
    const NodeData* __restrict__ nodeData,
    const EdgeIx* __restrict__ edgeMap,
    NodeIx* __restrict__ neighbors
) {
    EdgeIx edgeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (edgeIdx >= totalEdges) return;

    NodeIx tgtNode = neighbors[edgeIdx];
    if (tgtNode == INVALID_EDGE) {
        // edge already inactive
        return;
    }

    NodeIx tgtLabel = __ldg(&nodeData[tgtNode].label);

    EdgeIx revEdge = edgeMap[edgeIdx];
    NodeIx srcNode = neighbors[revEdge];
    NodeIx srcLabel = nodeData[srcNode].label;

    assert(edgeMap[revEdge] == edgeIdx);
    assert(nodeData[srcNode].nix == srcNode);


    assert(nodeData[tgtNode].nix == tgtNode);

    if (srcLabel != tgtLabel) {
        // inactive edges point to totalEdges
        neighbors[edgeIdx] = INVALID_EDGE;
    }
}




class PartitionManager {
    NodeIx numNodes;
    EdgeIx totalEdges;
    thrust::device_vector<NodeData> partition1;
    thrust::device_vector<NodeData> partition2;
    cub::DoubleBuffer<NodeData> partition;

    NodeIx numActiveClusters;
    thrust::device_vector<EdgeIx> volumes;
    thrust::device_vector<EdgeIx> activeDegrees;


    // CUB Buffers
    size_t temp_storage_bytes = 0;
    void *d_temp_storage = nullptr;


public:

    explicit PartitionManager(GraphManager& gm) :
        numNodes(gm.n),
        totalEdges(2*gm.m),
        partition1(gm.n),
        partition2(gm.n),
        partition(thrust::raw_pointer_cast(partition1.data()),
            thrust::raw_pointer_cast(partition2.data())),
        numActiveClusters{1},
//        activeLabels(gm.n, 0),
        volumes(gm.n, 2 * gm.m),
        activeDegrees(gm.n)
    {
        thrust::transform(
            thrust::make_counting_iterator<NodeIx>(0),
            thrust::make_counting_iterator(gm.n),
            partition1.begin(),
            InitFunctor(thrust::raw_pointer_cast(gm.getRanges().data()))
        );
        auto& ranges = gm.getRanges();
        thrust::transform(ranges.begin() + 1, ranges.end(), ranges.begin(), activeDegrees.begin(), thrust::minus<int>());


        const EdgeIx* edgeMapPtr = thrust::raw_pointer_cast(gm.getEdgeMap().data());
        const EdgeIx* rangesPtr = thrust::raw_pointer_cast(gm.getRanges().data());
        const NodeIx* neighbors = thrust::raw_pointer_cast(gm.getNeighbors().data());

        auto input_iter = thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                ActiveEdgeLogic{neighbors}
        );

        cub::DeviceSegmentedReduce::Sum(
                nullptr, temp_storage_bytes,
                input_iter,
                thrust::raw_pointer_cast(activeDegrees.data()),
                static_cast<int>(numNodes),
                rangesPtr,
                rangesPtr + 1,
                nullptr
        );

        cudaMalloc(&d_temp_storage, temp_storage_bytes);
    }

    void disableEdges(GraphManager& gm) {
        const EdgeIx* edgeMapPtr = thrust::raw_pointer_cast(gm.getEdgeMap().data());
        NodeIx* neighborsPtr = thrust::raw_pointer_cast(gm.getNeighbors().data());

        int gridSize = (totalEdges + threads - 1) / threads;

        disableEdgesKernel<<<gridSize, threads, 0, nullptr>>>(
            totalEdges,
            partition.Current(),
            edgeMapPtr,
            neighborsPtr
        );
    }

    std::vector<int> getActiveEdgeMap(GraphManager& gm) {
        std::vector<int> aem(2*gm.m);

        thrust::device_vector<int> d_aem(2*gm.m);

        const EdgeIx* edgeMapPtr = thrust::raw_pointer_cast(gm.getEdgeMap().data());
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

    void cutClusters(thrust::device_vector<SweepCutData>& sweepCuts, const thrust::device_vector<int>& uniqueLabels, int numClusters, int maxLabel) {
        SweepCutData* sweepCutPtr = thrust::raw_pointer_cast(sweepCuts.data());
        const int* uniqueLabelsPtr = thrust::raw_pointer_cast(uniqueLabels.data());

        thrust::for_each_n(
            thrust::device,
            partition.Current(),
            numNodes,
            [sweepCutPtr, uniqueLabelsPtr, numClusters, maxLabel] __device__ (NodeData& data) {
                const int clusterId = data.label;
                if (clusterId < 0) return; // this cluster is inactive

                const int* it = thrust::lower_bound(
                    thrust::seq,
                    uniqueLabelsPtr,
                    uniqueLabelsPtr + numClusters,
                    clusterId
                );

                int correspondingSweepCutIndex = static_cast<int>(it - uniqueLabelsPtr);

                if (clusterId != uniqueLabelsPtr[correspondingSweepCutIndex]) {
                    printf("ERRORRR!!!! clusterId does not match unique label! (clusterId = %d != %d = index\n", clusterId, correspondingSweepCutIndex);
                }

                SweepCutData sc = sweepCutPtr[correspondingSweepCutIndex];

                if (clusterId != sc.clusterId) {
                    printf("ERRORRR!!!! clusterId does not match sweep cut!\n%d is the clusterId, this is the scId: %d\n", clusterId, sc.clusterId);
                }

                if(sc.sparsity < sc_threshold && data.offsetInCluster > sc.offset) {
                    // printf("Cluster %d has sparsity %f < %f \t -> cutting at offset = %d\n", sc.clusterId, sc.sparsity, sc_threshold, sc.offset);

                    data.label += maxLabel + 1;
                }
            }
        );

        // numActiveClusters += numNewClusters;
    }

    /**
     * requires invariant partition[i].nix == i
     */
    void computeActiveDegrees(GraphManager& gm) {
        const EdgeIx* rangesPtr = thrust::raw_pointer_cast(gm.getRanges().data());
        const NodeIx* neighborsPtr = thrust::raw_pointer_cast(gm.getNeighbors().data());

        auto input_iter = thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                ActiveEdgeLogic{neighborsPtr}
        );

        cub::DeviceSegmentedReduce::Sum(
            d_temp_storage,
            temp_storage_bytes,
            input_iter,
            thrust::raw_pointer_cast(activeDegrees.data()),
            static_cast<int>(numNodes),
            rangesPtr,
            rangesPtr + 1,
            nullptr
        );
    }

    /**
     * Restore invariant that partition[i].nix == i
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

    thrust::device_vector<EdgeIx>& getActiveDegrees() {
        return activeDegrees;
    }


    cub::DoubleBuffer<NodeData>& getPartitionView() {
        return partition;
    }

    thrust::device_vector<EdgeIx>& getVolumes() {
        return volumes;
    }

    struct Temp {
        NodeIx ix;
        EdgeIx deg;
    };


    struct DegreeExtractor {
        __host__ __device__
        Temp operator()(const NodeData& n) const {
            return {n.nix, n.degree };
        }
    };


    std::vector<EdgeIx> downloadActiveDegrees() {
        std::vector<EdgeIx> degrees(numNodes);
        thrust::copy(activeDegrees.begin(), activeDegrees.end(), degrees.begin());
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
//        }
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
