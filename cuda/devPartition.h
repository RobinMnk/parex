//
// Created by robin on 26.01.2026.
//

#ifndef PAREX_DEVPARTITION_H
#define PAREX_DEVPARTITION_H

#include "types.h"
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>

struct InitFunctor {
    const NodeIx* ranges;

    explicit InitFunctor(const NodeIx* _ranges) : ranges(_ranges) {}

    __device__
    NodeData operator()(const int i) const {
        NodeIx start = ranges[i];
        NodeIx end   = ranges[i+1];

        return {
            static_cast<NodeIx>(i),
            static_cast<NodeIx>(0),
            end - start,
            start,
            end - start,
        };
    }
};

struct ActiveEdgeLogic {
    const NodeData* nodes;
    const NodeIx* neighbors;
    const EdgeIx* rowOffsets;
    NodeIx numNodes;

    __device__
    int operator()(EdgeIx edgeIdx) const {
        NodeIx srcIdx = thrust::upper_bound(thrust::seq, rowOffsets, rowOffsets + numNodes, edgeIdx) - rowOffsets - 1;

        NodeIx srcLabel = nodes[srcIdx].label;
        NodeIx tgtNode = neighbors[edgeIdx];
        NodeIx tgtLabel = __ldg(&nodes[tgtNode].label);

        // TODO: the map in the graph gives the edge in the other direction -> this can remove the binary search!
        // srcLabel = neighbors[map[edgeIdx]];

        return (srcLabel == tgtLabel) ? 1 : 0;
    }
};


class PartitionManager {
    NodeIx numNodes;
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


        NodeIx* rangesPtr = thrust::raw_pointer_cast(gm.getRanges().data());
        NodeIx* neighbors = thrust::raw_pointer_cast(gm.getNeighbors().data());

        auto input_iter = thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                ActiveEdgeLogic{partition.Current(), neighbors, rangesPtr, gm.n}
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

    void cutClusters(thrust::device_vector<SweepCutData>& sweepCuts, NodeIx numNewClusters) {
        SweepCutData* sweepCutPtr = thrust::raw_pointer_cast(sweepCuts.data());

        thrust::for_each_n(
            thrust::device,
            partition.Current(),
            numNodes,
            [sweepCutPtr, numNewClusters] __device__ (NodeData& data) {
                const NodeIx clusterId = data.label;
                SweepCutData sc = sweepCutPtr[clusterId];
                if(data.offsetInCluster > sc.offset) {
                    data.label += numNewClusters;
                }
            }
        );

        numActiveClusters += numNewClusters;
    }

    /**
     * requires invariant partition[i].nix == i
     */
    void computeActiveDegrees(GraphManager& gm) {
        NodeIx* ranges = thrust::raw_pointer_cast(gm.getRanges().data());
        NodeIx* neighbors = thrust::raw_pointer_cast(gm.getNeighbors().data());

        auto input_iter = thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                ActiveEdgeLogic{partition.Current(), neighbors, ranges, gm.n}
        );

        cub::DeviceSegmentedReduce::Sum(
            d_temp_storage,
            temp_storage_bytes,
            input_iter,
            thrust::raw_pointer_cast(activeDegrees.data()),
            static_cast<int>(numNodes),
            ranges,
            ranges + 1,
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
