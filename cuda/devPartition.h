//
// Created by robin on 26.01.2026.
//

#ifndef PAREX_DEVPARTITION_H
#define PAREX_DEVPARTITION_H

#include "types.h"
#include <thrust/device_vector.h>

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
            end - start,
            start
        };
    }
};


class PartitionManager {
    NodeIx numNodes;
    thrust::device_vector<NodeData> partition1;
    thrust::device_vector<NodeData> partition2;
    cub::DoubleBuffer<NodeData> partition;

    NodeIx numActiveClusters;
    thrust::device_vector<NodeIx> activeLabels;
    thrust::device_vector<EdgeIx> volumes;


public:

    explicit PartitionManager(GraphManager& gm) :
        numNodes(gm.n),
        partition1(gm.n),
        partition2(gm.n),
        partition(thrust::raw_pointer_cast(partition1.data()),
            thrust::raw_pointer_cast(partition2.data())),
        numActiveClusters{1},
        activeLabels(gm.n, 0),
        volumes(gm.n, 2 * gm.m)
    {
        thrust::transform(
            thrust::make_counting_iterator<NodeIx>(0),
            thrust::make_counting_iterator(gm.n),
            partition1.begin(),
            InitFunctor(thrust::raw_pointer_cast(gm.getRanges().data()))
        );
    }

    void cutClusters(thrust::device_vector<SweepCutData>& sweepCuts) {

        SweepCutData* sweeepCutPtr = thrust::raw_pointer_cast(sweepCuts.data());
        auto numNewClusters = sweepCuts.size();

        thrust::for_each_n(
                thrust::device,
                partition.Current(),
                numNodes,
                [sweeepCutPtr, numNewClusters] __device__ (NodeData& data) {
                    const NodeIx clusterId = data.label;
                    SweepCutData sc = sweeepCutPtr[clusterId];
                    if(data.offsetInCluster > sc.offset) {
                        data.label += numNewClusters;
                    }
                }
        );


    }



    /**
     * Restore invariant in Partition such that
     * partition[i].nix == i
     */
    void scatter() {
        NodeData* in  = partition.Current();
        NodeData* out = partition.Alternate();

        thrust::for_each_n(
                thrust::device,
                thrust::make_counting_iterator<NodeIx>(0),
                numNodes,
                [in, out] __device__ (NodeIx k) {
                    const NodeData nd = in[k];
                    out[nd.nix] = nd;
                }
        );

        partition.selector ^= 1;
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
        size_t n = partition1.size();
        std::vector<Temp> degrees(n);

        auto d_ptr = thrust::device_pointer_cast(getPartitionView().Current());

        auto first = thrust::make_transform_iterator(
                d_ptr, DegreeExtractor{}
        );
        auto last = first + n;

        thrust::copy(first, last, degrees.begin());

        std::vector<EdgeIx> result(n);
        for(int i = 0; i < n; i++) {
            result[degrees[i].ix] = degrees[i].deg;
        }
        return result;
    }

    std::vector<NodeData> downloadPartition() {
        size_t n = partition1.size();
        std::vector<NodeData> pt(n);

        auto d_ptr = thrust::device_pointer_cast(partition2.data());

        thrust::copy(d_ptr, d_ptr + n, pt.begin());

        return pt;
    }

};



#endif //PAREX_DEVPARTITION_H
