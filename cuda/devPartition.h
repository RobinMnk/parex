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
            start,
            end - start
        };
    }
};


class PartitionManager {
public:
    thrust::device_vector<NodeData> partition1;
    thrust::device_vector<NodeData> partition2;
    cub::DoubleBuffer<NodeData> partition;

    NodeIx numActiveClusters;
    thrust::device_vector<NodeIx> activeLabels;
    thrust::device_vector<EdgeIx> volumes;


public:

    explicit PartitionManager(GraphManager& gm) :
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
