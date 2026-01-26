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
            start
        };
    }
};


class DevPartition {
    thrust::device_vector<NodeData> partition;

    NodeIx numActiveClusters;
    thrust::device_vector<NodeIx> activeLabels;
    thrust::device_vector<EdgeIx> volumes;


public:

    explicit DevPartition(GraphManager& gm) : partition(gm.n), numActiveClusters{1}, activeLabels(gm.n, 0), volumes(gm.n, 1) {
        thrust::transform(
            thrust::make_counting_iterator<NodeIx>(0),
            thrust::make_counting_iterator(gm.n),
            partition.begin(),
            InitFunctor(thrust::raw_pointer_cast(gm.getRanges().data()))
        );
    }

    thrust::device_vector<NodeData>& getPartition() {
        return partition;
    }

};



#endif //PAREX_DEVPARTITION_H
