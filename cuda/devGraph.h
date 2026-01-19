//
// Created by robin on 19.01.2026.
//

#ifndef PAREX_DEVGRAPH_H
#define PAREX_DEVGRAPH_H

#include "core/definitions.h"

struct DevGraph {
    const NodeIx numNodes;
    const EdgeIx numEdges;

    // Graph
    NodeIx* neighbors;          // size: 2 * numEdges
    NodeIx* ranges;             // size: numNodes+1
    EdgeIx* active_degrees;     // size: numNodes

    // Buffers for Updates
    EdgeIx* edgeDeletionBuffer;
    NodeUpdate* nodeUpdateBuffer;

    __host__ __device__
    inline void deactivateEdge(EdgeIx idx) const;

    __host__ __device__
    inline void handleActiveDegrees(NodeIx idx) const;
};

__global__
void deactivateEdgeKernel(DevGraph gr, EdgeIx numEdgeDeletions) {
    NodeIx idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < numEdgeDeletions) gr.deactivateEdge(idx);
}

__global__
void degreeKernel(DevGraph gr, NodeIx numUpdates) {
    NodeIx idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numUpdates) gr.handleActiveDegrees(idx);
}



#endif //PAREX_DEVGRAPH_H
