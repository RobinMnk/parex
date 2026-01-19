//
// Created by robin on 14.01.2026.
//

#include "graph_manager.h"

__host__ __device__
inline void DevGraph::deactivateEdge(EdgeIx idx) const {
    // to deactivate an edge we "redirect it" to point to a garbage-node with index 2 * numEdges
    neighbors[idx] = 2 * numEdges;
}

__host__ __device__
inline void DevGraph::handleActiveDegrees(NodeIx idx) const {
    NodeUpdate pair = nodeUpdateBuffer[idx];
    active_degrees[pair.nix] -= pair.diff;
}

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
