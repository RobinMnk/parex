//
// Created by robin on 14.01.2026.
//

#include "device.h"

__host__ __device__
inline void DevGraph::deactivateEdge(EdgeIx idx) const {
    // to deactivate an edge we "redirect it" to point to a garbage-node with index 2 * numEdges
//    neighbors[idx] = 2 * numEdges;
}

__host__ __device__
inline void DevGraph::handleActiveDegrees(NodeIx idx) const {
    NodeUpdate pair = nodeUpdateBuffer[idx];
//    active_degrees[pair.nix] -= pair.diff;
}

//__host__ __device__
//inline void RandomWalkManager::computeNodeVal(NodeIx idx, const EdgeIx* active_degrees) const {
//    node_val[idx] = old_dist[idx] / active_degrees[idx];
//}
