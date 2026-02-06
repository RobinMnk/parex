//
// Created by robin on 06.02.2026.
//

#ifndef PAREX_DEVCONSOLIDATE_H
#define PAREX_DEVCONSOLIDATE_H

#include "types.h"
#include <thrust/device_vector.h>

#include "devGraph.h"



// __global__
// void linkEdges_kernel(
//     const NodeIx* neighbors,
//     NodeIx* parent,
//     const EdgeIx* offsets,
//     EdgeIx numEdges,
//     NodeIx numNodes)
// {
//     EdgeIx edgeIdx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (edgeIdx >= numEdges) return;
//
//     NodeIx v = neighbors[edgeIdx];
//     if (v == INVALID_EDGE) return;
//
//     // Fast way to find source 'u' from edgeIdx without extra O(E) array
//
//     // Standard Union-Find Link
//     NodeIx rootU = u;
//     NodeIx rootV = v;
//
//     // Find roots
//     while (parent[rootU] != rootU) rootU = parent[rootU];
//     while (parent[rootV] != rootV) rootV = parent[rootV];
//
//     if (rootU != rootV) {
//         // Atomic link: smaller index becomes the parent of the larger index
//         atomicMin(&parent[max(rootU, rootV)], min(rootU, rootV));
//     }
// }


class ConsolidationManager {

    NodeIx numNodes;
    thrust::device_vector<int> parentLabels;


public:

    explicit ConsolidationManager(NodeIx n) : numNodes(n), parentLabels(n) {}

    void consolidate(const GraphManager& gm, cub::DoubleBuffer<NodeData>& partition);

};

inline void ConsolidationManager::consolidate(const GraphManager& gm, cub::DoubleBuffer<NodeData>& partition) {



}



#endif //PAREX_DEVCONSOLIDATE_H