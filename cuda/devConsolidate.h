//
// Created by robin on 06.02.2026.
//

#ifndef PAREX_DEVCONSOLIDATE_H
#define PAREX_DEVCONSOLIDATE_H

#include "types.h"
#include <thrust/device_vector.h>
#include "devGraph.h"

__global__
inline void linkEdges_kernel(
    const LabeledNode* __restrict__ nodes,
    const NodeIx* __restrict__ neighbors,
    const NodeIx* __restrict__ nodeLookup,
    NodeIx* __restrict__ parent,
    EdgeIx numEdges,
    int* __restrict__ d_changed
) {
    EdgeIx edgeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (edgeIdx >= numEdges) return;

    NodeIx v = neighbors[edgeIdx];
    if (v == INVALID_EDGE) return;

    NodeIx u = nodeLookup[edgeIdx];

    if (nodes[u].clusterId != nodes[v].clusterId) return;

    NodeIx pU = parent[u];
    NodeIx pV = parent[v];

    if (pU != pV) {
        NodeIx high = max(pU, pV);
        NodeIx low = min(pU, pV);
        // Only one atomic needed to link components
        if (atomicMin(&parent[high], low) > low) {
            *d_changed = 1;
        }
    }
}

__global__
inline void compress_kernel(NodeIx* __restrict__ parent, NodeIx numNodes) {
    NodeIx i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) return;

    // Pointer jumping: parent[i] = parent[parent[i]]
    NodeIx p = parent[i];
    NodeIx gp = parent[p];
    if (p != gp) {
        parent[i] = gp; // Benign race condition: multiple threads might write gp
    }
}


__global__
inline void assignRootAsLabel_kernel(
    LabeledNode* __restrict__ nodes,
    const NodeIx* __restrict__ parent,
    NodeIx numNodes
) {
    NodeIx i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) return;

    // Only update nodes that are part of an active cluster
    if (nodes[i].clusterId >= 0) {
        nodes[i].clusterId = static_cast<int>(parent[nodes[i].nix]);
    }
}


class ConsolidationManager {

    NodeIx numNodes;
    EdgeIx totalEdges;
    thrust::device_vector<NodeIx> parentLabels;

    int* d_changed{};
public:

    explicit ConsolidationManager(NodeIx n, EdgeIx totM) : numNodes(n), totalEdges(totM), parentLabels(n) {
        cudaMalloc(&d_changed, sizeof(int));
    }
    ~ConsolidationManager() {
        cudaFree(d_changed);
    }

    void consolidate(GraphManager& gm, thrust::device_vector<LabeledNode>& labeledNodes);

};

inline void ConsolidationManager::consolidate(GraphManager& gm, thrust::device_vector<LabeledNode>& labeledNodes) {
    const NodeIx* neighborsPtr = thrust::raw_pointer_cast(gm.getNeighbors().data());
    const NodeIx* nodeLookupPtr = thrust::raw_pointer_cast(gm.getNodeLookup().data());
    NodeIx* parentsPtr = thrust::raw_pointer_cast(parentLabels.data());
    LabeledNode* nodes = thrust::raw_pointer_cast(labeledNodes.data());

    thrust::sequence(thrust::device, parentLabels.begin(), parentLabels.end());

    int h_changed = 1;
    size_t edgeBlocks = (totalEdges + threads - 1) / threads;
    size_t nodeBlocks = (numNodes + threads - 1) / threads;

    while (h_changed) {
        cudaMemsetAsync(d_changed, 0, sizeof(int));

        // 1. Hook nodes together
        linkEdges_kernel<<<edgeBlocks, threads>>>(nodes, neighborsPtr, nodeLookupPtr, parentsPtr, totalEdges, d_changed);

        // 2. Multi-pass compress to flatten trees created in the link step
        // Running this 2-3 times per link significantly speeds up convergence
        for(int i=0; i<3; ++i) {
            compress_kernel<<<nodeBlocks, threads>>>(parentsPtr, numNodes);
        }

        cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
    }

    assignRootAsLabel_kernel<<<nodeBlocks, threads>>>(nodes, parentsPtr, numNodes);
}



#endif //PAREX_DEVCONSOLIDATE_H