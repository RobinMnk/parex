//
// Created by robin on 06.02.2026.
//

#ifndef PAREX_DEVCONSOLIDATE_H
#define PAREX_DEVCONSOLIDATE_H

#include "types.h"
#include <thrust/device_vector.h>

#include "devGraph.h"



__global__
void linkEdges_kernel(
    const NodeIx* __restrict__ neighbors,
    const NodeIx* __restrict__ nodeLookup,
    NodeIx* __restrict__ parent,
    int* __restrict__ nextLabels,
    EdgeIx numEdges,
    int* __restrict__ d_changed
) {
    EdgeIx edgeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (edgeIdx >= numEdges) return;

    NodeIx v = neighbors[edgeIdx];
    if (v == INVALID_EDGE) return;

    NodeIx u = nodeLookup[edgeIdx];

    if (nextLabels[u] != nextLabels[v]) return;

    // Direct Hooking Strategy
    while (true) {
        NodeIx rootU = parent[u];
        NodeIx rootV = parent[v];

        if (rootU == rootV) break;

        // Always try to make the smaller index the parent
        NodeIx high = max(rootU, rootV);
        NodeIx low = min(rootU, rootV);

        // atomicMin returns the OLD value. If we successfully changed it, mark as changed.
        NodeIx old = atomicMin(&parent[high], low);
        if (old != low) {
            if (d_changed) *d_changed = 1;
            break;
        }
        break;
    }
}

__global__
void compress_kernel(NodeIx* __restrict__ parent, NodeIx numNodes) {
    NodeIx i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) return;

    // Pointer jumping: parent[i] = parent[parent[i]]
    NodeIx p = parent[i];
    NodeIx gp = parent[p];
    if (p != gp) {
        parent[i] = gp;
    }
}


__global__
void assignRootAsLabel_kernel(
    NodeData* __restrict__ nodeData,
    const NodeIx* __restrict__ parent,
    NodeIx numNodes
) {
    NodeIx i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) return;

    // Only update nodes that are part of an active cluster
    if (nodeData[i].label >= 0) {
        nodeData[i].label = static_cast<int>(parent[nodeData[i].nix]);
    }
}


class ConsolidationManager {

    NodeIx numNodes;
    EdgeIx totalEdges;
    thrust::device_vector<NodeIx> parentLabels;


public:

    explicit ConsolidationManager(NodeIx n, EdgeIx totM) : numNodes(n), totalEdges(totM), parentLabels(n) {}

    void consolidate(GraphManager& gm, cub::DoubleBuffer<NodeData>& partition, thrust::device_vector<int>& nextLabels);

};

inline void ConsolidationManager::consolidate(GraphManager& gm, cub::DoubleBuffer<NodeData>& partition, thrust::device_vector<int>& nextLabels) {
    const NodeIx* neighborsPtr = thrust::raw_pointer_cast(gm.getNeighbors().data());
    const NodeIx* nodeLookupPtr = thrust::raw_pointer_cast(gm.getNodeLookup().data());
    NodeIx* parentsPtr = thrust::raw_pointer_cast(parentLabels.data());
    int* nextLabelsPtr = thrust::raw_pointer_cast(nextLabels.data());

    thrust::sequence(thrust::device, parentLabels.begin(), parentLabels.end());

    int h_changed = 1;
    int* d_changed;
    cudaMalloc(&d_changed, sizeof(int));

    int edgeBlocks = (totalEdges + threads - 1) / threads;
    int nodeBlocks = (numNodes + threads - 1) / threads;

    while (h_changed) {
        h_changed = 0;
        cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice);

        // 1. Hook nodes together
        linkEdges_kernel<<<edgeBlocks, threads>>>(neighborsPtr, nodeLookupPtr, parentsPtr, nextLabelsPtr, totalEdges, d_changed);

        // 2. Multi-pass compress to flatten trees created in the link step
        // Running this 2-3 times per link significantly speeds up convergence
        for(int i=0; i<3; ++i) {
            compress_kernel<<<nodeBlocks, threads>>>(parentsPtr, numNodes);
        }

        cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
    }

    // thrust::device_vector<int> keys(numNodes);
    // int* keysPtr = thrust::raw_pointer_cast(keys.data());

    assignRootAsLabel_kernel<<<nodeBlocks, threads>>>(partition.Current(), parentsPtr, numNodes);

    cudaFree(d_changed);
}



#endif //PAREX_DEVCONSOLIDATE_H