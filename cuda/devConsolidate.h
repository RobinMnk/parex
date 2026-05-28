//
// Created by robin on 06.02.2026.
//

#ifndef PAREX_DEVCONSOLIDATE_H
#define PAREX_DEVCONSOLIDATE_H

#include "types.h"
#include <thrust/device_vector.h>
#include "devGraph.h"

__global__
void linkEdges_kernel_edgeView(
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

    // TODO: Use node-focues warp parallel
    // this does not work
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
void linkEdges_kernel_nodeView(
    const NodeIx numActiveNodes,
    const LabeledNode* __restrict__ nodes,
    const NodeIx* __restrict__ neighbors,
    const EdgeIx* __restrict__ ranges,
    const EdgeIx* __restrict__ degrees,
    const label_t* __restrict__ labels,
    NodeIx* __restrict__ parent,
    int* __restrict__ d_changed
) {
    const size_t warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP;
    const size_t lane   = threadIdx.x & WARPMASK;
    if (warpId >= numActiveNodes) return;

    const unsigned int SUBMASK = BASE_SUBWARP_MASK << (threadIdx.x & ~WARPMASK);

    NodeIx nix = 0, myParent = 0;
    label_t myLabel = 0;
    EdgeIx start = 0, degree = 0;

    // Only Lane 0 issues the memory requests to the uniform addresses
    if (lane == 0) {
        const LabeledNode& lNode = nodes[warpId];
        nix     = lNode.nix;
        myLabel = lNode.clusterId;
        start   = __ldg(&ranges[nix]);
        degree  = __ldg(&degrees[nix]);
        myParent = __ldg(&parent[nix]);
    }

    // Broadcast the uniform values to all lanes
    nix     = __shfl_sync(SUBMASK, nix, 0);
    myLabel = __shfl_sync(SUBMASK, myLabel, 0);
    start   = __shfl_sync(SUBMASK, start, 0);
    degree  = __shfl_sync(SUBMASK, degree, 0);
    myParent = __shfl_sync(SUBMASK, myParent, 0);

    const EdgeIx end = start + degree;

    if (myLabel < 0) {
        printf("ERROR: This should not happen here!!\n");
        return;
    }

    for (EdgeIx j = start + lane; j < end; j += WARP) {
        const NodeIx nb = __ldg(&neighbors[j]);

        if (nb == INVALID_EDGE) continue;

        const label_t nbLabel = __ldg(&labels[nb]);
        if (nbLabel != myLabel) continue;

        const NodeIx otherParent = __ldg(&parent[nb]);

        if (myParent != otherParent) {
            const NodeIx high = max(myParent, otherParent);
            const NodeIx low = min(myParent, otherParent);

            if (atomicMin(&parent[high], low) > low) {
                *d_changed = 1;
            }
        }
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
        parent[i] = gp; // Benign race condition: multiple threads might write gp
    }
}


__global__
void assignRootAsLabel_kernel(
    LabeledNode* __restrict__ nodes,
    label_t* __restrict__ labels,
    const NodeIx* __restrict__ parent,
    NodeIx numNodes
) {
    const NodeIx i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) return;

    // Only update nodes that are part of an active cluster
    if (nodes[i].clusterId >= 0) {
        label_t l = static_cast<label_t>(parent[nodes[i].nix]);
        nodes[i].clusterId = l;
        labels[nodes[i].nix] = l;
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

    void consolidate(GraphManager& gm, thrust::device_vector<LabeledNode>& labeledNodes, thrust::device_vector<label_t>& labels, NodeIx numActiveNodes);

};

inline void ConsolidationManager::consolidate(
    GraphManager& gm,
    thrust::device_vector<LabeledNode>& labeledNodes,
    thrust::device_vector<label_t>& labels,
    NodeIx numActiveNodes
) {
    const NodeIx* neighborsPtr = thrust::raw_pointer_cast(gm.getNeighbors().data());
    const NodeIx* rangesPtr = thrust::raw_pointer_cast(gm.getRanges().data());
    const EdgeIx* degreesPtr = thrust::raw_pointer_cast(gm.getDegrees().data());
    label_t* labelsPtr = thrust::raw_pointer_cast(labels.data());
    NodeIx* parentsPtr = thrust::raw_pointer_cast(parentLabels.data());
    LabeledNode* nodes = thrust::raw_pointer_cast(labeledNodes.data());

    thrust::sequence(thrust::device, parentLabels.begin(), parentLabels.begin() + numNodes);

    int h_changed = 1;
    size_t activeBlocks = getGridSizeWarpParallel(numActiveNodes);
    size_t nodeBlocks = getGridSize(numNodes);

    while (h_changed) {
        cudaMemsetAsync(d_changed, 0, sizeof(int));

        // 1. Hook nodes together
        linkEdges_kernel_nodeView<<<activeBlocks, threads>>>(numActiveNodes, nodes, neighborsPtr, rangesPtr, degreesPtr, labelsPtr, parentsPtr, d_changed);

        // 2. Multi-pass compress to flatten trees created in the link step
        // Running this 2-3 times per link significantly speeds up convergence
        for(int i=0; i<3; ++i) {
            compress_kernel<<<nodeBlocks, threads>>>(parentsPtr, numNodes);
        }

        cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
    }

    size_t assignBlocks = getGridSize(numActiveNodes);
    assignRootAsLabel_kernel<<<assignBlocks, threads>>>(nodes, labelsPtr, parentsPtr, numActiveNodes);
}


#endif //PAREX_DEVCONSOLIDATE_H